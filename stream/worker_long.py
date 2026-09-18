"""Бесконечный стрим на LTXI2VLongMultiPromptPipeline → ffmpeg → HLS.

Отличие от worker.py: вместо цепочки 4-секундных кусков, склеенных по хвосту,
генерируем батчи по ~20 секунд. Пайплайн внутри сам режет их на окна со
скользящим перекрытием, жёстко держит первый кадр маской токенов и выравнивает
статистику через AdaIN — поэтому сцена не уплывает, в отличие от worker.py,
где за две минуты от неё не оставалось ничего.

Следующий батч стартует с последнего кадра предыдущего, так что стык незаметен.
"""
import argparse, json, os, queue, random, subprocess, threading, time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import (LTXI2VLongMultiPromptPipeline, LTXVideoTransformer3DModel,
                       AutoencoderKLLTXVideo, FlowMatchEulerDiscreteScheduler)
from transformers import T5EncoderModel, T5Tokenizer
from diffusers.utils import load_image

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = "https://huggingface.co/Lightricks/LTX-Video/blob/main/ltxv-2b-0.9.8-distilled.safetensors"
T5_REPO = "Lightricks/LTX-Video"
SCHED_REPO = "Lightricks/LTX-Video-0.9.8-13B-distilled"

state = {"chunks": 0, "last_G": None, "buffer": 0, "started": time.time(),
         "base": "", "request": "", "prompt": "", "scene": 0, "fps": 24,
         "stalls": 0, "rejected": 0}
state_lock = threading.Lock()
prompt_box = {"next": None}

# Простой фильтр промптов. Список слов обходится элементарно и это не настоящая
# модерация — в docs/design.md на эту роль заложена маленькая LLM. Здесь он нужен,
# чтобы поле ввода не было открытой дверью в модель.
BANNED = (
    "naked", "nude", "nsfw", "topless", "undress", "undressed", "underwear", "lingerie",
    "bikini", "sex", "sexual", "porn", "erotic", "explicit", "nipple", "genital",
    "голая", "голый", "обнаж", "раздет", "секс", "порно", "эрот", "без одежды",
    "child", "kid", "teen", "loli", "ребён", "ребен", "детск", "школьниц",
    "gore", "blood", "kill", "corpse", "кровь", "труп", "убий",
)
MAX_PROMPT = 300


def check_prompt(text, limit=MAX_PROMPT):
    """Возвращает причину отказа или None, если промпт допустим."""
    t = " ".join(text.lower().split())
    if not t:
        return "пустой запрос"
    if len(t) > limit:
        return f"слишком длинный запрос (>{limit} символов)"
    for w in BANNED:
        if w in t:
            return "запрос отклонён фильтром"
    return None




def build_pipe(dtype=torch.bfloat16):
    pipe = LTXI2VLongMultiPromptPipeline(
        transformer=LTXVideoTransformer3DModel.from_single_file(CKPT, dtype=dtype),
        vae=AutoencoderKLLTXVideo.from_single_file(CKPT, dtype=dtype),
        text_encoder=T5EncoderModel.from_pretrained(T5_REPO, subfolder="text_encoder", dtype=dtype),
        tokenizer=T5Tokenizer.from_pretrained(T5_REPO, subfolder="tokenizer"),
        scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(SCHED_REPO, subfolder="scheduler"),
    )
    return pipe.to("cuda")


def start_ffmpeg(out_dir, width, height, fps):
    os.makedirs(out_dir, exist_ok=True)
    for f in os.listdir(out_dir):
        if f.endswith((".ts", ".m3u8")):
            os.remove(os.path.join(out_dir, f))
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "warning", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
        "-r", str(fps), "-re", "-i", "-",
        "-c:v", "libx264", "-preset", "veryfast", "-tune", "zerolatency",
        "-pix_fmt", "yuv420p", "-g", str(fps * 2), "-sc_threshold", "0",
        "-f", "hls", "-hls_time", "2", "-hls_list_size", "6",
        "-hls_flags", "delete_segments+independent_segments",
        "-hls_segment_filename", os.path.join(out_dir, "seg_%05d.ts"),
        os.path.join(out_dir, "stream.m3u8"),
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def encode_anchor(pipe, image, width, height):
    """Латенты опорного кадра для negative_index_latents.

    По умолчанию пайплайн берёт якорем те же cond_latents, то есть последний кадр
    предыдущего батча — уже подпорченный. Якорь деградирует вместе с картинкой,
    и удерживать сцену становится нечем. Подставляем вместо него чистый исходник:
    он участвует в обусловливании, но в выход не попадает.
    """
    img = pipe.video_processor.preprocess(image, height=height, width=width)
    img = img.to(device="cuda", dtype=pipe.vae.dtype)
    enc = pipe.vae.encode(img.unsqueeze(2))
    lat = enc.latent_dist.mode() if hasattr(enc, "latent_dist") else enc.latents
    return pipe._normalize_latents(lat.to(torch.float32), pipe.vae.latents_mean,
                                   pipe.vae.latents_std, pipe.vae.config.scaling_factor)


def color_stats(rgb01):
    """Среднее и разброс по каналам. rgb01: [3, ...] в [0, 1]."""
    flat = rgb01.reshape(3, -1).float()
    return flat.mean(dim=1), flat.std(dim=1).clamp_min(1e-4)


def match_color(x, ref, factor):
    """Подтягивает статистику батча к опорной.

    Без этого получается положительная обратная связь: каждый батч стартует с
    последнего кадра предыдущего, тот чуть насыщеннее — и за 15 батчей картинка
    уходит в кислоту. factor=1 держит цвет жёстко, 0 отключает.
    """
    m, sd = color_stats(x)
    rm, rsd = ref
    shape = (3,) + (1,) * (x.dim() - 1)
    fixed = (x - m.view(shape)) * (rsd / sd).view(shape) + rm.view(shape)
    return (x * (1 - factor) + fixed * factor).clamp(0, 1)


def producer(pipe, args, start_image, q, stop):
    """Генерирует батчи по ~batch_seconds; каждый стартует с последнего кадра прошлого."""
    cond, i = start_image, 0
    anchor = (encode_anchor(pipe, start_image, args.width, args.height)
              if args.anchor > 0 else None)
    pool, last_frame = [], None
    if args.pool_dir and os.path.isdir(args.pool_dir):
        files = sorted(f for f in os.listdir(args.pool_dir) if f.endswith((".jpg", ".png")))
        pool = [Image.open(os.path.join(args.pool_dir, f)).convert("RGB") for f in files]
        print(f"пул загружен с диска: {len(pool)} кадров", flush=True)
    ref_stats = None
    if args.color_fix > 0:
        ref = torch.from_numpy(np.asarray(start_image.convert("RGB"), dtype=np.float32) / 255.0)
        ref_stats = color_stats(ref.permute(2, 0, 1).to("cuda"))
    while not stop.is_set():
        with state_lock:
            if prompt_box["next"] is not None:
                state["request"] = prompt_box["next"]
                prompt_box["next"] = None
            base, req = state["base"], state["request"]
            prompt = f"{base}, {req}" if req else base
            state["prompt"] = prompt

        t0 = time.time()
        lat = pipe(prompt=prompt, cond_image=cond, width=args.width, height=args.height,
                   num_frames=args.frames, frame_rate=args.fps,
                   num_inference_steps=args.steps, guidance_scale=1.0,
                   negative_index_latents=anchor, negative_index_strength=args.anchor,
                   seed=args.seed + i, output_type="latent").frames
        t_den = time.time() - t0

        t0 = time.time()
        # кадры уходят тензором: конвертация 480 картинок в PIL стоила бы 9 секунд
        video = pipe.vae_decode_tiled(lat, decode_timestep=0.05, decode_noise_scale=0.025,
                                      horizontal_tiles=args.decode_tiles,
                                      vertical_tiles=args.decode_tiles, overlap=3,
                                      output_type="pt")
        # при output_type="pt" возвращается сырой тензор [B, C, T, H, W] в [-1, 1],
        # постпроцессор не применяется. В uint8 переводим на карте: "np" тащил бы
        # 2,5 ГБ float32 через шину вместо 640 МБ.
        x = (video[0].clamp(-1, 1) + 1) / 2                         # [3, T, H, W] в [0, 1]
        if ref_stats is not None and args.color_fix > 0:
            x = match_color(x, ref_stats, args.color_fix)
        x = (x * 255).to(torch.uint8)
        arr = x.permute(1, 2, 3, 0).contiguous().cpu().numpy()      # [T, H, W, 3]
        del video, x
        if last_frame is not None and args.crossfade > 0:
            # дубли стартуют с разных кадров пула, поэтому стык — это склейка;
            # растворение прячет её, иначе на коротких дублях эфир рубит
            n = min(args.crossfade, len(arr))
            a = last_frame.astype(np.float32)
            for k in range(n):
                w = (k + 1) / (n + 1)
                arr[k] = np.clip(a * (1 - w) + arr[k].astype(np.float32) * w, 0, 255).astype(np.uint8)
        last_frame = arr[-1]
        t_dec = time.time() - t0
        # feedback=1 — стык со следующим батчем, но выход модели идёт ей же на вход,
        # и артефакты копятся до коллапса. feedback=0 — каждый батч от чистого
        # исходника: деградации накапливаться негде, ценой склейки раз в 20 секунд.
        if args.feedback:
            cond = Image.fromarray(arr[-1])
        elif pool:
            # пул набирается из ПЕРВОГО батча, обусловленного чистым исходником,
            # поэтому все кадры в нём ровно одно поколение от оригинала и не
            # деградируют. Разные позы дают разнообразие без петли.
            cond = random.choice(pool)
        else:
            cond = start_image
        if not args.feedback and not pool and args.pool_size > 0:
            idx = np.linspace(len(arr) // 6, len(arr) - 1, args.pool_size).astype(int)
            pool.extend(Image.fromarray(arr[k]) for k in idx)
            print(f"пул опорных кадров собран: {len(pool)}", flush=True)
        i += 1

        G, L = t_den + t_dec, len(arr) / args.fps
        with state_lock:
            state["chunks"], state["last_G"] = i, round(G, 2)
        print(f"батч {i:4d}  денойз={t_den:5.1f}s + декод={t_dec:4.1f}s = {G:5.1f}s  "
              f"L={L:5.1f}s  G/L={G / L:4.2f}", flush=True)
        q.put(arr)


def writer(ff, q, stop):
    while not stop.is_set():
        try:
            arr = q.get(timeout=1)
        except queue.Empty:
            with state_lock:
                state["stalls"] += 1
            continue
        with state_lock:
            state["buffer"] = q.qsize()
        try:
            ff.stdin.write(np.ascontiguousarray(arr).tobytes())
            ff.stdin.flush()
        except (BrokenPipeError, ValueError):
            break


class Handler(SimpleHTTPRequestHandler):
    extensions_map = {**SimpleHTTPRequestHandler.extensions_map,
                      ".m3u8": "application/vnd.apple.mpegurl", ".ts": "video/mp2t"}

    def do_GET(self):
        if self.path.startswith("/status"):
            with state_lock:
                body = json.dumps({**state, "uptime": round(time.time() - state["started"])},
                                  ensure_ascii=False).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path.startswith("/prompt?"):
            from urllib.parse import unquote, urlparse, parse_qs
            text = unquote((parse_qs(urlparse(self.path).query).get("text") or [""])[0])
            reason = check_prompt(text)
            if reason:
                body = json.dumps({"ok": False, "reason": reason}, ensure_ascii=False).encode()
                self.send_response(422)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers(); self.wfile.write(body)
                with state_lock:
                    state["rejected"] += 1
                print(f"промпт отклонён ({reason}): {text[:80]!r}", flush=True)
                return
            with state_lock:
                prompt_box["next"] = text
            self.send_response(204); self.end_headers()
            return
        super().do_GET()

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, *a):
        pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", default=os.path.join(HERE, "start.jpg"))
    p.add_argument("--prompt", default="")
    p.add_argument("--width", type=int, default=576)
    p.add_argument("--height", type=int, default=768)
    p.add_argument("--frames", type=int, default=481, help="кадров в батче, вида 8k+1")
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--buffer", type=int, default=2)
    p.add_argument("--crossfade", type=int, default=12, help="кадров растворения на стыке дублей")
    p.add_argument("--pool-dir", default="", help="папка с готовым пулом (build_pool.py)")
    p.add_argument("--pool-size", type=int, default=8,
                   help="сколько опорных кадров набрать из первого батча (при --feedback 0)")
    p.add_argument("--anchor", type=float, default=1.0,
                   help="сила якоря на исходный кадр (negative_index), 0 — выключить")
    p.add_argument("--feedback", type=int, default=1,
                   help="1 — следующий батч от последнего кадра; 0 — всегда от исходника")
    p.add_argument("--color-fix", type=float, default=0.8,
                   help="сила подтяжки цвета к опорному кадру, 0 — выключить")
    p.add_argument("--decode-tiles", type=int, default=4, help="2 быстрее, но 28 ГБ пик")
    p.add_argument("--port", type=int, default=17070)
    p.add_argument("--out", default=os.path.join(HERE, "hls"))
    args = p.parse_args()
    if (args.frames - 1) % 8 != 0:
        raise SystemExit(f"--frames должен быть вида 8k+1, получено {args.frames}")
    if args.prompt and (r := check_prompt(args.prompt, limit=1000)):
        raise SystemExit(f"базовый промпт не прошёл фильтр: {r}")

    with open(os.path.join(HERE, "worker.pid"), "w") as f:
        f.write(str(os.getpid()))
    state["base"] = state["prompt"] = args.prompt
    state["fps"] = args.fps
    start_image = load_image(args.image)

    print("загружаю модель…", flush=True)
    pipe = build_pipe()
    print(f"готово, VRAM {torch.cuda.memory_allocated() / 1e9:.1f} ГБ", flush=True)

    os.makedirs(args.out, exist_ok=True)
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), partial(Handler, directory=args.out))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"http://127.0.0.1:{args.port}/ — страница плеера", flush=True)

    ff = start_ffmpeg(args.out, args.width, args.height, args.fps)
    q, stop = queue.Queue(maxsize=args.buffer), threading.Event()
    threading.Thread(target=producer, args=(pipe, args, start_image, q, stop), daemon=True).start()

    while q.qsize() < 1:
        time.sleep(0.5)
    print("первый батч готов, стрим пошёл", flush=True)
    threading.Thread(target=writer, args=(ff, q, stop), daemon=True).start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stop.set()
        try:
            ff.stdin.close()
        except Exception:
            pass
        ff.wait(timeout=10)


if __name__ == "__main__":
    main()
