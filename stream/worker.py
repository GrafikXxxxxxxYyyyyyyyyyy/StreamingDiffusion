"""Бесконечная генерация LTX-Video → ffmpeg → HLS.

Шаг 1 из docs/design.md: цикл генерации кусков, склеенных по хвосту, непрерывно
подаётся в один долгоживущий ffmpeg, который нарезает HLS. Рядом поднимается
http-сервер со страницей-плеером.
"""
import argparse, json, os, queue, subprocess, threading, time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import (LTXConditionPipeline, LTXVideoTransformer3DModel,
                       AutoencoderKLLTXVideo, FlowMatchEulerDiscreteScheduler)
from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition
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
    pipe = LTXConditionPipeline(
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
        # -re заставляет ffmpeg читать stdin в реальном времени: это и есть
        # обратное давление, из-за которого генератор не убегает вперёд
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


class DriftCorrector:
    """Замкнутая петля против уезжающей камеры.

    Разомкнутая стабилизация здесь не работает: если править только картинку
    на выходе, а в условие отдавать неисправленный хвост, модель продолжает
    уезжать от уже уехавшего кадра и поправка растёт без предела (замеряли
    230 px за 25 с). Поэтому выравниваем кусок целиком и хвост тоже —
    следующий кусок стартует с выровненного кадрирования, и дрейф не копится.
    Компенсировать приходится движение ровно одного куска, а это единицы процентов.
    """

    def __init__(self, width, height, crop=0.05, scale=0.5, every=2):
        self.w, self.h = width, height
        self.center = (width / 2, height / 2)
        self.zoom = 1.0 / (1.0 - 2 * crop)
        self.limit = (crop * 0.9 * width, crop * 0.9 * height)
        self.scale, self.every = scale, every

    def _gray(self, rgb):
        g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        return cv2.resize(g, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)

    def _step(self, prev, cur):
        pts = cv2.goodFeaturesToTrack(prev, maxCorners=200, qualityLevel=0.01,
                                      minDistance=8, blockSize=7)
        if pts is None or len(pts) < 12:
            return 0.0, 0.0, 0.0, 1.0
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, cur, pts, None)
        ok = st.ravel() == 1
        if ok.sum() < 12:
            return 0.0, 0.0, 0.0, 1.0
        M, _ = cv2.estimateAffinePartial2D(pts[ok], nxt[ok], method=cv2.RANSAC)
        if M is None:
            return 0.0, 0.0, 0.0, 1.0
        return (float(M[0, 2]) / self.scale, float(M[1, 2]) / self.scale,
                float(np.arctan2(M[1, 0], M[0, 0])), float(np.hypot(M[0, 0], M[1, 0])) or 1.0)

    def correct(self, frames):
        """Выравнивает все кадры куска по его первому кадру."""
        dx = dy = da = 0.0
        ds = 1.0
        prev = self._gray(frames[0])
        out = [self._warp(frames[0], 0, 0, 0, 1.0)]
        for k in range(1, len(frames)):
            if k % self.every == 0 or k == len(frames) - 1:
                cur = self._gray(frames[k])
                sx, sy, sa, ss = self._step(prev, cur)
                dx += sx; dy += sy; da += sa; ds *= ss
                prev = cur
            out.append(self._warp(frames[k], dx, dy, da, ds))
        return out

    def _warp(self, rgb, dx, dy, da, ds):
        dx = float(np.clip(dx, -self.limit[0], self.limit[0]))
        dy = float(np.clip(dy, -self.limit[1], self.limit[1]))
        ds = float(np.clip(ds, 0.9, 1.12))
        T = np.array([[1, 0, -dx], [0, 1, -dy], [0, 0, 1]], dtype=np.float64)
        R = np.vstack([cv2.getRotationMatrix2D(self.center, np.degrees(da),
                                               self.zoom / ds), [0, 0, 1]])
        return cv2.warpAffine(rgb, (R @ T)[:2], (self.w, self.h),
                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def crossfade(prev_frame, frames, n):
    """Растворяет начало новой сцены из последнего показанного кадра."""
    if prev_frame is None or n <= 0:
        return frames
    a = np.asarray(prev_frame, dtype=np.float32)
    n = min(n, len(frames))
    for k in range(n):
        w = (k + 1) / (n + 1)
        b = np.asarray(frames[k], dtype=np.float32)
        frames[k] = np.clip(a * (1 - w) + b * w, 0, 255).astype(np.uint8)
    return frames


def producer(pipe, args, start_image, q, stop, corrector):
    """Генерирует куски, каждый продолжает хвост предыдущего."""
    tail, i, last_frame = None, 0, None
    while not stop.is_set():
        # смена сцены — лекарство от наезда камеры, который копится от куска к куску
        new_scene = tail is None or (args.reset_every and args.reset_mode != "off"
                                     and i % args.reset_every == 0)
        if tail is None:
            cond = [LTXVideoCondition(image=start_image, frame_index=0)]
        elif new_scene and args.reset_mode == "text":
            cond = None          # чистый text-to-video: каждый раз новая сцена
        elif new_scene:
            cond = [LTXVideoCondition(image=start_image, frame_index=0)]
        else:
            cond = [LTXVideoCondition(video=tail, frame_index=0)]
        continues_tail = cond is not None and cond[0].video is not None

        with state_lock:
            if prompt_box["next"] is not None:
                state["request"] = prompt_box["next"]
                prompt_box["next"] = None
            base, req = state["base"], state["request"]
            # база держит сцену и неподвижную камеру; запрос лишь дополняет её
            prompt = f"{base}, {req}" if req else base
            state["prompt"] = prompt

        t0 = time.time()
        out = pipe(
            conditions=cond, prompt=prompt, negative_prompt=None,
            width=args.width, height=args.height, num_frames=args.chunk,
            num_inference_steps=args.steps, guidance_scale=1.0,
            generator=torch.Generator("cuda").manual_seed(args.seed + i),
        ).frames[0]
        G = time.time() - t0

        arrs = [np.asarray(f, dtype=np.uint8) for f in out]
        if corrector is not None:
            arrs = corrector.correct(arrs)          # и картинка, и хвост — выровненные
        new = arrs[args.overlap:] if continues_tail else arrs
        if new_scene and last_frame is not None and args.crossfade:
            new = crossfade(last_frame, list(new), args.crossfade)
        tail = [Image.fromarray(a) for a in arrs[-args.overlap:]]
        last_frame = new[-1]
        i += 1

        with state_lock:
            state["chunks"], state["last_G"] = i, round(G, 2)
            if new_scene:
                state["scene"] += 1
        L = len(new) / args.fps
        print(f"кусок {i:4d}  G={G:5.2f}s  L={L:4.2f}s  G/L={G / L:4.2f}"
              f"{'  [новая сцена]' if new_scene else ''}", flush=True)

        q.put((list(new), new_scene))


def writer(ff, q, stop):
    """Отдаёт кадры в ffmpeg; блокируется на его темпе из-за -re."""
    while not stop.is_set():
        try:
            frames, new_scene = q.get(timeout=1)
        except queue.Empty:
            with state_lock:
                state["stalls"] += 1        # буфер пуст — генератор не успевает
            continue
        with state_lock:
            state["buffer"] = q.qsize()
        data = b"".join(np.ascontiguousarray(f).tobytes() for f in frames)
        try:
            ff.stdin.write(data)
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
    p.add_argument("--chunk", type=int, default=97)      # 8k+1
    p.add_argument("--overlap", type=int, default=9)     # 8k+1
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--buffer", type=int, default=2, help="кусков в запасе")
    p.add_argument("--reset-every", type=int, default=15, help="каждые N кусков менять сцену")
    p.add_argument("--reset-mode", choices=("text", "image", "off"), default="text",
                   help="text — новая сцена из промпта; image — вернуться к стартовому кадру; off — не менять")
    p.add_argument("--stabilize", type=int, default=1, help="1 — гасить дрейф камеры (замкнутая петля)")
    p.add_argument("--stab-crop", type=float, default=0.07, help="запас по краям под стабилизацию")
    p.add_argument("--crossfade", type=int, default=10, help="кадров растворения при смене сцены")
    p.add_argument("--port", type=int, default=17070)
    p.add_argument("--out", default=os.path.join(HERE, "hls"))
    args = p.parse_args()
    for name, v in (("chunk", args.chunk), ("overlap", args.overlap)):
        if (v - 1) % 8 != 0:
            raise SystemExit(f"--{name} должен быть вида 8k+1, получено {v}")

    with open(os.path.join(HERE, "worker.pid"), "w") as f:
        f.write(str(os.getpid()))          # чтобы останавливать по pid, а не шаблоном pkill
    if args.prompt and (r := check_prompt(args.prompt, limit=1000)):
        raise SystemExit(f"базовый промпт не прошёл фильтр: {r}")
    state["base"] = args.prompt
    state["prompt"] = args.prompt
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
    corrector = (DriftCorrector(args.width, args.height, crop=args.stab_crop)
                 if args.stabilize else None)
    tp = threading.Thread(target=producer, args=(pipe, args, start_image, q, stop, corrector),
                          daemon=True)
    tp.start()

    while q.qsize() < min(args.buffer, 1):        # ждём первый кусок
        time.sleep(0.2)
    print("первый кусок готов, стрим пошёл", flush=True)
    tw = threading.Thread(target=writer, args=(ff, q, stop), daemon=True)
    tw.start()

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
