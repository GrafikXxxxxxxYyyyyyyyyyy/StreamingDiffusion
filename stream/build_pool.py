"""Сборка пула опорных кадров для worker_long.py --pool-dir.

Каждый кадр пула — ровно одно поколение от исходной фотографии: генерируем
короткие дубли от неё с разными промптами и сидами и вынимаем кадры. Поэтому
пул не деградирует, сколько бы стрим ни крутился, а разные ракурсы и позы
убирают ощущение петли.
"""
import argparse, os, time
import numpy as np, torch, cv2
from PIL import Image
from diffusers import (LTXI2VLongMultiPromptPipeline, LTXVideoTransformer3DModel,
                       AutoencoderKLLTXVideo, FlowMatchEulerDiscreteScheduler)
from transformers import T5EncoderModel, T5Tokenizer
from diffusers.utils import load_image

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = "https://huggingface.co/Lightricks/LTX-Video/blob/main/ltxv-2b-0.9.8-distilled.safetensors"
T5_REPO, SCHED_REPO = "Lightricks/LTX-Video", "Lightricks/LTX-Video-0.9.8-13B-distilled"

BASE = ("Webcam stream of a young woman gamer with headphones at her desk, fixed framing, "
        "she stays centered in frame, dark room lit by pink and purple LED strips, "
        "neon heart sign on the wall, gaming chair, plush toys on the shelf")

# разные действия и планы — чтобы дубли не повторяли друг друга
VARIANTS = [
    "medium shot from the front, she smiles and talks to the camera",
    "medium shot, she laughs and leans back in her chair",
    "medium close-up, she leans forward toward the camera and points",
    "medium shot, she waves at the chat with one hand",
    "medium shot, she adjusts her headphones with both hands",
    "medium shot, she drinks from a mug and smiles",
    "medium shot, she looks aside at a second monitor, then back",
    "medium shot, she gestures with both hands while explaining",
    "close-up of her face, she winks and grins",
    "medium shot, she gives a thumbs up to the camera",
    "medium shot, she tilts her head and looks surprised",
    "slightly wider shot showing the desk, she types and looks at the screen",
]


def hist(rgb):
    a = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h = cv2.calcHist([a], [0, 1], None, [36, 32], [0, 180, 0, 256])
    return cv2.normalize(h, h).flatten()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", default=os.path.join(HERE, "start.jpg"))
    p.add_argument("--out", default=os.path.join(HERE, "pool"))
    p.add_argument("--width", type=int, default=576)
    p.add_argument("--height", type=int, default=768)
    p.add_argument("--frames", type=int, default=161, help="длина дубля, 8k+1")
    p.add_argument("--per-variant", type=int, default=5, help="кадров с дубля")
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--min-sim", type=float, default=0.45,
                   help="порог похожести на исходник: отсекает уехавшие кадры")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    DT = torch.bfloat16
    pipe = LTXI2VLongMultiPromptPipeline(
        transformer=LTXVideoTransformer3DModel.from_single_file(CKPT, dtype=DT),
        vae=AutoencoderKLLTXVideo.from_single_file(CKPT, dtype=DT),
        text_encoder=T5EncoderModel.from_pretrained(T5_REPO, subfolder="text_encoder", dtype=DT),
        tokenizer=T5Tokenizer.from_pretrained(T5_REPO, subfolder="tokenizer"),
        scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(SCHED_REPO, subfolder="scheduler"),
    ).to("cuda")
    img = load_image(args.image)
    ref = hist(np.asarray(img.convert("RGB").resize((args.width, args.height))))
    print(f"модель загружена, вариантов: {len(VARIANTS)}", flush=True)

    kept = dropped = 0
    for vi, act in enumerate(VARIANTS):
        t0 = time.time()
        lat = pipe(prompt=f"{BASE}, {act}", cond_image=img, width=args.width, height=args.height,
                   num_frames=args.frames, frame_rate=24, num_inference_steps=args.steps,
                   guidance_scale=1.0, seed=1000 + vi, output_type="latent").frames
        vid = pipe.vae_decode_tiled(lat, decode_timestep=0.05, decode_noise_scale=0.025,
                                    horizontal_tiles=4, vertical_tiles=4, overlap=3,
                                    output_type="pt")
        x = ((vid[0].clamp(-1, 1) + 1) * 127.5).to(torch.uint8)
        arr = x.permute(1, 2, 3, 0).contiguous().cpu().numpy()
        del vid, x, lat
        torch.cuda.empty_cache()

        idx = np.linspace(len(arr) // 5, len(arr) - 1, args.per_variant).astype(int)
        for k in idx:
            sim = cv2.compareHist(ref, hist(arr[k]), cv2.HISTCMP_CORREL)
            if sim < args.min_sim:          # кадр уехал от исходника — в пул не берём
                dropped += 1
                continue
            Image.fromarray(arr[k]).save(os.path.join(args.out, f"v{vi:02d}_f{k:04d}.jpg"), quality=95)
            kept += 1
        print(f"вариант {vi + 1:2d}/{len(VARIANTS)} за {time.time() - t0:4.1f}s  "
              f"в пуле {kept}, отброшено {dropped}", flush=True)

    print(f"готово: {kept} кадров в {args.out}")


if __name__ == "__main__":
    main()
