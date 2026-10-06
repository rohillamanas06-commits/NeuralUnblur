from __future__ import annotations

import argparse
import hmac
import io
import logging
import math
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from PIL import Image, ImageOps
from starlette.concurrency import run_in_threadpool
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("deblur")


def _get(name: str, default, cast=str):
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return cast(raw.strip())
    except ValueError:
        log.warning("Bad value for %s=%r, using default %r", name, raw, default)
        return default


HOST = _get("HOST", "0.0.0.0")
PORT = _get("PORT", 8000, int)
ALLOWED_ORIGINS = [o.strip() for o in _get("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
ADMIN_TOKEN = _get("ADMIN_TOKEN", "")
MAX_UPLOAD_MB = _get("MAX_UPLOAD_MB", 10, int)
MAX_IMAGE_PIXELS = _get("MAX_IMAGE_PIXELS", 16_000_000, int)

MODEL_PATH = _get("MODEL_PATH", "checkpoints/deblurnet.pt")
DEVICE_PREF = _get("DEVICE", "auto")
MODEL_CHANNELS = _get("MODEL_CHANNELS", 48, int)
MODEL_BLOCKS = _get("MODEL_BLOCKS", 6, int)
TILE_SIZE = _get("TILE_SIZE", 384, int)
TILE_OVERLAP = _get("TILE_OVERLAP", 32, int)
if TILE_SIZE <= 2 * TILE_OVERLAP:
    TILE_OVERLAP = max(TILE_SIZE // 4, 0)

TRAIN_DATA_DIR = _get("TRAIN_DATA_DIR", "")
DATA_CACHE_DIR = _get("DATA_CACHE_DIR", "data")
TRAIN_SUBSET = _get("TRAIN_SUBSET", 20000, int)
PATCH_SIZE = _get("PATCH_SIZE", 64, int)
EPOCHS = _get("EPOCHS", 15, int)
BATCH_SIZE = _get("BATCH_SIZE", 64, int)
LEARNING_RATE = _get("LEARNING_RATE", 2e-3, float)
VAL_SPLIT = _get("VAL_SPLIT", 0.05, float)
NUM_WORKERS = _get("NUM_WORKERS", 0, int)
SEED = _get("SEED", 42, int)

BLUR_MAX_SIGMA = max(_get("BLUR_MAX_SIGMA", 2.0, float), 0.6)
BLUR_MAX_MOTION = max(_get("BLUR_MAX_MOTION", 11, int), 4)
NOISE_MAX = _get("NOISE_MAX", 3.0, float)


def resolve_device(pref: str) -> torch.device:
    pref = pref.lower()
    if pref in ("auto", "cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if pref in ("auto", "mps") and mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = resolve_device(DEVICE_PREF)


class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1),
        )

    def forward(self, x):
        return x + self.body(x)


class DeblurNet(nn.Module):
    def __init__(self, channels: int = 48, blocks: int = 6):
        super().__init__()
        self.head = nn.Sequential(nn.Conv2d(3, channels, 3, padding=1), nn.ReLU(inplace=True))
        self.down = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 3, stride=2, padding=1), nn.ReLU(inplace=True)
        )
        self.body = nn.Sequential(*[ResBlock(channels * 2) for _ in range(blocks)])
        self.up = nn.Sequential(
            nn.Conv2d(channels * 2, channels * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
        )
        self.tail = nn.Conv2d(channels, 3, 3, padding=1)

    def forward(self, x):
        h, w = x.shape[-2:]
        ph, pw = h % 2, w % 2
        x_in = F.pad(x, (0, pw, 0, ph), mode="reflect") if (ph or pw) else x
        f0 = self.head(x_in)
        f = self.down(f0)
        f = f + self.body(f)
        f = self.up(f) + f0
        out = x_in + self.tail(f)
        return out[..., :h, :w]


def motion_kernel(length: int, angle: float) -> np.ndarray:
    length = max(int(length), 3)
    k = np.zeros((length, length), np.float32)
    k[length // 2, :] = 1.0
    c = (length - 1) / 2
    k = cv2.warpAffine(k, cv2.getRotationMatrix2D((c, c), float(angle), 1.0), (length, length))
    s = float(k.sum())
    if s < 1e-6:
        k[:] = 0
        k[length // 2, length // 2] = 1.0
        s = 1.0
    return k / s


def apply_blur(img: np.ndarray, kind: str, strength: float, angle: float = 0.0) -> np.ndarray:
    if kind == "gaussian":
        sigma = max(float(strength), 0.1)
        k = int(2 * math.ceil(3 * sigma) + 1)
        return cv2.GaussianBlur(img, (k, k), sigma, borderType=cv2.BORDER_REFLECT)
    return cv2.filter2D(img, -1, motion_kernel(int(strength), angle), borderType=cv2.BORDER_REFLECT)


def random_degrade(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if rng.random() < 0.5:
        out = apply_blur(img, "gaussian", rng.uniform(0.5, BLUR_MAX_SIGMA))
    else:
        out = apply_blur(img, "motion", int(rng.integers(3, BLUR_MAX_MOTION + 1)), rng.uniform(0, 180))
    noise = rng.uniform(0, NOISE_MAX)
    if noise > 0:
        out = np.clip(out.astype(np.float32) + rng.normal(0, noise, out.shape), 0, 255).astype(np.uint8)
    return out


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class CifarSource:
    patch = 32

    def __init__(self, root: str):
        self.data = CIFAR10(root=root, train=True, download=True).data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


class FolderSource:
    def __init__(self, root: Path, patch: int):
        self.patch = patch
        self.paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        n = len(self.paths)
        for off in range(n):
            try:
                return np.asarray(Image.open(self.paths[(i + off) % n]).convert("RGB"))
            except Exception:
                continue
        raise RuntimeError("No readable images in TRAIN_DATA_DIR")


class BlurDataset(Dataset):
    def __init__(self, source, indices, train: bool):
        self.source, self.indices, self.train = source, list(indices), train
        self.patch = source.patch

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        img = self.source[self.indices[i]]
        p = self.patch
        h, w = img.shape[:2]
        if min(h, w) < p:
            s = p / min(h, w)
            img = cv2.resize(img, (max(p, round(w * s)), max(p, round(h * s))), interpolation=cv2.INTER_CUBIC)
            h, w = img.shape[:2]

        rng = np.random.default_rng() if self.train else np.random.default_rng(SEED + i)
        if self.train:
            t, l = int(rng.integers(0, h - p + 1)), int(rng.integers(0, w - p + 1))
        else:
            t, l = (h - p) // 2, (w - p) // 2
        sharp = img[t : t + p, l : l + p]
        if self.train:
            if rng.random() < 0.5:
                sharp = sharp[:, ::-1]
            if rng.random() < 0.5:
                sharp = sharp[::-1, :]
        sharp = np.ascontiguousarray(sharp)
        blurry = random_degrade(sharp, rng)

        def to_t(a):
            return torch.from_numpy(a).permute(2, 0, 1).float() / 255.0

        return to_t(blurry), to_t(sharp)


def build_datasets():
    if TRAIN_DATA_DIR and Path(TRAIN_DATA_DIR).is_dir():
        source = FolderSource(Path(TRAIN_DATA_DIR), PATCH_SIZE)
        log.info("Using %d images from %s", len(source), TRAIN_DATA_DIR)
    else:
        source = CifarSource(DATA_CACHE_DIR)
        log.info("TRAIN_DATA_DIR not set -> using CIFAR-10 (%d images)", len(source))
    n = len(source)
    if n == 0:
        raise RuntimeError("No training images found")
    idx = np.random.default_rng(SEED).permutation(n)
    if TRAIN_SUBSET > 0:
        idx = idx[:TRAIN_SUBSET]
    n_val = max(1, int(len(idx) * VAL_SPLIT))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    if len(train_idx) == 0:
        raise RuntimeError("Not enough images to split into train/val")
    return BlurDataset(source, train_idx, True), BlurDataset(source, val_idx, False)


TRAIN_STATE = {
    "running": False,
    "epoch": 0,
    "epochs": 0,
    "train_loss": None,
    "val_psnr": None,
    "baseline_psnr": None,
    "best_psnr": None,
    "error": None,
    "device": str(DEVICE),
}
_train_lock = threading.Lock()


def psnr(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    mse = ((a - b) ** 2).flatten(1).mean(1).clamp_min(1e-10)
    return 10 * torch.log10(1.0 / mse)


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    p_sum = b_sum = 0.0
    n = 0
    for blurry, sharp in loader:
        blurry, sharp = blurry.to(device), sharp.to(device)
        pred = model(blurry).clamp(0, 1)
        p_sum += psnr(pred, sharp).sum().item()
        b_sum += psnr(blurry, sharp).sum().item()
        n += blurry.size(0)
    return p_sum / n, b_sum / n


def train_model():
    if not _train_lock.acquire(blocking=False):
        raise RuntimeError("Training already running")
    try:
        TRAIN_STATE.update(running=True, error=None, epoch=0, epochs=EPOCHS, best_psnr=None)
        torch.manual_seed(SEED)

        train_ds, val_ds = build_datasets()
        pin = DEVICE.type == "cuda"
        train_dl = DataLoader(
            train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
            pin_memory=pin, drop_last=len(train_ds) > BATCH_SIZE,
            persistent_workers=NUM_WORKERS > 0,
        )
        val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

        model = DeblurNet(MODEL_CHANNELS, MODEL_BLOCKS).to(DEVICE)
        log.info("Params: %.2fM | device: %s", sum(p.numel() for p in model.parameters()) / 1e6, DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(EPOCHS, 1))
        use_amp = DEVICE.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        ac_dev = "cuda" if use_amp else "cpu"
        criterion = nn.L1Loss()
        best = -1.0

        for epoch in range(1, EPOCHS + 1):
            model.train()
            total, seen = 0.0, 0
            for blurry, sharp in train_dl:
                blurry = blurry.to(DEVICE, non_blocking=True)
                sharp = sharp.to(DEVICE, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type=ac_dev, dtype=torch.float16, enabled=use_amp):
                    loss = criterion(model(blurry), sharp)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                total += loss.item() * blurry.size(0)
                seen += blurry.size(0)
            sched.step()

            val_psnr, base_psnr = evaluate(model, val_dl, DEVICE)
            TRAIN_STATE.update(
                epoch=epoch, train_loss=round(total / seen, 5),
                val_psnr=round(val_psnr, 3), baseline_psnr=round(base_psnr, 3),
            )
            log.info(
                "epoch %d/%d | loss %.4f | val PSNR %.2f dB (blurry input: %.2f dB)",
                epoch, EPOCHS, total / seen, val_psnr, base_psnr,
            )

            if val_psnr > best:
                best = val_psnr
                TRAIN_STATE["best_psnr"] = round(best, 3)
                Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {"state_dict": model.state_dict(), "channels": MODEL_CHANNELS,
                     "blocks": MODEL_BLOCKS, "val_psnr": best},
                    MODEL_PATH,
                )
                log.info("saved checkpoint -> %s", MODEL_PATH)
    except Exception as e:
        TRAIN_STATE["error"] = str(e)
        log.exception("Training failed")
        raise
    finally:
        TRAIN_STATE["running"] = False
        _train_lock.release()


class ModelStore:
    def __init__(self):
        self.model: Optional[DeblurNet] = None
        self.meta: dict = {}
        self._lock = threading.Lock()

    def load(self) -> bool:
        path = Path(MODEL_PATH)
        if not path.is_file():
            return False
        ckpt = torch.load(path, map_location=DEVICE, weights_only=True)
        model = DeblurNet(ckpt.get("channels", MODEL_CHANNELS), ckpt.get("blocks", MODEL_BLOCKS)).to(DEVICE)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        with self._lock:
            self.model = model
            self.meta = {"val_psnr": ckpt.get("val_psnr"), "channels": ckpt.get("channels"), "blocks": ckpt.get("blocks")}
        log.info("Model loaded from %s", path)
        return True


store = ModelStore()


def tiled_forward(model, x: torch.Tensor, tile: int, overlap: int) -> torch.Tensor:
    _, _, H, W = x.shape
    if max(H, W) <= tile:
        return model(x)
    out = torch.zeros_like(x)
    stride = tile - 2 * overlap
    for top in range(0, H, stride):
        for left in range(0, W, stride):
            t0, l0 = max(top - overlap, 0), max(left - overlap, 0)
            t1, l1 = min(top + stride + overlap, H), min(left + stride + overlap, W)
            y = model(x[:, :, t0:t1, l0:l1])
            h, w = min(stride, H - top), min(stride, W - left)
            ct, cl = top - t0, left - l0
            out[:, :, top : top + h, left : left + w] = y[:, :, ct : ct + h, cl : cl + w]
    return out


@torch.inference_mode()
def deblur_image(img: Image.Image) -> Image.Image:
    arr = np.array(img, dtype=np.float32) / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    y = tiled_forward(store.model, x, TILE_SIZE, TILE_OVERLAP).clamp_(0, 1)
    out = (y.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(out)


def read_upload(data: bytes) -> Image.Image:
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File too large (max {MAX_UPLOAD_MB} MB)")
    try:
        img = Image.open(io.BytesIO(data))
        size = img.size
    except Exception:
        raise HTTPException(400, "Invalid or unsupported image")
    if size[0] * size[1] > MAX_IMAGE_PIXELS:
        raise HTTPException(413, f"Image too large (max {MAX_IMAGE_PIXELS:,} pixels)")
    try:
        img.load()
        return ImageOps.exif_transpose(img).convert("RGB")
    except Exception:
        raise HTTPException(400, "Could not decode image")


def encode_image(img: Image.Image, fmt: str):
    fmt = fmt.lower()
    if fmt not in {"png", "jpeg", "jpg", "webp"}:
        raise HTTPException(400, "output_format must be png, jpeg or webp")
    pil_fmt = "JPEG" if fmt == "jpg" else fmt.upper()
    buf = io.BytesIO()
    img.save(buf, format=pil_fmt, **({"quality": 95} if pil_fmt in ("JPEG", "WEBP") else {}))
    return buf.getvalue(), f"image/{'jpeg' if pil_fmt == 'JPEG' else fmt}"


@asynccontextmanager
async def lifespan(_: FastAPI):
    log.info("Device: %s", DEVICE)
    try:
        if not store.load():
            log.warning("No checkpoint at %s. Run `python backend.py train` or POST /train", MODEL_PATH)
    except Exception:
        log.exception("Failed to load checkpoint")
    yield


app = FastAPI(title="Image Deblur API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Process-Time-Ms", "X-Image-Size"],
)


def require_admin(x_admin_token: Optional[str] = Header(default=None)):
    if ADMIN_TOKEN and not hmac.compare_digest(x_admin_token or "", ADMIN_TOKEN):
        raise HTTPException(401, "Invalid admin token")


@app.get("/")
def root():
    index_path = Path(__file__).resolve().parent / "templates" / "index.html"
    if index_path.is_file():
        return FileResponse(index_path)
    return {"name": "Image Deblur API", "docs": "/docs", "model_loaded": store.model is not None}


@app.get("/api")
def api_info():
    return {"name": "Image Deblur API", "docs": "/docs", "model_loaded": store.model is not None}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": store.model is not None, "device": str(DEVICE), "model": store.meta}


@app.post("/deblur")
async def deblur(file: UploadFile = File(...), output_format: str = Query("png")):
    if store.model is None:
        raise HTTPException(503, "Model not trained yet. Run `python backend.py train` or POST /train")
    data = await file.read(MAX_UPLOAD_MB * 1024 * 1024 + 1)
    img = read_upload(data)
    t0 = time.perf_counter()
    out = await run_in_threadpool(deblur_image, img)
    body, mime = encode_image(out, output_format)
    return Response(
        body, media_type=mime,
        headers={"X-Process-Time-Ms": f"{(time.perf_counter() - t0) * 1000:.0f}",
                 "X-Image-Size": f"{img.width}x{img.height}"},
    )


@app.post("/blur-demo")
async def blur_demo(
    file: UploadFile = File(...),
    kind: str = Query("gaussian", pattern="^(gaussian|motion)$"),
    strength: float = Query(2.0, ge=0.5, le=25),
    angle: float = Query(0.0, ge=0, le=180),
    output_format: str = Query("png"),
):
    data = await file.read(MAX_UPLOAD_MB * 1024 * 1024 + 1)
    img = read_upload(data)
    blurred = await run_in_threadpool(apply_blur, np.array(img), kind, strength, angle)
    body, mime = encode_image(Image.fromarray(blurred), output_format)
    return Response(body, media_type=mime)


def _train_background():
    try:
        train_model()
        store.load()
    except Exception:
        pass


@app.post("/train", status_code=202, dependencies=[Depends(require_admin)])
def start_training():
    if _train_lock.locked():
        raise HTTPException(409, "Training already running")
    threading.Thread(target=_train_background, daemon=True).start()
    return {"message": "Training started", "status_url": "/train/status"}


@app.get("/train/status")
def train_status():
    return TRAIN_STATE


def main():
    parser = argparse.ArgumentParser(description="Image deblur CNN + FastAPI")
    parser.add_argument("command", nargs="?", default="serve", choices=["serve", "train"])
    args = parser.parse_args()

    if args.command == "train":
        train_model()
    else:
        import uvicorn

        uvicorn.run(app, host=HOST, port=PORT)


if __name__ == "__main__":
    main()
