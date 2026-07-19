"""
Backend orchestrator (FastAPI) — thin layer that wires Frontend <-> Model Service <-> RAG Service.
Run with: uvicorn main:app --host 0.0.0.0 --port 8000

Path model & RAG disesuaikan dengan struktur folder:
    models/segformer_best.pt
    rag/sop_faiss.index
    rag/sop_chunks.json
    services/model_service.py, rag_service.py, gemini_service.py, pdf_service.py
"""
import base64
import logging
import os
import time

import cv2
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse

from services.model_service import ModelService
from services.rag_service import RagService

# Baca file .env di root project (kalau ada) supaya GOOGLE_API_KEY, dsb.
# tidak perlu di-set manual lewat $env: di setiap sesi terminal baru.
load_dotenv()

# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("main")

# ----------------------------------------------------------------------
# Konfigurasi (semua lewat environment variable, tidak ada yang hardcode)
# ----------------------------------------------------------------------
MODEL_DIR = os.environ.get("MODEL_DIR", "models")
RAG_DIR = os.environ.get("RAG_DIR", "rag")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
CHECKPOINT_NAME = os.environ.get("CHECKPOINT_NAME", "segformer_best.pt")
# Asumsi ground sample distance (meter/piksel) untuk estimasi luas area — sesuaikan dengan
# resolusi citra satelit/drone yang sebenarnya dipakai.
GSD_METERS_PER_PIXEL = float(os.environ.get("GSD_METERS_PER_PIXEL", "0.3"))

app = FastAPI(title="Post-Disaster Damage Assessment API", version="1.1.0")

# ----------------------------------------------------------------------
# Load services sekali saat startup (bukan per-request)
# ----------------------------------------------------------------------
logger.info("Memuat ModelService dari %s/%s ...", MODEL_DIR, CHECKPOINT_NAME)
try:
    model_service = ModelService(checkpoint_path=f"{MODEL_DIR}/{CHECKPOINT_NAME}")
    logger.info("ModelService siap.")
except Exception:
    logger.exception("Gagal memuat ModelService — API tidak akan bisa melayani /analyze.")
    raise

rag_service = None
if GOOGLE_API_KEY:
    try:
        rag_service = RagService(
            faiss_index_path=f"{RAG_DIR}/sop_faiss.index",
            chunks_path=f"{RAG_DIR}/sop_chunks.json",
            google_api_key=GOOGLE_API_KEY,
        )
        logger.info("RagService siap.")
    except Exception:
        logger.exception("Gagal memuat RagService — RAG akan dinonaktifkan, /analyze tetap jalan tanpa laporan AI.")
        rag_service = None
else:
    logger.warning("GOOGLE_API_KEY tidak diset — RAG service dinonaktifkan.")


# ----------------------------------------------------------------------
# Helper: encoding gambar hasil jadi base64 PNG untuk dikirim ke frontend
# ----------------------------------------------------------------------
def _encode_png_base64(arr: np.ndarray) -> str:
    """Encode numpy image array (grayscale atau BGR) menjadi base64 PNG string."""
    ok, buf = cv2.imencode(".png", arr)
    if not ok:
        raise ValueError("Gagal encode array gambar ke format PNG.")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _mask_overlay(img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Overlay mask bangunan (kuning) di atas citra aslinya — jauh lebih terbaca
    daripada mask mentah hitam-putih."""
    vis = img_bgr.copy()
    m = mask.astype(bool)
    vis[m] = (0.4 * vis[m] + 0.6 * np.array((0, 220, 255))).astype(np.uint8)  # BGR kuning
    return vis


def _diff_overlay(post_bgr: np.ndarray, diff: np.ndarray) -> np.ndarray:
    """Overlay difference map di atas citra post: hijau = utuh, merah = rusak/hilang."""
    vis = post_bgr.copy()
    utuh, rusak = diff == 1, diff == 2
    vis[utuh] = (0.45 * vis[utuh] + 0.55 * np.array((80, 220, 80))).astype(np.uint8)
    vis[rusak] = (0.35 * vis[rusak] + 0.65 * np.array((60, 60, 255))).astype(np.uint8)
    return vis


def _building_counts(mask_pre: np.ndarray, diff: np.ndarray,
                     min_area_px: int = 20, min_damage_px: int = 10):
    """Hitung (total, rusak) bangunan dari komponen terhubung mask pre.
    Bangunan = blob mask_pre >= min_area_px (filter noise). Bangunan dihitung RUSAK
    kalau blob itu kehilangan >= min_damage_px piksel di difference map — dengan cara
    ini 'rusak' selalu subset dari 'total' (satu bangunan yang pecah jadi beberapa
    blob kerusakan tidak dihitung berkali-kali)."""
    n_labels, labels_img, comp_stats, _ = cv2.connectedComponentsWithStats(mask_pre.astype(np.uint8))
    valid = comp_stats[:, cv2.CC_STAT_AREA] >= min_area_px
    valid[0] = False  # index 0 = background
    damaged_ids, damage_px = np.unique(labels_img[diff == 2], return_counts=True)
    n_damaged = sum(1 for i, c in zip(damaged_ids, damage_px)
                    if i != 0 and valid[i] and c >= min_damage_px)
    return int(valid.sum()), int(n_damaged)


# ----------------------------------------------------------------------
# Endpoint utama
# ----------------------------------------------------------------------
@app.post("/analyze")
async def analyze(
    pre_image: UploadFile = File(...),
    post_image: UploadFile = File(...),
    location: str = Form(None),
    disaster_type: str = Form(None),
    center_lat: float = Form(None),
    center_lon: float = Form(None),
    gsd: float = Form(None),
):
    request_start = time.time()
    logger.info("Request diterima: pre=%s, post=%s", pre_image.filename, post_image.filename)

    # --- 1) Baca & decode gambar, dengan validasi ---
    try:
        pre_np = np.frombuffer(await pre_image.read(), np.uint8)
        post_np = np.frombuffer(await post_image.read(), np.uint8)
        pre_bgr = cv2.imdecode(pre_np, cv2.IMREAD_COLOR)
        post_bgr = cv2.imdecode(post_np, cv2.IMREAD_COLOR)
    except Exception as e:
        logger.error("Gagal membaca file upload: %s", e)
        return JSONResponse(status_code=400, content={"error": f"Gagal membaca file upload: {e}"})

    if pre_bgr is None or post_bgr is None:
        logger.error("Salah satu gambar tidak valid / corrupt / bukan format gambar yang didukung.")
        return JSONResponse(
            status_code=400,
            content={"error": "Salah satu gambar tidak valid atau gagal dibaca. "
                              "Pastikan format PNG/JPG dan file tidak corrupt."},
        )

    MAX_SIDE = 8192  # batas dimensi supaya gambar raksasa tidak bikin backend kehabisan memori
    if max(pre_bgr.shape[:2]) > MAX_SIDE or max(post_bgr.shape[:2]) > MAX_SIDE:
        return JSONResponse(
            status_code=400,
            content={"error": f"Dimensi gambar melebihi batas {MAX_SIDE}px. Perkecil gambar terlebih dahulu."},
        )

    if pre_bgr.shape[:2] != post_bgr.shape[:2]:
        logger.warning("Ukuran pre (%s) dan post (%s) berbeda.", pre_bgr.shape[:2], post_bgr.shape[:2])
        return JSONResponse(
            status_code=400,
            content={"error": f"Ukuran gambar pre {pre_bgr.shape[:2]} dan post {post_bgr.shape[:2]} berbeda. "
                              "Gunakan pasangan citra pre/post dengan dimensi yang sama."},
        )

    # --- 2) Model Service: segmentasi + difference map + damage stats ---
    try:
        t0 = time.time()
        mask_pre, mask_post, diff, stats = model_service.analyze(pre_bgr, post_bgr)
        model_time = round(time.time() - t0, 2)
        logger.info("Model inference selesai dalam %.2fs", model_time)
    except Exception as e:
        logger.exception("Model inference gagal")
        return JSONResponse(status_code=500, content={"error": f"Model inference gagal: {e}"})

    # --- 3) Enrich stats: estimasi jumlah bangunan & luas area ---
    # GSD per-request (mis. dari metadata label xBD) menang atas default env.
    gsd_eff = gsd if gsd and gsd > 0 else GSD_METERS_PER_PIXEL
    try:
        n_total_buildings, n_damaged_buildings = _building_counts(mask_pre, diff)
        n_safe_buildings = n_total_buildings - n_damaged_buildings
        area_m2 = round(stats.get("total_building_pixels", 0) * (gsd_eff ** 2), 1)
        stats.update({
            "buildings_total": n_total_buildings,
            "buildings_damaged": n_damaged_buildings,
            "buildings_safe": n_safe_buildings,
            "area_m2": area_m2,
            "gsd_meters_per_pixel": gsd_eff,
        })
    except Exception as e:
        logger.warning("Gagal menghitung estimasi jumlah bangunan/area (non-fatal): %s", e)

    # --- 3b) Konteks kejadian + koordinat bangunan rusak (untuk tim lapangan/SAR) ---
    if location:
        stats["location"] = location
    if disaster_type:
        stats["disaster_type"] = disaster_type
    try:
        h, w = diff.shape[:2]
        n_blobs, _, blob_stats, blob_cents = cv2.connectedComponentsWithStats((diff == 2).astype(np.uint8))
        blobs = [(blob_cents[i][0], blob_cents[i][1], int(blob_stats[i, cv2.CC_STAT_AREA]))
                 for i in range(1, n_blobs) if blob_stats[i, cv2.CC_STAT_AREA] >= 20]
        blobs.sort(key=lambda b: -b[2])  # terbesar dulu — prioritas tim lapangan
        locations = []
        for cx, cy, area_px in blobs[:100]:
            item = {"pixel_x": int(cx), "pixel_y": int(cy),
                    "area_m2": round(area_px * gsd_eff ** 2, 1)}
            if center_lat is not None and center_lon is not None:
                # Konversi piksel -> lat/lon: aproksimasi equirectangular di sekitar pusat citra.
                # Cukup akurat untuk area kecil (beberapa km); bukan pengganti georeference GeoTIFF asli.
                dy_m = (cy - h / 2) * gsd_eff
                dx_m = (cx - w / 2) * gsd_eff
                item["lat"] = round(center_lat - dy_m / 111320.0, 6)
                item["lon"] = round(center_lon + dx_m / (111320.0 * max(float(np.cos(np.radians(center_lat))), 1e-6)), 6)
            locations.append(item)
        stats["damaged_building_locations"] = locations
    except Exception as e:
        logger.warning("Gagal menghitung koordinat bangunan rusak (non-fatal): %s", e)

    # --- 4) RAG Service: retrieval + Gemini report generation ---
    rag_time = None
    if rag_service:
        try:
            t0 = time.time()
            report_text, retrieved = rag_service.generate_report(stats)
            rag_time = round(time.time() - t0, 2)
            stats["ai_report"] = report_text
            def _tidy_chunk(chunk: str, max_len: int = 220) -> str:
                """Rapikan chunk untuk ditampilkan: buang pecahan kata di awal, potong di batas kata."""
                text = " ".join(chunk.split())
                if text and text[0].islower() and " " in text:
                    text = "..." + text[text.find(" "):]
                if len(text) > max_len:
                    text = text[:max_len].rsplit(" ", 1)[0] + "..."
                return text

            stats["rag_sources_used"] = [
                {"text": _tidy_chunk(chunk), "score": round(score, 3)} for chunk, score in retrieved
            ]
            logger.info("RAG + Gemini report selesai dalam %.2fs", rag_time)
        except Exception as e:
            logger.exception("RAG/Gemini report gagal (non-fatal, analisis CV tetap dikembalikan)")
            stats["ai_report"] = f"Gagal membuat laporan AI: {e}"
            stats["rag_sources_used"] = []
    else:
        stats["ai_report"] = "GOOGLE_API_KEY not set — RAG service disabled."
        stats["rag_sources_used"] = []

    total_time = round(time.time() - request_start, 2)
    stats["inference_time"] = {
        "model_seconds": model_time,
        "rag_seconds": rag_time,
        "total_seconds": total_time,
    }
    stats["confidence_note"] = (
        "confidence = rata-rata probabilitas softmax (TTA 3-arah: asli + flip H/V) pada piksel "
        "interior bangunan citra post-disaster — piksel tepi dikecualikan karena ketidakpastian "
        "batas adalah sifat bawaan segmentasi. Bukan angka akurasi model (lihat mIoU/Dice validasi)."
    )
    logger.info("Request selesai dalam %.2fs (model=%.2fs, rag=%s)", total_time, model_time, rag_time)

    # --- 5) Encode visual outputs sebagai base64 PNG agar bisa dirender langsung di frontend ---
    try:
        payload = {
            "stats": stats,
            "mask_pre": _encode_png_base64(_mask_overlay(pre_bgr, mask_pre)),
            "mask_post": _encode_png_base64(_mask_overlay(post_bgr, mask_post)),
            "difference_map": _encode_png_base64(_diff_overlay(post_bgr, diff)),
        }
    except Exception as e:
        logger.exception("Gagal encode gambar hasil ke base64")
        # stats tetap dikembalikan walau gambar gagal di-encode, supaya frontend tidak kehilangan semua info
        return JSONResponse(
            status_code=500,
            content={"error": f"Gagal encode gambar hasil: {e}", "stats": stats},
        )

    return JSONResponse(payload)


# ----------------------------------------------------------------------
# Endpoint pendukung: health, version, model-info
# ----------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok", "rag_enabled": rag_service is not None}


@app.get("/version")
async def version():
    return {"version": app.version, "service": "post-disaster-damage-assessment-api"}


@app.get("/model-info")
async def model_info():
    return {
        "model": "SegFormer-B0 (fine-tuned, binary building segmentation)",
        "checkpoint": f"{MODEL_DIR}/{CHECKPOINT_NAME}",
        "rag_enabled": rag_service is not None,
        "gsd_meters_per_pixel": GSD_METERS_PER_PIXEL,
    }
