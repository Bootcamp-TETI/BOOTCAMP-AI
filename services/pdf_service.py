"""
PDF Service — menangani DUA arah proses PDF dalam sistem ini:

1. INGESTION (PDF masuk)  : bnpb_sop.pdf -> extract text -> clean -> chunk -> embedding -> FAISS index
   Dipakai sekali di awal (atau setiap kali SOP diperbarui) lewat build_sop_index(),
   biasanya dijalankan sebagai script terpisah (mis. scripts/build_index.py), bukan setiap request.

2. GENERATION (PDF keluar): stats + ai_report -> PDF laporan yang bisa diunduh user
   Dipakai oleh main.py (endpoint /report/pdf) atau langsung oleh streamlit_app.py.
"""
import io
import json
import os
import re
from typing import List, Tuple

import faiss
import numpy as np
from pypdf import PdfReader
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas as rl_canvas

from services.gemini_service import GeminiService


# ======================================================================
# 1) INGESTION: PDF SOP -> teks bersih -> chunk -> FAISS index
# ======================================================================

def extract_raw_text(pdf_path: str) -> str:
    """Ekstrak teks mentah dari semua halaman PDF."""
    reader = PdfReader(pdf_path)
    raw_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            raw_text += text + "\n"
    return raw_text


def clean_pdf_text(text: str) -> str:
    """
    Pembersih teks hasil ekstraksi PDF untuk dokumen berbahasa Indonesia.
    Menangani masalah umum: hyphenation, newline acak di tengah kalimat, spasi
    berlebih, dan beberapa singkatan birokrasi umum di dokumen SOP pemerintah.
    """
    # 1. Hapus nomor halaman yang berdiri sendiri di satu baris
    text = re.sub(r'(?m)^\s*\d+\s*$', '', text)
    # 2. Perbaiki kata terpenggal garis hubung di akhir baris
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    # 3. Hapus newline yang memutus kalimat di tengah jalan
    text = re.sub(r'(?<![.:;!?])\n(?!\s*[-•A-Z0-9])', ' ', text)
    # 4. Hapus spasi sebelum tanda baca
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)
    # 5. Normalisasi singkatan birokrasi umum
    abbreviations = {
        r'\bGub/Bup/Wako\b': 'Gubernur / Bupati / Walikota',
        r'\bPusdalops PB\b': 'Pusat Pengendalian Operasi Penanggulangan Bencana',
        r'\bBPBD\b': 'Badan Penanggulangan Bencana Daerah',
        r'\bBNPB\b': 'Badan Nasional Penanggulangan Bencana',
        r'\bSOP\b': 'Standar Operasional Prosedur',
        r'\btgl\.?\b': 'tanggal',
        r'\bkpd\.?\b': 'kepada',
    }
    for abbr, full in abbreviations.items():
        text = re.sub(abbr, full, text)
    # 6. Normalisasi spasi/newline berlebih
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def split_text(text: str, chunk_size: int = 350, overlap: int = 50) -> List[str]:
    """Text splitter sederhana: gabungkan paragraf sampai mendekati chunk_size,
    lalu bawa sedikit teks (overlap) ke chunk berikutnya agar konteks tidak putus."""
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks: List[str] = []
    buffer = ""
    for para in paragraphs:
        if len(buffer) + len(para) + 1 <= chunk_size:
            buffer = (buffer + " " + para).strip()
        else:
            if buffer:
                chunks.append(buffer)
            carry = buffer[-overlap:] if buffer else ""
            buffer = (carry + " " + para).strip()
            while len(buffer) > chunk_size:
                chunks.append(buffer[:chunk_size])
                buffer = buffer[chunk_size - overlap:]
    if buffer:
        chunks.append(buffer)
    return chunks


def build_sop_index(pdf_path: str, output_dir: str, gemini_service: GeminiService,
                     chunk_size: int = 350, overlap: int = 50) -> Tuple[str, str]:
    """
    Pipeline lengkap ingestion: PDF -> extract -> clean -> chunk -> embed -> FAISS.
    Menyimpan sop_faiss.index dan sop_chunks.json ke output_dir, lalu mengembalikan
    path keduanya. Ini dipanggil SEKALI (offline), bukan per-request di FastAPI.
    """
    os.makedirs(output_dir, exist_ok=True)

    raw_text = extract_raw_text(pdf_path)
    clean_text = clean_pdf_text(raw_text)
    chunks = split_text(clean_text, chunk_size=chunk_size, overlap=overlap)
    if len(chunks) <= 1:
        raise ValueError("Chunking gagal menghasilkan lebih dari 1 chunk — cek isi PDF.")

    embeddings = gemini_service.embed(chunks, task_type="RETRIEVAL_DOCUMENT")

    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    index_path = os.path.join(output_dir, "sop_faiss.index")
    chunks_path = os.path.join(output_dir, "sop_chunks.json")
    faiss.write_index(index, index_path)
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    return index_path, chunks_path


# ======================================================================
# 2) GENERATION: stats + laporan Gemini -> PDF yang bisa diunduh
# ======================================================================

def generate_pdf_report(stats: dict, title: str = "Laporan Penilaian Kerusakan Pasca-Bencana") -> io.BytesIO:
    """
    Susun PDF laporan dari `stats` (hasil /analyze di main.py) — dipakai baik oleh
    endpoint FastAPI (mis. GET /report/pdf) maupun langsung oleh streamlit_app.py
    lewat st.download_button, supaya logikanya tidak diduplikasi di dua tempat.
    """
    buf = io.BytesIO()
    c = rl_canvas.Canvas(buf, pagesize=A4)
    width, height = A4
    y = height - 60

    def new_page_if_needed(current_y, margin=60):
        if current_y < margin:
            c.showPage()
            return height - 60
        return current_y

    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, title)
    y -= 30

    c.setFont("Helvetica-Bold", 11)
    c.drawString(50, y, f"Priority: {stats.get('priority', 'N/A')}")
    y -= 16
    c.setFont("Helvetica", 10)

    summary_lines = [
        f"Damage percentage : {stats.get('damage_percentage', 'N/A')}%",
        f"Damaged pixels     : {stats.get('damaged_pixels', 'N/A'):,}"
        if isinstance(stats.get("damaged_pixels"), int) else f"Damaged pixels     : {stats.get('damaged_pixels', 'N/A')}",
        f"Total building px  : {stats.get('total_building_pixels', 'N/A')}",
        f"Confidence         : {stats.get('confidence', 'N/A')}",
        f"Recommended action : {stats.get('recommended_action', 'N/A')}",
        f"Evacuation radius  : {stats.get('evacuation_radius_km', 'N/A')} km",
        f"Required logistics : {', '.join(stats.get('required_logistics', []))}",
    ]
    for line in summary_lines:
        c.drawString(50, y, line)
        y -= 14
        y = new_page_if_needed(y)

    y -= 10
    c.setFont("Helvetica-Bold", 11)
    c.drawString(50, y, "AI Report (RAG-Grounded):")
    y -= 18
    c.setFont("Helvetica", 9)

    report_text = stats.get("ai_report", "Report not available.")
    for raw_line in report_text.split("\n"):
        wrapped_lines = [raw_line[i:i + 100] for i in range(0, max(len(raw_line), 1), 100)] or [""]
        for wrapped in wrapped_lines:
            c.drawString(50, y, wrapped)
            y -= 12
            y = new_page_if_needed(y)

    c.save()
    buf.seek(0)
    return buf
