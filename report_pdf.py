"""Generator PDF laporan penilaian kerusakan — gaya situation report (sitrep) OCHA:
header band navy + eyebrow, key figures besar, strip severity berwarna prioritas,
tabel koordinat, grid visual, dan render markdown-lite untuk laporan AI.

Murni reportlab, tanpa streamlit — bisa dipakai frontend maupun diuji standalone.
"""
import io
import re
import textwrap
from datetime import datetime

from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas as rl_canvas

NAVY = HexColor("#16355A")
INK = HexColor("#1F2937")
MUT = HexColor("#6B7280")
FAINT = HexColor("#9CA3AF")
LINE = HexColor("#E5E7EB")
BG = HexColor("#F8FAFC")
WHITE = HexColor("#FFFFFF")
HDR_SUB = HexColor("#9DB2CC")

PRIO = {
    "GREEN": (HexColor("#16A34A"), HexColor("#DCFCE7")),
    "YELLOW": (HexColor("#CA8A04"), HexColor("#FEF9C3")),
    "ORANGE": (HexColor("#EA580C"), HexColor("#FFEDD5")),
    "RED": (HexColor("#DC2626"), HexColor("#FEE2E2")),
}

W, H = A4
M = 42
BOTTOM = 52


def build_pdf(stats: dict, images, confidence_pct: str) -> io.BytesIO:
    """stats: dict hasil /analyze. images: list[(label, bytes)]. Return BytesIO PDF."""
    buf = io.BytesIO()
    c = rl_canvas.Canvas(buf, pagesize=A4)
    prio = stats.get("priority", "N/A")
    p_col, p_tint = PRIO.get(prio, (MUT, BG))
    state = {"page": 0}

    # ------------------------------------------------------------------
    def new_page(cont=False):
        if state["page"] > 0:
            c.showPage()
        state["page"] += 1
        # Header band
        c.setFillColor(NAVY)
        c.rect(0, H - 84, W, 84, fill=1, stroke=0)
        c.setFillColor(HDR_SUB)
        c.setFont("Helvetica-Bold", 7)
        c.drawString(M, H - 26, "L A P O R A N   S I T U A S I   ·   P E N I L A I A N   "
                                "K E R U S A K A N   P A S C A - B E N C A N A")
        title_bits = [stats.get("disaster_type"), stats.get("location")]
        title = " — ".join(str(b) for b in title_bits if b) or "Penilaian Kerusakan Pasca-Bencana"
        if cont:
            title += "  (lanjutan)"
        c.setFillColor(WHITE)
        c.setFont("Helvetica-Bold", 16)
        c.drawString(M, H - 47, title[:64])
        c.setFillColor(HDR_SUB)
        c.setFont("Helvetica", 8)
        bulan = ["Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli",
                 "Agustus", "September", "Oktober", "November", "Desember"]
        now = datetime.now()
        c.drawString(M, H - 63, f"Dibuat {now.day} {bulan[now.month - 1]} {now.year}, "
                                f"{now:%H:%M} WIB  ·  dihasilkan otomatis dari citra satelit pre/post-bencana")
        # Badge prioritas kanan
        label = f"{prio}  ·  {stats.get('damage_percentage', '?')}%"
        bw = c.stringWidth(label, "Helvetica-Bold", 10) + 24
        c.setFillColor(p_col)
        c.roundRect(W - M - bw, H - 55, bw, 25, 12.5, fill=1, stroke=0)
        c.setFillColor(WHITE)
        c.setFont("Helvetica-Bold", 10)
        c.drawCentredString(W - M - bw / 2, H - 47, label)
        # Footer
        c.setStrokeColor(LINE)
        c.line(M, 40, W - M, 40)
        c.setFont("Helvetica", 7)
        c.setFillColor(FAINT)
        c.drawString(M, 29, "Estimasi otomatis — WAJIB diverifikasi tim lapangan sebelum menjadi "
                            "dasar keputusan operasional")
        c.drawRightString(W - M, 29, f"Halaman {state['page']}")
        return H - 106

    def ensure(y, need, cont=True):
        if y - need < BOTTOM:
            return new_page(cont=cont)
        return y

    def section(y, num, text):
        y = ensure(y, 46)
        c.setFillColor(p_col)
        c.rect(M, y - 2, 3.5, 11, fill=1, stroke=0)
        c.setFillColor(FAINT)
        c.setFont("Helvetica-Bold", 9)
        c.drawString(M + 10, y, num)
        c.setFillColor(NAVY)
        c.setFont("Helvetica-Bold", 11)
        c.drawString(M + 32, y, text.upper())
        c.setStrokeColor(LINE)
        c.line(M, y - 9, W - M, y - 9)
        return y - 26

    # ------------------------------------------------------------------
    y = new_page()

    # Strip severity: prioritas + aksi
    c.setFillColor(p_tint)
    c.roundRect(M, y - 34, W - 2 * M, 34, 5, fill=1, stroke=0)
    c.setFillColor(p_col)
    c.setFont("Helvetica-Bold", 10.5)
    aksi = stats.get("recommended_action", "-")
    c.drawString(M + 12, y - 15, f"PRIORITAS {prio}")
    c.setFont("Helvetica", 9.5)
    c.setFillColor(INK)
    c.drawString(M + 12, y - 27, f"Tindakan: {aksi}"[:110])
    y -= 52

    # Key figures — gaya OCHA: angka besar + label kecil + pemisah tipis
    figs = [
        (f"{stats.get('damage_percentage', '?')}%", "TINGKAT KERUSAKAN"),
        (str(stats.get("buildings_total", "N/A")), "BANGUNAN TERDETEKSI"),
        (str(stats.get("buildings_damaged", "N/A")), "BANGUNAN RUSAK"),
        (f"{stats.get('area_m2', 'N/A'):,}" if isinstance(stats.get("area_m2"), (int, float))
         else "N/A", "LUAS TERDAMPAK (m2)"),
    ]
    cw = (W - 2 * M) / 4
    for i, (val, lab) in enumerate(figs):
        x = M + i * cw
        c.setFillColor(p_col if i == 0 else NAVY)
        c.setFont("Helvetica-Bold", 21)
        c.drawString(x + 6, y - 22, val)
        c.setFillColor(MUT)
        c.setFont("Helvetica", 6.5)
        c.drawString(x + 6, y - 34, lab)
        if i > 0:
            c.setStrokeColor(LINE)
            c.line(x - 8, y - 36, x - 8, y - 4)
    y -= 50

    # Baris statistik sekunder
    sec_stats = [
        ("Keyakinan segmentasi", f"pre {round(stats.get('confidence_pre', 0) * 100, 1)}% -> "
                                 f"post {confidence_pct}"),
        ("Radius evakuasi", f"{stats.get('evacuation_radius_km', '?')} km"),
        ("Piksel rusak", f"{stats.get('damaged_pixels', 0):,}"),
        ("Resolusi (GSD)", f"{stats.get('gsd_meters_per_pixel', '?')} m/piksel"),
    ]
    c.setFillColor(BG)
    c.roundRect(M, y - 30, W - 2 * M, 30, 5, fill=1, stroke=0)
    for i, (lab, val) in enumerate(sec_stats):
        x = M + 10 + i * cw
        c.setFillColor(MUT)
        c.setFont("Helvetica", 6.5)
        c.drawString(x, y - 12, lab.upper())
        c.setFillColor(INK)
        c.setFont("Helvetica-Bold", 9)
        c.drawString(x, y - 24, val)
    y -= 48

    # 01 — Decision support
    y = section(y, "01", "Decision Support")
    c.setFillColor(INK)
    c.setFont("Helvetica", 9.5)
    c.drawString(M, y, f"Aksi yang direkomendasikan : {aksi}"[:118])
    y -= 14
    c.drawString(M, y, f"Radius evakuasi            : {stats.get('evacuation_radius_km', '?')} km "
                       f"dari pusat area terdampak")
    y -= 14
    logi = stats.get("required_logistics", [])
    c.drawString(M, y, "Kebutuhan logistik         : " + ", ".join(logi)[:100])
    y -= 26

    # 02 — Titik prioritas
    locs = [l for l in stats.get("damaged_building_locations", []) if "lat" in l][:8]
    if locs:
        y = section(y, "02", "Titik Prioritas Tim Lapangan")
        rows = [("#", "LATITUDE", "LONGITUDE", "LUAS (m2)")] + [
            (str(i + 1), f"{l['lat']}", f"{l['lon']}", f"{l['area_m2']:,}")
            for i, l in enumerate(locs)]
        col_x = [M, M + 40, M + 160, M + 280]
        row_h = 15
        c.setFillColor(NAVY)
        c.rect(M - 4, y - 4, W - 2 * M + 8, row_h, fill=1, stroke=0)
        for j, val in enumerate(rows[0]):
            c.setFillColor(WHITE)
            c.setFont("Helvetica-Bold", 7.5)
            c.drawString(col_x[j], y, val)
        y -= row_h
        for i, row in enumerate(rows[1:]):
            if i % 2 == 1:
                c.setFillColor(BG)
                c.rect(M - 4, y - 4, W - 2 * M + 8, row_h, fill=1, stroke=0)
            c.setFillColor(INK)
            c.setFont("Helvetica", 8.5)
            for j, val in enumerate(row):
                c.drawString(col_x[j], y, val)
            y -= row_h
        c.setFillColor(MUT)
        c.setFont("Helvetica-Oblique", 7.5)
        c.drawString(M, y - 2, "Diurutkan dari kerusakan terluas. Koordinat siap dipakai di GPS/Google Maps; "
                               "GeoJSON lengkap tersedia dari dashboard.")
        y -= 22

    # 03 — Visual
    y = section(y, "03", "Visual Analisis")
    img_w = (W - 2 * M - 14) / 2
    img_h = 145
    col = 0
    for label, img_bytes in images:
        if y - img_h - 20 < BOTTOM:
            y = new_page(cont=True)
            col = 0
        x = M + col * (img_w + 14)
        try:
            c.drawImage(ImageReader(io.BytesIO(img_bytes)), x, y - img_h, width=img_w,
                        height=img_h, preserveAspectRatio=True, anchor="c")
            c.setStrokeColor(LINE)
            c.rect(x, y - img_h, img_w, img_h, fill=0, stroke=1)
            c.setFillColor(MUT)
            c.setFont("Helvetica", 7.5)
            c.drawString(x + 1, y - img_h - 11, label)
        except Exception:
            c.setFillColor(MUT)
            c.drawString(x, y - 12, f"[Gagal render gambar: {label}]")
        col += 1
        if col == 2:
            col = 0
            y -= img_h + 26
    if col == 1:
        y -= img_h + 26

    # 04 — Laporan AI (markdown-lite)
    y = new_page(cont=True)
    y = section(y, "04", "Laporan AI (RAG-Grounded, SOP BNPB)")
    report = stats.get("ai_report", "Report not available.")

    def emit(text, font, size, color, indent=0, leading=12):
        nonlocal y
        for wline in textwrap.wrap(text, width=int((110 - indent / 4))) or [""]:
            y = ensure(y, leading + 2)
            c.setFont(font, size)
            c.setFillColor(color)
            c.drawString(M + indent, y, wline)
            y -= leading

    for raw in report.split("\n"):
        line = raw.strip()
        clean = re.sub(r"\*\*?|`", "", line)
        if not line:
            y -= 5
        elif line.startswith("###") or line.startswith("##"):
            y -= 6
            emit(clean.lstrip("# ").upper(), "Helvetica-Bold", 9.5, NAVY, leading=13)
            c.setStrokeColor(LINE)
            c.line(M, y + 8, M + 150, y + 8)
            y -= 3
        elif line.startswith(">"):
            emit(clean.lstrip("> "), "Helvetica-Oblique", 8.5, MUT, indent=10)
        elif line.startswith(("- ", "* ", "• ")):
            content = clean.strip()
            if content[:2] in ("- ", "• "):
                content = content[2:]
            emit("•  " + content.strip(), "Helvetica", 9, INK, indent=8)
        elif re.match(r"^\d+\.", line):
            emit(clean, "Helvetica", 9, INK, indent=8)
        else:
            emit(clean, "Helvetica", 9, INK)

    c.save()
    buf.seek(0)
    return buf
