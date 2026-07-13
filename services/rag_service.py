import json
import numpy as np
import faiss

from services.gemini_service import GeminiService


class RagService:
    def __init__(self, faiss_index_path, chunks_path, google_api_key=None, gemini_service: GeminiService = None):
        self.index = faiss.read_index(faiss_index_path)
        with open(chunks_path, encoding="utf-8") as f:
            self.chunks = json.load(f)
        # boleh pakai instance GeminiService yang sudah ada (dishare dari main.py),
        # atau bikin baru kalau dipanggil berdiri sendiri.
        self.gemini = gemini_service or GeminiService(api_key=google_api_key)

    def retrieve(self, query, top_k=3):
        q_emb = self.gemini.embed([query], task_type="RETRIEVAL_QUERY")
        faiss.normalize_L2(q_emb)
        scores, idxs = self.index.search(q_emb, top_k)
        return [(self.chunks[i], float(scores[0][j])) for j, i in enumerate(idxs[0]) if i != -1]

    def generate_report(self, stats: dict, top_k=3):
        query = (f"Priority {stats['priority']}, damage {stats['damage_percentage']}%. "
                 f"Rekomendasi alokasi sumber daya dan evakuasi.")
        retrieved = self.retrieve(query, top_k=top_k)
        context = "\n".join(f"- {chunk}" for chunk, _score in retrieved)

        prompt = f"""
Anda adalah analis triase bencana AI. Buat laporan penilaian kerusakan singkat dalam Bahasa Indonesia.

ATURAN:
1. Semua angka HARUS berasal dari JSON di bawah. Jangan mengarang angka.
2. Gunakan KONTEKS SOP di bawah untuk mendasari rekomendasi Anda.
3. Sebutkan recommended_action, evacuation_radius_km, dan required_logistics secara eksplisit.

KONTEKS SOP (hasil retrieval FAISS):
{context}

DATA:
{json.dumps(stats, indent=2)}
"""
        report_text = self.gemini.generate_content(prompt, timeout=60)
        return report_text, retrieved
