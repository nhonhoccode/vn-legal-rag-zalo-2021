# VN Legal RAG — Vietnamese Legal Text Retrieval-Augmented Generation

Hệ thống truy xuất ngữ nghĩa và sinh câu trả lời tự nhiên cho văn bản pháp luật Việt Nam, được fine-tune trên bộ dữ liệu **Zalo AI Challenge 2021 — Legal Text Retrieval**. Pipeline kết hợp Dense retrieval (ChromaDB) + Sparse retrieval (BM25 + pyvi) + Cross-encoder reranker (`bge-reranker-v2-m3`) + LLM generation (Gemini 2.5 Flash qua OpenRouter), tinh chỉnh bốn mô hình embedding multilingual theo hai chiến lược lấy negative samples.

Live demo: <https://nlp-rag.votrongnhon.cloud>

---

## Kết quả thực nghiệm

Đánh giá trên 660 mẫu test (split 80/20, seed = 42) của Zalo AI Challenge 2021:

| Mô hình | Chiến lược | F2@10 | Recall@10 | MRR | nDCG@10 |
|:---|:---|---:|---:|---:|---:|
| **bge-m3** | **hardneg (ep1)** | **0,3209** | **0,8985** | **0,7444** | **0,7821** |
| gte-multilingual-base | hardneg (ep1) | 0,3117 | 0,8727 | 0,7127 | 0,7520 |
| multilingual-e5-base | inbatch (ep2) | 0,3057 | 0,8561 | 0,6540 | 0,7036 |
| vietnamese-sbert | hardneg (ep2) | 0,2787 | 0,7803 | 0,5690 | 0,6197 |

Bốn checkpoint đã được công bố public trên Hugging Face Hub:

- `nhonhoccode/zalo-legal-bge-m3-finetuned`
- `nhonhoccode/zalo-legal-e5-multilingual-base-finetuned`
- `nhonhoccode/zalo-legal-gte-multilingual-base-finetuned`
- `nhonhoccode/zalo-legal-vietnamese-sbert-finetuned`

---

## Quick start

Yêu cầu trên máy host: Docker 24+, Docker Compose v2, khoảng 6 GB RAM trống và 5 GB ổ đĩa cho images + Chroma index.

```bash
git clone git@github.com:nhonhoccode/vn-legal-rag-zalo-2021.git
cd vn-legal-rag-zalo-2021
bash scripts/setup.sh
```

Script `setup.sh` sẽ tự động: kiểm tra prerequisites, sinh `.env` từ template, generate `JWT_SECRET` random 32 chars, hỏi các API key tối thiểu, pull và build images, khởi động stack. Sau khi xong, frontend chạy tại `http://localhost:3000`, backend tại `http://localhost:8000`.

Các lệnh thường dùng sau setup:

```bash
docker compose ps                   # trạng thái container
docker compose logs -f api          # log backend realtime
docker compose down                 # dừng toàn bộ
docker compose up -d --build        # rebuild và khởi động lại
```

---

## Kiến trúc tổng quát

Hệ thống tổ chức theo ba tầng độc lập (chi tiết trong `report/`):

- **Frontend** — Next.js 15 + Tailwind + shadcn/ui + NextAuth.js (Google / GitHub OAuth). Chat UI với streaming SSE, citations clickable, filter theo ba domain pháp lý, multi-turn dialog.
- **Backend FastAPI** — điều phối intent classification (regex), standalone query rewriting (LLM), hybrid retrieval, prompt build với strict citation, LLM streaming.
- **External services** — Redis (cache LLM 24h, query embed 7d, session 1h), ChromaDB (40.266 vectors 1024-dim, HNSW), BM25 index (pyvi tokenizer), Cross-encoder reranker, OpenRouter API.

Toàn bộ chạy bằng Docker Compose, expose qua Cloudflare Tunnel ở môi trường production (zero-port-open, HTTPS tự động).

---

## Cấu trúc thư mục

```
.
├── backend/                # FastAPI + Pydantic + uvicorn
│   ├── src/                # RAG pipeline, retrieval, generation, eval, vectorstore
│   ├── scripts/            # preprocess, chunk, embed, import, eval
│   ├── tests/              # pytest unit + integration
│   └── Dockerfile
├── frontend/               # Next.js 15 App Router
│   ├── app/                # routes + API
│   ├── components/         # chat UI
│   ├── auth.ts             # NextAuth.js config
│   └── Dockerfile
├── notebooks/              # 4 fine-tune notebooks (Kaggle T4) + compare + embed
├── data/
│   ├── raw/                # Zalo AI 2021 corpus + qa (gitignored)
│   ├── processed/          # chunks.jsonl, articles_clean.jsonl (gitignored)
│   └── chroma_db/          # ChromaDB persistent (gitignored)
├── report/                 # Báo cáo đồ án + bài báo
│   ├── BaoCao_VN_Legal_RAG.docx                # Đồ án 45 trang
│   ├── BaiBao_VoTrongNhon_VNLegalRAG.docx      # Bài báo 12 trang
│   └── images/             # 4 diagram + screenshots + convergence curves
├── scripts/
│   └── setup.sh            # Auto installer
├── nginx/                  # Reverse proxy config
├── docker-compose.yml      # Dev stack
├── docker-compose.prod.yml # Prod stack (Cloudflare Tunnel)
└── .env.example            # Top-level env template
```

---

## Cài đặt thủ công (không dùng setup.sh)

### Yêu cầu

- Docker 24+, Docker Compose v2
- Python 3.11+ (chỉ cần khi chạy backend trực tiếp không qua Docker)
- Node.js 20+ (chỉ cần khi chạy frontend trực tiếp)
- API key của OpenRouter hoặc Google AI Studio (cho LLM Gemini)

### Bước 1 — Clone và cấu hình env

```bash
git clone git@github.com:nhonhoccode/vn-legal-rag-zalo-2021.git
cd vn-legal-rag-zalo-2021
cp .env.example .env
cp backend/.env.example backend/.env
cp frontend/.env.example frontend/.env.local
```

Mở `backend/.env` và điền tối thiểu:

```bash
LLM_PROVIDER=openrouter
LLM_MODEL=google/gemini-2.5-flash
OPENROUTER_API_KEY=sk-or-v1-...
JWT_SECRET=$(openssl rand -hex 32)
```

`JWT_SECRET` phải giống với `NEXTAUTH_SECRET` trong `frontend/.env.local`.

### Bước 2 — Tải dữ liệu Zalo AI Challenge 2021

```bash
mkdir -p data/raw/zalo_legal
# Tải corpus.jsonl và qa.jsonl từ https://challenge.zalo.ai/
# rồi đặt vào data/raw/zalo_legal/
```

### Bước 3 — Build ChromaDB index

Có hai cách:

**Cách A — Build từ scratch trên máy local (chậm, khoảng 30 phút trên CPU):**

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/02_preprocess.py
python scripts/03_chunk.py
python scripts/04_embed.py
python scripts/05_import.py
```

**Cách B — Tải pre-built ChromaDB từ GitHub Releases (nhanh, khoảng 2 phút):**

```bash
# Tải chroma_db.zip từ GitHub Releases và giải nén vào data/chroma_db/
```

### Bước 4 — Khởi động stack

```bash
docker compose up -d
docker compose logs -f api
```

Mở browser tại `http://localhost:3000`.

---

## Fine-tune embedding trên Kaggle GPU T4

Bốn notebook fine-tune độc lập trong `notebooks/`:

| Notebook | Mô hình | Thời gian (T4) |
|:---|:---|---:|
| `01_finetune_bge_m3.ipynb` | `BAAI/bge-m3` (560M) | ~2,5 giờ |
| `02_finetune_e5_multilingual_base.ipynb` | `intfloat/multilingual-e5-base` (270M) | ~1,4 giờ |
| `03_finetune_gte_multilingual_base.ipynb` | `Alibaba-NLP/gte-multilingual-base` (305M) | ~1,4 giờ |
| `04_finetune_vietnamese_sbert.ipynb` | `keepitreal/vietnamese-sbert` (66M) | ~1,1 giờ |
| `Compare_model.ipynb` | (CPU, so sánh 12 cấu hình) | ~10 phút |

Mỗi notebook chạy quy trình hai pha:

1. **Pha 1** — In-batch negatives qua `MultipleNegativesRankingLoss`, 5 epoch tối đa, eval F2@10 sau mỗi epoch, early stopping với patience = 2.
2. **Pha 2** — Hard negatives mining từ BM25 top-30, warm-start từ checkpoint Pha 1, learning rate giảm còn 1e-5.

Sau khi train xong, notebook tự push checkpoint thắng cuộc lên Hugging Face Hub.

Để thay model đang dùng trong production, sửa `backend/.env`:

```bash
EMBEDDING_MODEL=nhonhoccode/zalo-legal-bge-m3-finetuned
```

Rồi re-encode corpus qua `notebooks/kaggle_embed.ipynb` để tạo ChromaDB collection mới.

---

## Đánh giá retrieval

Backend có sẵn script đánh giá retrieval trên test set Zalo:

```bash
cd backend
python scripts/05_eval.py \
  --golden ../data/raw/zalo_legal/qa.jsonl \
  --top-k 10 \
  --rerank
```

Output là `data/eval/results/eval_<timestamp>.json` với đầy đủ F2@k, Recall@k, Precision@k, MRR, nDCG@k cho k ∈ {1, 5, 10}.

---

## Triển khai production qua Cloudflare Tunnel

Đã chạy production tại `nlp-rag.votrongnhon.cloud`:

```bash
# Trên VPS (Ubuntu 22.04+)
git clone git@github.com:nhonhoccode/vn-legal-rag-zalo-2021.git
cd vn-legal-rag-zalo-2021
cp .env.example .env
# Điền CLOUDFLARE_TUNNEL_TOKEN, DOMAIN_FRONTEND, DOMAIN_API trong .env
# Điền OPENROUTER_API_KEY, JWT_SECRET trong backend/.env

docker compose -f docker-compose.prod.yml up -d
```

Cloudflare Tunnel cần được cấu hình trước tại Cloudflare Zero Trust → Tunnels → New tunnel.

---

## Tech stack

- **Backend**: Python 3.11, FastAPI, Pydantic v2, uvicorn, ChromaDB, Redis, sentence-transformers 3.3.1, pyvi, rank_bm25, OpenRouter SDK
- **Frontend**: Next.js 15 (App Router), TypeScript, TailwindCSS, shadcn/ui, NextAuth.js, Server-Sent Events
- **Embedding**: `BAAI/bge-m3` fine-tuned (1024-dim, sản phẩm chính)
- **Reranker**: `BAAI/bge-reranker-v2-m3` (cross-encoder)
- **LLM**: `google/gemini-2.5-flash` qua OpenRouter API
- **Infrastructure**: Docker 24+, Docker Compose v2, Cloudflare Tunnel
- **CI/CD**: GitHub Actions

---

## Báo cáo và bài báo khoa học

- [`report/BaoCao_VN_Legal_RAG.docx`](report/BaoCao_VN_Legal_RAG.docx) — Báo cáo đồ án 45 trang theo format đại học, gồm Cover / Lời cảm ơn / Cam đoan / Đánh giá GV / Tóm tắt / Mục lục / Danh mục / 5 Chương / Tài liệu tham khảo / Phụ lục / Tự đánh giá theo rubric.
- [`report/BaiBao_VoTrongNhon_VNLegalRAG.docx`](report/BaiBao_VoTrongNhon_VNLegalRAG.docx) — Bài báo khoa học 12 trang theo format conference paper, gồm Tóm tắt / Giới thiệu / Nghiên cứu liên quan / Phương pháp / Thực nghiệm / Kết luận / 13 references IEEE.

---

## Tài liệu tham khảo

- Karpukhin et al. (2020). *Dense Passage Retrieval for Open-Domain Question Answering*. EMNLP.
- Reimers & Gurevych (2019). *Sentence-BERT*. EMNLP-IJCNLP.
- Chen et al. (2024). *BGE M3-Embedding*. arXiv:2402.03216.
- Cormack et al. (2009). *Reciprocal Rank Fusion*. SIGIR.
- Robertson & Zaragoza (2009). *BM25 and Beyond*.
- Lewis et al. (2020). *Retrieval-Augmented Generation*. NeurIPS.
- Zalo AI Challenge (2021). *Legal Text Retrieval Track*. <https://challenge.zalo.ai/>

---

## Tác giả

**Võ Trọng Nhơn** — MSSV 22658441, Lớp DHKHDL18A, Khoa Công nghệ Thông tin, Trường Đại học Công nghiệp TP. Hồ Chí Minh.

**Giảng viên hướng dẫn**: TS. Bùi Thanh Hùng.

Đồ án cuối kì học phần Xử lý ngôn ngữ tự nhiên, năm học 2025–2026.

---

## License

MIT License. Bộ dữ liệu Zalo AI Challenge 2021 thuộc bản quyền của Zalo Group và phải tuân theo điều khoản công bố tại <https://challenge.zalo.ai/>.
