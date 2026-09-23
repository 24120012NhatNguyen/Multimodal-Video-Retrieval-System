# Phân Tích Kiến Trúc Hệ Thống: Multimodal Video Retrieval System

> Hệ thống tìm kiếm video đa phương thức cho cuộc thi AIC2026, xử lý dữ liệu video tin tức tiếng Việt.

---

## 1. Tổng Quan Kiến Trúc

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        FRONTEND (Next.js)                                │
│  Tìm kiếm văn bản │ Panel object │ TRAKE │ Q/A │ Autofill nộp bài        │
└─────────────────────────────┬────────────────────────────────────────────┘
                              │  REST API
          ┌───────────────────┴──────────────────────┐
          │      Backend LOCAL (socket_app.py)        │  ← Q/A + submit
          │      Backend KAGGLE (app.py / FastAPI)    │  ← tìm kiếm chính
          └───────────┬──────────────────────────────┘
                      │
          ┌───────────┴──────────────────────────────┐
          │          Retrieval Engine                 │
          │  ┌──────────┐ ┌─────────┐ ┌──────────┐  │
          │  │  SigLIP  │ │  BM25   │ │ Objects  │  │
          │  │  Visual  │ │ Text    │ │  Index   │  │
          │  │  Channel │ │ Channels│ │ (panel)  │  │
          │  └──────────┘ └─────────┘ └──────────┘  │
          │              RRF Fusion                   │
          │          DP Temporal Alignment            │
          └───────────────────────────────────────────┘
                      │
          ┌───────────┴──────────────────────────────┐
          │            ArtifactStore                  │
          │  features/*.npy  |  keyframes/*.csv       │
          │  asr/*.json      |  ocr/*.json            │
          │  metadata/*.json | objects/*.json         │
          └───────────────────────────────────────────┘
```

---

## 2. Pipeline Xử Lý Dữ Liệu (Offline)

### 2.1 Trích Xuất Keyframe

**Script:** [`extract_keyframes_safe.py`](file:///Users/ngovietthanhbinh/Project/AIC2026/AIC26/Multimodal-Video-Retrieval-System/scripts/extract_keyframes_safe.py)

- Đọc `data/keyframe.zip` (dữ liệu từ BTC)
- Hỗ trợ **resume** khi bị gián đoạn: lưu trạng thái JSON
- Giải nén từ cuối lên đầu (truncate-as-you-go) → tiết kiệm disk 
- Ghi file theo kiểu **atomic** (write → `.tmp` → rename) để tránh corrupt

### 2.2 Trích Xuất Visual Features — SigLIP

**Model:** `google/siglip-so400m-patch14-384` (1152 chiều)

| Artifact | Vị trí | Định dạng |
|---|---|---|
| Visual embeddings | `data/artifacts/<pack>/features/<video_id>.npy` | float32, (N×1152), L2-normalized |
| Keyframe metadata | `data/artifacts/<pack>/keyframes/<video_id>.csv` | n, frame_idx, pts_time, lap_var, cos_to_prev |

**Điều kiện nguy hiểm được chặn:**
- `npy` và `csv` phải khớp số dòng hoàn toàn — lệch → bỏ cả video
- Encoder phải đúng chiều (1152) với `ArtifactStore.assert_encoder_matches()`

### 2.3 ASR (Automatic Speech Recognition)

- **Model:** Whisper → transcript tiếng Việt có dấu đúng
- **Lưu:** `data/artifacts/<pack>/asr/<video_id>.json`
- **BM25 tokenizer:** `tokenize_vi` (giữ dấu tiếng Việt)

### 2.4 OCR (Optical Character Recognition)

- Chạy trên từng keyframe → chuỗi text
- **Lưu:** `data/artifacts/<pack>/ocr/<video_id>.json`
- **Vấn đề đặc thù:** OCR chạy sai model ngôn ngữ → dấu tiếng Việt **sai** (`NO LU'C` ← `no luc`)
- **BM25 tokenizer:** `tokenize_fold` (bỏ toàn bộ dấu thanh, dấu nháy)

### 2.5 Object Detection Index

**Module:** [`retrieval/objects.py`](file:///Users/ngovietthanhbinh/Project/AIC2026/AIC26/Multimodal-Video-Retrieval-System/retrieval/objects.py)

- Đọc `data/artifacts/objects-aic25-b1/objects/<video_id>/*.json` (~177K file JSON, 2.1 GB)
- **Đóng gói thành 1 file `.npz`** (~50 MB, mmap) — CSR format giống sparse matrix:
  - `kf_ptr`, `kf_video`, `kf_n`, `kf_pts` — keyframe
  - `det_ent`, `det_score`, `det_box` — detection phẳng
- Ngưỡng lọc detection: **score ≥ 0.15** (bỏ ~80% detection nhiễu)
- Tọa độ bbox: OpenImages `[ymin, xmin, ymax, xmax]` (chú ý: UI gọi trục dọc là "x" → mapping ngược)

---

## 3. ArtifactStore — Lớp Dữ Liệu Runtime

**File:** [`retrieval/store.py`](file:///Users/ngovietthanhbinh/Project/AIC2026/AIC26/Multimodal-Video-Retrieval-System/retrieval/store.py)

```python
class ArtifactStore:
    X: np.ndarray     # (N_total, 1152) float32 – toàn bộ visual embeddings
    meta: DataFrame   # N_total dòng: gidx, video_id, pack, frame_idx, pts_time, ...
    video_slice: dict # video_id → (start, end) slice trên X
    fps: dict         # video_id → fps tính từ frame_idx/pts_time
```

- `gidx` = chỉ số toàn cục (global index) → khóa bất biến, dùng xuyên suốt
- Tất cả phép lọc/sắp xếp đều đi qua `gidx` → không bao giờ lệch giữa `X` và `meta`
- Tính FPS từ dữ liệu thực (median), không hardcode

---

## 4. Pipeline Xử Lý Truy Vấn (Online)

### 4.1 Bước 1: Query Decomposition

**File:** [`retrieval/query.py`](file:///Users/ngovietthanhbinh/Project/AIC2026/AIC26/Multimodal-Video-Retrieval-System/retrieval/query.py)

```
Query tiếng Việt
    │
    ├─→ [Bậc 1] LLM (Claude Haiku / Gemini Pro)
    │     └─→ JSON: {clauses_en[], query_vi, kind, anchors[], question}
    │
    ├─→ [Bậc 2] deep-translator (Google Translate)
    │     └─→ translate + phân loại heuristic
    │
    └─→ [Bậc 3] Không-LLM
          └─→ query_en = None (KHÔNG dùng tiếng Việt cho SigLIP)
```

**Output là `DecomposedQuery`:**

| Trường | Ý nghĩa |
|---|---|
| `query_en` | Mệnh đề tiếng Anh (nhiều dòng) → cho SigLIP |
| `query_vi` | Tiếng Việt giữ nguyên → cho BM25 |
| `clauses_en[]` | Từng mệnh đề riêng → cho DP alignment |
| `kind` | `anchored` / `generic_chain` |
| `anchors[]` | Chi tiết hiếm (tên riêng, biển hiệu) |
| `question` | Câu hỏi cần VLM đọc hình (Q&A task) |

**Phân loại `kind`:**
- `anchored`: Có chi tiết hiếm → tìm phẳng đủ, **không** cần DP
- `generic_chain`: Toàn cảnh phổ biến → **phải** dóng hàng thời gian bằng DP

**Cache:**
- Translation cache: `data/translation_cache.json` (Google Translate không ổn định)
- Decompose cache: `data/decompose_cache.json` (LLM không tất định)

### 4.2 Bước 2: Fusion Engine

**File:** [`retrieval/engine.py`](file:///Users/ngovietthanhbinh/Project/AIC2026/AIC26/Multimodal-Video-Retrieval-System/retrieval/engine.py)

#### 4.2.1 Kênh SigLIP (Visual)

```python
Q = encoder.encode_texts(queries)  # (nq, 1152)
S = store.X @ Q.T                  # (N, nq) cosine similarity
# Group by video → max score per query → sum across queries
```

**`pack_queries()`** — xử lý giới hạn 64 token của SigLIP:
- Gộp nhiều mệnh đề thành **ít nhóm nhất có thể**, mỗi nhóm ≤ 64 token
- Nếu không vừa 1 nhóm → chia thành nhiều nhóm, trung bình điểm
- Lý do: SigLIP cắt ngầm ở 64 token → mất chi tiết mà không báo lỗi

#### 4.2.2 Kênh BM25 (Text)

4 kênh, mỗi kênh là một `MetaIndex` BM25:

| Kênh | Tokenizer | Nguồn dữ liệu |
|---|---|---|
| `meta` | `tokenize_vi` (giữ dấu) | Metadata BTC (title×3, description, keywords) |
| `meta_fold` | `tokenize_fold` (bỏ dấu) | Cùng metadata → hỗ trợ query thiếu dấu |
| `asr` | `tokenize_vi` | Whisper transcript |
| `ocr` | `tokenize_fold` | OCR text (bỏ dấu do OCR lỗi dấu) |

**BM25 score gate** (`SCORE_GATE_RATIO = 0.35`):
- Chỉ tính video có điểm ≥ 35% điểm cao nhất của kênh
- Phòng: BM25 không có tín hiệu thật vẫn trả ra danh sách có thứ hạng → cộng nhiễu vào RRF

#### 4.2.3 Weights (Trọng số theo loại query)

```python
# LUÔN là visual-only theo mặc định (sau nhiều lần đo)
_AUTO_WEIGHTS = {"siglip": 1.0, "meta": 0.0, "asr": 0.0, "ocr": 0.0}
```

Lý do: Thêm bất kỳ trọng số BM25 nào với query thị giác đều làm tệ hơn. BM25 không có tín hiệu thật nhưng vẫn trả ra danh sách có thứ hạng → RRF tính điểm cho cả "nhiễu".

Người dùng có thể bật kênh text thủ công → hệ thống đặt weight ≥ 1.0 cho kênh đó.

#### 4.2.4 RRF Fusion

**File:** [`fusion.py`](file:///Users/ngovietthanhbinh/Project/AIC2026/AIC26/Multimodal-Video-Retrieval-System/fusion.py)

```python
score[vid] += w / (k + rank)   # k=60, w=trọng số kênh
```

- Dùng **thứ hạng**, không phải điểm → không cần chuẩn hóa giữa cosine và BM25
- Gộp `meta` và `meta_fold` thành 1 nhóm nguồn → lấy thứ hạng tốt nhất (không bỏ phiếu 2 lần)

#### 4.2.5 Frame Reranking

Sau RRF xếp hạng video, tính điểm SigLIP cao nhất trong mỗi video → xếp lại:

```
RRF biết "video được bao nhiêu kênh bỏ phiếu"
Frame rerank biết "video có khoanh khắc nào thật sự giống query"
```

#### 4.2.6 Frame Allocation

Ba chế độ (config `frame_alloc`):
- `global`: sort toàn cục → video không có frame điểm cao biến mất khỏi lưới
- `round_robin`: 1 frame/video/vòng → video luôn hiện nhưng recall thấp
- `hybrid`: đảm bảo 1 frame cho top 1/3 video, phần còn lại theo điểm

### 4.3 Bước 3: DP Temporal Alignment (TRAKE)

**File:** [`retrieval/trake.py`](file:///Users/ngovietthanhbinh/Project/AIC2026/AIC26/Multimodal-Video-Retrieval-System/retrieval/trake.py)

#### Bài toán:
Tìm dãy frame **tăng dần theo thời gian** sao cho tổng điểm khớp chuỗi sự kiện là lớn nhất, cho phép bỏ qua sự kiện với một khoản phạt.

#### Z-score normalization (event_stats):
```python
# Tính phân bố điểm của từng mệnh đề trên corpus
S = store.X[::37] @ Q.T    # sample 1/37 → ~4180 điểm
mu, sigma = S.mean(0), S.std(0)
# Dùng cho mọi video → thang đo thống nhất
z_score = (cosine - mu) / sigma - tau   # tau=2.0
```

- **Tại sao cần z-score:** 4 mệnh đề cùng query có trung bình cosine từ -0.015 đến +0.009 (khác nhau ~4×). Cộng thẳng cosine thô = mệnh đề "dễ ăn điểm" được tính nặng hơn.
- **Tại sao cần `tau=2.0`:** Nếu không có ngưỡng, sự kiện vắng mặt VẪN có frame nào đó trong video có z > 0 → DP không bao giờ chọn nhánh "bỏ qua".

#### DP Algorithm — K-Best Monotonic DP (O(N·K·K_best log(N·K_best))):
- Khác với thuật toán 1-Best thông thường (chỉ lưu 1 trạng thái tốt nhất cho mỗi ô `dp[i][k]`), hệ thống sử dụng **K-Best Monotonic DP** để giữ lại Top-K đường đi xuất sắc nhất.
- Trạng thái: `dp[i][k][b]` = điểm của đường đi xếp hạng thứ `b` (từ 1 đến `K_best`) khi đã khớp `k` sự kiện, kết thúc tại frame `i`.
- **Max-Heap với Lazy Deletion**: Để tối ưu việc tìm kiếm Top-K trong cửa sổ trượt `[t[i]-delta, t[i]-min_gap]`, một Heap (hàng đợi ưu tiên) được sử dụng. Thay vì xóa phần tử khi cửa sổ trượt qua (phức tạp $O(N)$), hệ thống sử dụng Lazy Deletion (chỉ loại bỏ phần tử hết hạn khi trích xuất nó khỏi Heap). 
- Tại mỗi ô, thuật toán kết hợp cả các nhánh **khớp sự kiện k** và nhánh **bỏ qua sự kiện k** (trừ điểm phạt `gamma`), sắp xếp và giữ lại chính xác `K_best` ứng viên tốt nhất.
- **Backtracking** (lần ngược): Dò ngược từ các đỉnh tốt nhất ở bước `K`, sử dụng Deduplication (tập hợp `set`) để đảm bảo sinh ra chính xác `K_best` chuỗi thời gian (temporal sequences) hoàn toàn phân biệt.

#### Lấp khoảng trống (`fill_skipped`):
- DP trả về `-1` cho sự kiện bị bỏ qua
- Cuộc thi TRAKE yêu cầu 1 frame cho mỗi sự kiện → lấy frame tốt nhất trong khoảng thời gian hợp lệ

---

## 5. Tìm Kiếm Object / Panel

**File:** [`retrieval/panels.py`](file:///Users/ngovietthanhbinh/Project/AIC2026/AIC26/Multimodal-Video-Retrieval-System/retrieval/panels.py), [`retrieval/objects.py`](file:///Users/ngovietthanhbinh/Project/AIC2026/AIC26/Multimodal-Video-Retrieval-System/retrieval/objects.py)

### Luồng `/panel`:
```
1. Filter text: OCR/ASR BM25 → thu hẹp video
2. Filter object: lớp + vị trí (IoU) + số lượng
3. Bắc cầu: keyframe BTC → keyframe của ta qua pts_time (±2.5s)
4. Xếp theo: conf * IoU, vị trí được tính nặng 2×
```

### Các tính năng:
- **IoU-based spatial search**: kéo thả object lên panel → bbox → IoU với detection
- **Ràng buộc số lượng**: `Person 2` → chỉ frame có ≤ 2 person
- **Alias COCO→OpenImages**: UI dùng 80 lớp COCO, dữ liệu là 545 lớp OpenImages

---

## 6. Tìm Kiếm Phụ

### `/imgsearch` — KNN từ vector ảnh
- Lấy vector của keyframe `video_id#frame_idx` → `X @ v` → top-k tương đồng nhất
- Dùng `argpartition` → O(N) thay vì O(N log N)

### `/framerange` — Xem toàn video theo đoạn thời gian
- Query SigLIP trong [start, end] frame_idx
- Nếu không có keyframe trong đoạn → tạo frame ảo từ fps

### `/relatedimg` — Bối cảnh xung quanh ảnh
- `span` frame trước/sau trên trục thời gian
- Trả về link YouTube gốc + mốc giây + link clip ngắn

### `/feedback` — Rocchio Relevance Feedback
```python
q = mean(pos_vectors) - 0.35 * mean(neg_vectors)
q = q / ||q||
sims = store.X @ q    # tìm kiếm bằng vector mới
```
Beta = 0.35 (nhỏ hơn 1): ảnh sai chỉ đẩy ra xa, không lấn át hướng ảnh đúng.

---

## 7. Q&A — Visual Question Answering

**File:** [`retrieval/qa.py`](file:///Users/ngovietthanhbinh/Project/AIC2026/AIC26/Multimodal-Video-Retrieval-System/retrieval/qa.py)

```
Question → dịch sang EN → encode_query (tự chia ≤64 token)
         → pick_frames: điểm = W_relevance×SigLIP + W_sharpness×lap_var
         → VLM (Haiku/Flash tier): OCR + ASR + ảnh → câu trả lời
         → fallback: trả OCR/ASR thô nếu VLM lỗi
```

Hoạt động trên **server local** (có ảnh thật từ `data/videos`), không trên Kaggle.

---

## 8. Autofill — Tự động điền 100 ô nộp bài

**File:** [`retrieval/autofill.py`](file:///Users/ngovietthanhbinh/Project/AIC2026/AIC26/Multimodal-Video-Retrieval-System/retrieval/autofill.py)

Công thức điểm BTC:
```
R@1=1.0, R@2-5=0.80, R@6-20=0.60, R@21-50=0.40, R@51-100=0.20
Final = mean(R@1, R@5, R@20, R@50, R@100)
```

### Chiến lược autofill (Deferred MMR):
1. **Rank-Preserving Zone (Ô 1-20)**: Giữ nguyên thứ hạng tìm kiếm gốc của hệ thống (vì điểm RRF/SigLIP trên top đầu có độ chính xác rất cao). Các frame do người dùng chọn thủ công ưu tiên tuyệt đối đứng đầu.
2. **Densify (Xen frame)**: Xung quanh các frame top đầu, lấy thêm các frame lân cận (cách 25 frame/1 giây) để đảm bảo không bỏ sót keyframe quan trọng do phân phối thưa thớt (trung vị BTC là 75 frame).
3. **Deferred MMR (Ô 21-100)**: Áp dụng thuật toán **Deferred Maximal Marginal Relevance** để đa dạng hóa kết quả mà không làm hỏng nhóm đầu.
   - Hàm mục tiêu: `MMR_Score = (1 - lambda) * Relevance - lambda * Penalty`
   - **Relevance**: Điểm gốc Z-score hoặc RRF được Min-Max chuẩn hóa về khoảng `[0, 1]`.
   - **Penalty**: Hình phạt khoảng cách thời gian `max(0, 1.0 - delta_time / tail_gap)` nếu frame đang xét nằm trong cùng một video và quá gần (về mặt thời gian) với bất kỳ frame nào đã được chọn. Nếu khác video thì Penalty = 0.
   - Thuật toán vòng lặp tham lam (Greedy loop) sẽ liên tục chọn candidate có `MMR_Score` cao nhất để điền nốt các slot còn lại.

Chiến lược này giải quyết được điểm yếu của thuật toán MMR toàn cục (MMR thường đẩy các kết quả rất xuất sắc xuống dưới vì chúng quá giống nhau về mặt thời gian/visual). Bằng cách "trì hoãn" (defer) MMR cho tới vị trí thứ 21, hệ thống vừa bảo toàn được độ chính xác tuyệt đối ở top đầu, vừa đạt được độ đa dạng (diversity) mong muốn cho 80 ô còn lại.

---

## 9. LLM Client & Circuit Breaker

**File:** [`retrieval/llm_client.py`](file:///Users/ngovietthanhbinh/Project/AIC2026/AIC26/Multimodal-Video-Retrieval-System/retrieval/llm_client.py)

- **Hai tier:** `flash` (Q&A, nhanh rẻ) và `pro` (phân rã query)
- **Circuit breaker:** ≥3 lỗi liên tiếp → ngưng gọi API trong 60s
- **Không hardcode model ID** → biến môi trường hoàn toàn
- **Cảnh báo alias:** phát hiện model ID thiếu hậu tố version (alias không ổn định)
- **Fallback bắt buộc:** mọi lỗi → `LLMResult(ok=False)`, không throw exception

---

## 10. API Endpoints

| Endpoint | Phương thức | Mô tả |
|---|---|---|
| `POST /textsearch` | POST | **Tìm kiếm chính** (SigLIP + BM25 + RRF + DP) |
| `POST /panel` | POST | Tìm theo object lớp + vị trí + OCR/ASR |
| `GET /imgsearch` | GET | KNN từ vector ảnh |
| `GET /relatedimg` | GET | Bối cảnh thời gian của ảnh |
| `GET /framerange` | GET | Toàn bộ keyframe trong đoạn video |
| `GET /getvideoshot` | GET | Toàn bộ video, nhóm theo phút |
| `POST /feedback` | POST | Rocchio relevance feedback |
| `POST /autofill` | POST | Tự động điền 100 ô nộp bài |
| `POST /trake` | POST | TRAKE: dóng hàng chuỗi sự kiện |
| `POST /keyframe_context` | POST | Object/OCR/ASR quanh 1 keyframe |
| `POST /translate` | POST | Dịch query hiển thị trên UI |
| `GET /data` | GET | Từ vựng lớp object, danh sách video |
| `GET /diagnostics` | GET | Kiểm tra sức khỏe hệ thống |

---

## 11. Kỹ Thuật Đặc Biệt

### 11.1 Silent Failure Prevention
Toàn bộ hệ thống được thiết kế để tránh "lỗi im lặng":
- SigLIP cắt ngầm ở 64 token → `pack_queries()` chia trước
- Google Translate trả HTML lỗi → `_looks_like_translation()` kiểm tra
- OCR sai dấu → `tokenize_fold()` cho toàn bộ kênh OCR
- Vector SigLIP sai model → `assert_encoder_matches()` chặn sớm

### 11.2 Token Budget Management
```python
# Truy vấn 4 mệnh đề → 93 token → vượt 64 token của SigLIP
# pack_queries gộp thành các nhóm vừa 64 token
# Cùng 1 query được encode thành nhiều chunk → điểm trung bình (phép HỘI)
```

### 11.3 Dual-Language Strategy
- `query_en` → SigLIP (tiếng Anh, hiểu nghĩa thị giác)
- `query_vi` → BM25 (tiếng Việt, khớp tên riêng chính xác: "Nhân Nghĩa Đường")

### 11.4 Temporal Refinement
- **Lọc kết quả cũ:** `restrict_gidx` — chỉ giữ frame trong cửa sổ `±refine_window_sec` của kết quả vòng trước
- **Lọc hướng thời gian:** `filtervideo=1/2` — chỉ frame SAU hoặc TRƯỚC mốc cũ

### 11.5 Evaluation-Driven Design
Mọi quyết định thiết kế đều có số đo:

```
Trọng số kênh:
  siglip 6 / text 0.3    →  R@1 42.9%
  chỉ siglip             →  R@1 57.1%  ← dùng

Frame allocation:
  global sort            →  Recall@10 frame 71.4%
  round robin            →  Recall@10 frame 42.9%
  hybrid                 →  (cân bằng)

Autofill:
  top-100 thô            →  Final 0.5714
  MMR lambda=0.7         →  Final 0.3429  ← tệ hơn!
  chiến lược hiện tại    →  Final 0.6286
```

---

## 12. Cấu Trúc Thư Mục

```
Multimodal-Video-Retrieval-System/
├── app.py                    # FastAPI (Kaggle/backend chính)
├── socket_app.py             # Backend local (Q/A, submit)
├── fusion.py                 # RRF, BM25 MetaIndex, siglip_video_rank
├── config/
│   └── fusion.json           # Hyperparameters tunable
├── retrieval/
│   ├── engine.py             # FusionEngine (core)
│   ├── query.py              # DecomposedQuery + LLM fallback ladder
│   ├── store.py              # ArtifactStore (X, meta)
│   ├── encoder.py            # SigLipTextEncoder (lazy load)
│   ├── textindex.py          # TextChannels (4 BM25)
│   ├── objects.py            # ObjectIndex (CSR .npz)
│   ├── panels.py             # PanelSearch (/panel, /imgsearch, /relatedimg)
│   ├── trake.py              # DP temporal alignment
│   ├── autofill.py           # Autofill 100 ô
│   ├── qa.py                 # VLM Q&A
│   ├── llm_client.py         # LLM abstraction + circuit breaker
│   ├── config.py             # FusionConfig dataclass
│   ├── frames.py             # On-demand keyframe extraction từ MP4
│   ├── bridge.py             # Context tại 1 keyframe (OCR/ASR/Object)
│   ├── answers.py            # Schema câu trả lời
│   ├── clips.py              # Clip ngắn từ MP4
│   └── service.py            # Khởi tạo lazy singleton
├── scripts/
│   ├── extract_keyframes_safe.py  # Giải nén có resume
│   ├── normalize_ocr.py           # Chuẩn hóa OCR
│   ├── prewarm_keyframes.py       # Preheat cache
│   └── reset_session.py           # Reset session đáp án
├── frontend/                 # Next.js UI
└── data/
    ├── artifacts/            # Features, keyframes, ASR, OCR, metadata, objects
    ├── videos/               # MP4 gốc (local only)
    └── keyframe_cache/       # Cache ảnh on-demand
```

---

## 13. Điểm Cần Chú Ý Khi Phát Triển

> [!WARNING]
> **SigLIP 64-token limit**: Luôn dùng `pack_queries()` hoặc `encode_query()` cho query dài, KHÔNG gọi thẳng `encode_texts()` với text dài.

> [!WARNING]
> **Object coordinate system**: OpenImages `[ymin, xmin, ymax, xmax]` nhưng UI panel gọi trục dọc là "x". Mapping sai → kết quả lật chéo.

> [!IMPORTANT]
> **DP tau parameter**: `dp_tau=2.0` là ngưỡng z-score để sự kiện được coi là "có mặt". Nếu đặt quá thấp, DP không bao giờ bỏ qua sự kiện nào. Nếu quá cao, bỏ qua sự kiện thật.

> [!NOTE]
> **Dual backend**: Q&A chạy trên `socket_app.py` (local, có ảnh MP4), tìm kiếm chính chạy trên `app.py` (Kaggle). Hai backend cùng tồn tại song song.
