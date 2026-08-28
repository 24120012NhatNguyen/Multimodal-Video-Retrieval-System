# Hệ truy vấn video — AI Challenge 2026

Tìm khoảnh khắc trong 873 video tin tức tiếng Việt bằng mô tả
tiếng Việt. 

# Phiên bản hiện tại : 

Có người ngồi trong vòng lặp: máy đưa bạn tới đúng vùng, bạn chốt.

Kiến trúc tách đôi, cố ý:

| | chạy ở đâu | làm gì |
|---|---|---|
| `app.py` (:8080) | nơi có GPU | tìm kiếm — không giữ trạng thái |
| `socket_app.py` (:8081) | **luôn ở máy bạn** | đáp án, ảnh, clip, Hỏi–Đáp |
| `frontend/` (:3000) | máy bạn | giao diện |

Mọi thứ có trạng thái phải ở local: phiên Kaggle chết là mất sạch.

---

## 1. Dữ liệu

```
data/
├── artifacts/          2,6 GB  ← BẮT BUỘC
│   ├── L21 … L30/      features (.npy) + keyframes (.csv) + asr/ + ocr/
│   ├── map-keyframes/  mốc thời gian của keyframe BTC
│   ├── media-info-aic25-b1/    metadata YouTube
│   ├── objects-aic25-b1/       detection thô
│   └── object_index.npz  27 MB  ← index object đã nén sẵn
├── keyframe_cache/    32 GB   ← NÊN CÓ (154.640 ảnh, 873 video)
└── videos/                    ← TUỲ CHỌN (BTC không chia sẻ)
```

| | tải ở đâu |
|---|---|
| **artifacts** | https://www.kaggle.com/datasets/nhatnguyenhcmusk24/artifact-v4 |
| **keyframe_cache** | https://drive.google.com/drive/u/0/folders/1hFGfwJvyKpm2PnWGhuPhgwNcBy8emHB1 |
| **videos** | nguồn của BTC, không chia sẻ |

### Không có `data/videos` thì sao

**Không sao cả** — hệ chạy bình thường, . Chỉ mất hai thứ:

- xem **clip ±(2,4,8)s** quanh một frame
- trích ảnh cho frame **chưa có** trong `keyframe_cache`

Các endpoint đó trả HTTP 404 kèm lý do rõ ràng, **không ném lỗi, không sập**.
Tìm kiếm, chấm điểm, nộp bài không phụ thuộc video gốc.

Có videos thì trích được frame bất kỳ — kể cả frame *nằm giữa* các keyframe, thứ
mà cơ chế lấp 100 dòng dùng tới.

### Nếu bạn tự trích keyframe

```bash
python data/extract_keyframes.py L24 L25 L30      # ghi thẳng vào data/keyframe_cache
```

---

## 2. Cài đặt

Python 3.10+, ffmpeg.

```bash
conda create -n ai_env python=3.10 && conda activate ai_env
pip install -r requirements.txt
cd frontend && npm install && cd ..
```

Khoá Anthropic — **tự cung cấp**, lấy ở console.anthropic.com. Không có cũng chạy
(chế độ không-LLM: mất phân rã truy vấn và Hỏi–Đáp, phần còn lại nguyên vẹn):

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

---

## 3. Chạy

### A · Có GPU → chạy hết ở máy

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python -m uvicorn app:app --host 0.0.0.0 --port 8080   # tìm kiếm
python socket_app.py                                    # đáp án + ảnh + Q&A
cd frontend && npm run dev                              # http://localhost:3000
```

`frontend/.env.local`:

```
NEXT_PUBLIC_WEB_URL=http://localhost:8080
NEXT_PUBLIC_SOCKET_URL=http://localhost:8081
```

SigLIP tự tải ~3,5 GB lần đầu (CPU cũng chạy được, chậm hơn).

### B · Không GPU → tìm kiếm trên Kaggle

Trên Kaggle (bật GPU, thêm dataset `artifact-v4`), đặt **Add-ons → Secrets**:
`ANTHROPIC_API_KEY`, `NGROK_TOKEN`. Rồi:

```python
!git clone <repo> && cd AIC2025-main && python kaggle_run.py
```

Script kiểm hết trước khi mở cổng — thiếu gì nó nói ra thay vì im lặng chạy sai:
đủ artifacts, index object, LLM gọi được, SigLIP nạp xong. Chép URL ngrok in ra
vào `NEXT_PUBLIC_WEB_URL`.

Ở **máy bạn** vẫn phải chạy:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python socket_app.py
cd frontend && npm run dev
```

Ảnh và Hỏi–Đáp bắt buộc ở local: Kaggle không mount `keyframe_cache`/`videos`.

### Kiểm tra

```bash
curl localhost:8080/diagnostics   # kênh nào sống, index, LLM, cắt cụt token
curl localhost:8081/health        # ảnh, clip, LLM ở local
```

---

## 4. Module 


### Kênh bằng chứng

Bốn nguồn xếp hạng video độc lập, gộp bằng RRF.

| kênh | tìm trong | bật khi truy vấn có |
|---|---|---|
| **Hình ảnh** | SigLIP trên 154.640 keyframe | luôn bật (mặc định) |
| **Metadata** | tiêu đề / mô tả YouTube | tên chương trình, tên kênh |
| **Lời nói** | ASR, 859 video | tên riêng, con số **được đọc lên** |
| **Chữ trên hình** | OCR, 873 video | biển hiệu, dòng chữ, lower-third |

Mặc định "Tự" = **chỉ Hình ảnh**. Đo trên ground truth: mọi trọng số văn bản *tự
động* đều làm tệ đi — có mẫu SigLIP đặt đáp án ở hạng #1 toàn corpus rồi bị ba
kênh văn bản kéo xuống #115. Nhưng khi **bạn tự bật**, chúng có ích thật: một mẫu
lên từ hạng #10 → #4.

### Các module khác

| module | bật khi |
|---|---|
| **Dóng hàng thời gian** | chuỗi hành động **theo thứ tự**, từng cảnh rời đều tầm thường |
| **LLM** (mặc định tắt) | dịch máy sai thuật ngữ Việt (*"múa lân"* → *"the unicorn"*) |
| **Trong kết quả cũ** | thu hẹp dần: tìm A → lọc → tìm B, giữ frame có **cả hai** |
| **Bảng object** | vị trí trên khung là manh mối; 545 lớp, 2,05 triệu detection |
| **Chấm ảnh → tìm lại** | thấy đúng *kiểu* cảnh nhưng sai video; tìm bằng chính ảnh đã chấm |
| **TRAKE** | chuỗi khoảnh khắc, nộp cả dãy frame; gõ `(1) … (2) … (3) …` |
| **Hỏi VLM** | dạng Hỏi–Đáp, hoặc đáp án nằm trong **lời thoại** |

### Nộp bài

100 dòng, chấm theo thứ hạng — ô 1 đáng **1.00**, ô 51–100 chỉ **0.20**. Nên
"Lấp 100 dòng" giữ **nguyên** 20 ô đầu theo thứ hạng tìm kiếm, từ ô 21 mới xen
thêm frame nằm giữa các keyframe.

Hai nút dễ nhầm trên mỗi ảnh:

- **Thêm vào bài** → vào danh sách 100 dòng của câu hỏi đang chọn
- **Nộp ngay** → nộp **thẳng** lên server BTC, ngay lập tức

---

## 5. Đo chất lượng

```bash
python eval/run_gt.py       # KIS + Q&A trên ground truth thật
python eval/run_trake.py    # TRAKE
```

Hiện tại: Final Score **0.5636**, video đúng nằm trong top-30 ở **90,9%** truy vấn.

Kết quả **tất định** — bản dịch và bản phân rã LLM đều được đệm trên đĩa
(`data/translation_cache.json`, `data/decompose_cache.json`). Không có đệm thì hai
lần chạy cùng cấu hình ra hai điểm khác nhau, và không hiệu chỉnh được gì cả.

Mọi hằng số hiệu chỉnh được nằm ở [`config/fusion.json`](config/fusion.json).
