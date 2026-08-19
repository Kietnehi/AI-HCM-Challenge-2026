# Rules: Cấu trúc dữ liệu AI Challenge trên Google Drive

Tài liệu này mô tả cấu trúc Google Drive của dự án **AI Challenge (AIC 2026 – Batch 1)** để AI Agent
biết dữ liệu nằm ở đâu, đặt tên theo quy ước nào, và được phép làm gì.

> Cập nhật: 2026-08-08. Nguồn: quét trực tiếp Drive qua MCP connector `claude.ai Google Drive`,
> đối chiếu với đề bài chính thức.

---

## 0. Quy tắc bắt buộc (đọc trước khi thao tác)

0. **ĐỌC `task.txt` TRƯỚC TIÊN.** File `task.txt` ở thư mục gốc repo là **đề bài chính thức của
   cuộc thi** (Hội thi Thử thách Trí tuệ Nhân tạo TP.HCM 2026 – vòng sơ tuyển): 3 dạng truy vấn
   (Textual KIS, Q&A, TRAKE), định dạng nộp và cách chấm điểm. Mọi quyết định về output, chọn
   frame, xếp hạng kết quả đều phải bám theo file đó, không suy đoán.
1. **KHÔNG đổi tên, di chuyển, hay xoá bất kỳ file/thư mục nào trên Drive.** Chủ sở hữu đã ghi rõ
   yêu cầu này trong Google Doc ở thư mục gốc. Toàn bộ pipeline phụ thuộc vào tên file khớp chính xác.
2. Mặc định thao tác **chỉ đọc** (`search_files`, `read_file_content`, `get_file_metadata`).
   Mọi thao tác ghi (`create_file`, `copy_file`) phải được người dùng xác nhận trước.
3. **Không tải video/zip.** File video 100 MB – 13 GB, zip lên tới ~12.8 GB. Chỉ đọc metadata,
   hoặc đọc các file text nhỏ (CSV/JSON).
4. Drive là **thư mục được share** (owner gốc: `phatleh.224@gmail.com`), nhiều người cùng upload —
   luôn kiểm tra `modifiedTime` trước khi kết luận dữ liệu là mới nhất.
5. Khi cần liệt kê thư mục, dùng `parentId = '<id>'` thay vì `title contains` — nhanh và chính xác hơn.

---

## 1. Cây thư mục tổng quan

```
AI Challenge/                                   (root, id: 19Ij-K7r3tHoaW5vAKHkIw2ql6v_Tw3Ay)
├── Dataset_Directory/                          # DỮ LIỆU ĐÃ GIẢI NÉN — nguồn chính để đọc
│   ├── Videos_L<NN>_<x>/video/*.mp4
│   ├── Keyframes_L<NN>[_x]/keyframes/<VIDEO_ID>/*.jpg
│   ├── map-keyframes-aic25-b1/map-keyframes/<VIDEO_ID>.csv
│   ├── media-info-aic25-b1/media-info/<VIDEO_ID>.json
│   ├── objects-aic25-b1/objects/<VIDEO_ID>/*.json
│   └── clip-features-32-aic25-b1/clip-features-32/
├── Dataset  File Zip (Đã tải file zip 10-17, 23-33 Phát-Kiệt) 10-17, )/   # BẢN NÉN GỐC
│   └── *.zip
├── zip_vs_unzip_reports/                       # Báo cáo đối chiếu zip ↔ đã giải nén
│   └── *.csv
├── Kiệt/                                       # Notebook cá nhân
│   └── Check_Matching.ipynb
├── Code/                                       # (hiện trống)
└── Document                                    # Google Doc ghi chú/nội quy nhóm
```

### ID các thư mục hay dùng

| Thư mục | ID |
|---|---|
| AI Challenge (root) | `19Ij-K7r3tHoaW5vAKHkIw2ql6v_Tw3Ay` |
| Dataset_Directory | `17fAKQ6UZqVgU4PDsuyxvHnxW48kXbyLW` |
| Dataset File Zip | `10pumPBela25TNZNzJVJft3fYWTY5pIOy` |
| zip_vs_unzip_reports | `1atgiEYCVklx5G9Tjar_YctBbLLcZSOYf` |
| Kiệt | `1WFKVV4aomNexu4lqe7uW88R6dYXvJIor` |
| Code | `1yykuy7vZ8ZaPX5hJcU7hJrjh_CvczLOa` |
| map-keyframes-aic25-b1 | `16EzmSTfdKHp4dN6O25L9NltjSQQoM0CR` |
| media-info-aic25-b1 | `1gmNrxFkoQhZqJhanH4gP_I7NzsMy97xK` |
| objects-aic25-b1 | `1M1VRFY1_jSVNHIYq4zRsIVk5pbH2ucPh` |
| clip-features-32-aic25-b1 | `12s3SF9v8CmAR4eN28VLFp_GuLivuJKDT` |

---

## 2. Quy ước đặt tên

| Khái niệm | Dạng | Ví dụ |
|---|---|---|
| Batch / nhóm video | `L<NN>` với NN = 21…30 | `L28` |
| Phân mảnh của một batch | hậu tố `_a`, `_b`, … `_e` | `Videos_L26_c` |
| Video ID | `L<NN>_V<NNN>` | `L21_V003`, `L30_V096` |
| File video | `<VIDEO_ID>.mp4` | `L28_V006.mp4` |
| Keyframe | `<VIDEO_ID>/<NNN>.jpg`, số thứ tự 3 chữ số | `L21_V003/003.jpg` |
| Map keyframe | `<VIDEO_ID>.csv` | `L30_V096.csv` |
| Media info | `<VIDEO_ID>.json` | `L21_V002.json` |
| Object detection | `<VIDEO_ID>/<frame>.json` | `L21_V002/261.json` |

**Lưu ý bất đối xứng:** thư mục `Videos_*` luôn có hậu tố `_a..._e`, nhưng `Keyframes_*` thì không
đồng nhất — có `Keyframes_L21`, `Keyframes_L27`, `Keyframes_L30` (không hậu tố) lẫn
`Keyframes_L26_a`…`Keyframes_L26_e`. **Không suy ra tên thư mục — luôn liệt kê để xác nhận.**

---

## 3. Chi tiết từng loại dữ liệu

### 3.1 `Dataset_Directory/` — dữ liệu đã giải nén
Đây là nơi agent nên đọc mặc định.

- **Videos_L21_a, L22_a, L23_a, L24_a, L25_a, L26_a…e, L27_a, L28_a, L29_a, L30_a**
  → mỗi thư mục có thư mục con `video/` chứa `.mp4`.
- **Keyframes_L21…L30** (L26 chia thành `_a`…`_e`)
  → `keyframes/<VIDEO_ID>/<NNN>.jpg`. Ảnh JPEG ~150–250 KB/frame.
- **map-keyframes-aic25-b1** → `map-keyframes/<VIDEO_ID>.csv` (~1–2 KB). **Cực kỳ quan trọng:**
  tên file keyframe (`003.jpg`) chỉ là số thứ tự tăng dần, **KHÔNG phải frame index thật**.
  File CSV này là nơi tra ra `frame_id` thật để nộp bài. Không bao giờ nộp số thứ tự keyframe
  trực tiếp làm `frame_id`.
- **media-info-aic25-b1** → `media-info/<VIDEO_ID>.json` (~2–3 KB), metadata video lấy từ YouTube
  của kênh cung cấp dữ liệu. **Một số video có thể không có file metadata** — phải xử lý thiếu file.
- **objects-aic25-b1** → `objects/<VIDEO_ID>/<NNN>.json` (~10 KB). Kết quả Faster R-CNN pretrained
  trên OpenImages V4, theo định dạng object detection của TensorFlow. Tên file JSON khớp 1-1 với
  tên keyframe: `L01_V001/0000.jpg` ↔ `L01_V001/0000.json`.
- **clip-features-32-aic25-b1** → `clip-features-32/`. Feature trích từ **clip-ViT-B-32**. Toàn bộ
  CLIP features của keyframe trong một video được lưu trong **một file `.npy` duy nhất**, thứ tự
  vector tăng dần theo chỉ số keyframe (⇒ dòng thứ i của `.npy` ↔ keyframe thứ i ↔ dòng thứ i
  trong `map-keyframes`).

> **Dữ liệu thi chính thức là Video.** Keyframes / Objects / CLIP features / Metadata chỉ là dữ liệu
> hỗ trợ do BTC cung cấp thêm. Hiện Drive mới có **batch 1** (= batch 1 của AIC 2025); batch 2 sẽ
> được BTC thông báo sau.

### 3.2 Thư mục zip — bản nén gốc
Chứa `.zip` tương ứng 1-1 với các thư mục trong `Dataset_Directory`:
`Videos_L21_a.zip` … `Videos_L30_a.zip`, `Keyframes_L21.zip` … `Keyframes_L30.zip`,
`objects-aic25-b1.zip`, `map-keyframes-aic25-b1.zip`, `media-info-aic25-b1.zip`,
`clip-features-32-aic25-b1.zip`.

Kích thước: Keyframes ~0.5–6 GB, Videos ~2–13 GB. **Không tải, không giải nén tự động.**

### 3.3 `zip_vs_unzip_reports/` — nguồn kiểm tra tính toàn vẹn
Sinh ra bởi `Kiệt/Check_Matching.ipynb`. Khi cần trả lời "dữ liệu đã đủ chưa", đọc ở đây thay vì
đếm file thủ công:

- `summary_size_latest.csv` — bảng tổng hợp mới nhất (dùng file này).
- `summary_size_<YYYYMMDD_HHMMSS>.csv` — snapshot theo thời điểm.
- `<TÊN_THƯ_MỤC>_differences.csv` — khác biệt giữa zip và bản giải nén.
  File 36 byte ≈ chỉ có header ⇒ **không có sai lệch**.
- `folders_without_matching_zip.csv` — thư mục chưa có zip đối ứng.

---

## 4. Công thức truy vấn cho Agent

```
# Liệt kê nội dung một thư mục
parentId = '<FOLDER_ID>'

# Tìm mọi tài nguyên của một video
title contains 'L21_V003'

# Chỉ lấy metadata JSON
parentId = '1gdn34updwBHFFRdKDunywxB0S_fCqnR0' and mimeType = 'application/json'

# File thay đổi gần đây
modifiedTime > '2026-08-05T00:00:00Z'
```

Luôn `excludeContentSnippets: true` khi chỉ cần cấu trúc — tiết kiệm token đáng kể.
Dùng `nextPageToken` để phân trang; thư mục keyframes có thể chứa hàng nghìn ảnh.

---

## 5. Cách trả lời câu hỏi thường gặp

| Câu hỏi | Nguồn dữ liệu |
|---|---|
| "Đã có batch nào rồi?" | liệt kê `parentId = '17fAKQ6UZqVgU4PDsuyxvHnxW48kXbyLW'` |
| "Dữ liệu giải nén có đủ không?" | `zip_vs_unzip_reports/summary_size_latest.csv` |
| "Video X có bao nhiêu keyframe?" | đếm trong `Keyframes_L<NN>/keyframes/<VIDEO_ID>/` |
| "Metadata của video X?" | `media-info-aic25-b1/media-info/<VIDEO_ID>.json` |
| "Object trong frame N?" | `objects-aic25-b1/objects/<VIDEO_ID>/<N>.json` |

Nếu không tìm thấy dữ liệu: **báo rõ là chưa có trên Drive**, không suy đoán đường dẫn và không
tự tạo thư mục thay thế.
