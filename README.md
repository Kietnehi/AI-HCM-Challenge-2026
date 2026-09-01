# AI-HCM Challenge 2026

Kho mã nguồn, notebook và tài liệu phục vụ cuộc thi **AI Challenge 2026**. Dự án tập trung vào bài toán truy hồi video đa phương thức: tìm video và khung hình phù hợp với truy vấn, tinh chỉnh kết quả theo thời gian, hỗ trợ kiểm duyệt thủ công và xuất tệp submission theo đúng định dạng của cuộc thi.

> **Lưu ý:** Repository chỉ chứa mã nguồn, notebook, tài liệu và các tệp mẫu cần thiết. Dataset lớn, tệp liên kết nội bộ, prompt nội bộ và thông tin bí mật được loại khỏi GitHub theo `.gitignore`.

## Mục lục

- [Giới thiệu](#giới-thiệu)
- [Tính năng chính](#tính-năng-chính)
- [Cấu trúc repository](#cấu-trúc-repository)
- [Luồng xử lý tổng quát](#luồng-xử-lý-tổng-quát)
- [Yêu cầu môi trường](#yêu-cầu-môi-trường)
- [Bắt đầu nhanh](#bắt-đầu-nhanh)
- [Pipeline API](#pipeline-api)
- [Pipeline Local](#pipeline-local)
- [Cấu hình API và chi phí](#cấu-hình-api-và-chi-phí)
- [Định dạng submission](#định-dạng-submission)
- [Dữ liệu không được đưa lên GitHub](#dữ-liệu-không-được-đưa-lên-github)
- [Lưu ý bảo mật](#lưu-ý-bảo-mật)
- [Trạng thái dự án](#trạng-thái-dự-án)

## Giới thiệu

Hệ thống được tổ chức thành nhiều pipeline phục vụ các giai đoạn khác nhau của bài toán:

1. Trích xuất đặc trưng từ video và keyframe.
2. Chuẩn hóa dữ liệu thành các bản ghi có thể truy hồi.
3. Xây dựng nhiều loại chỉ mục cho tìm kiếm văn bản, hình ảnh và đối tượng.
4. Phân tích, dịch và mở rộng truy vấn.
5. Truy hồi ứng viên, hợp nhất kết quả, rerank và tinh chỉnh frame theo thời gian.
6. Kiểm tra kết quả, kiểm duyệt thủ công và tạo gói submission.

## Tính năng chính

- Truy hồi đa nhánh dựa trên đặc trưng hình ảnh, văn bản, OCR, ASR, caption, metadata và object detection.
- Kết hợp FAISS, BM25, chỉ mục đối tượng, weighted RRF, video prior và reranker.
- Tinh chỉnh frame bằng cách giải mã video gốc trong một cửa sổ thời gian hẹp.
- Hỗ trợ các dạng truy vấn và submission KIS, Q&A và TRAKE.
- Có checkpoint/resume để chạy lại từng giai đoạn hoặc từng truy vấn.
- Có validator giúp phát hiện lỗi định dạng trước khi nộp bài.
- Có chế độ `DRY_RUN` mặc định để ước tính token và chi phí trước khi gọi API thật.

## Cấu trúc repository

| Thư mục/tệp | Mô tả |
|---|---|
| `Code-Extract-Input/` | Notebook và mã nguồn trích xuất đặc trưng từ video, keyframe, OCR, ASR, caption, VLM và bản dịch. |
| `Code-ThuNghiem-AIC/Pipeline-API/` | Pipeline sử dụng mô hình/API bên ngoài để phân tích truy vấn, embedding, rerank và xuất submission. |
| `Code-ThuNghiem-AIC/Local/` | Pipeline local sử dụng các mô hình chạy trên GPU, kèm bước review và validator. |
| `Code-ThuNghiem-AIC/Pipeline-Cũ-4.8-Point/` | Phiên bản pipeline cũ dùng để tham khảo và đối chiếu. |
| `Planning/` | Kế hoạch, ghi chú và tài liệu phục vụ quá trình phát triển. |
| `TheLeCuocThi-DeThi/` | Thể lệ, quy định định dạng và các truy vấn của cuộc thi. |
| `submission_example_AIC26/` | Các tệp submission mẫu. |
| `Information.txt` | Ghi chú về dataset và các tài nguyên liên quan. |
| `.gitignore` | Danh sách dữ liệu lớn, tệp nội bộ, secret và tệp sinh ra không được commit. |

## Luồng xử lý tổng quát

```text
Video và metadata
        ↓
Trích xuất đặc trưng / keyframe / OCR / ASR / caption
        ↓
Xây dựng các chỉ mục FAISS, BM25 và object index
        ↓
Phân tích, dịch và mở rộng truy vấn
        ↓
Truy hồi ứng viên đa nhánh → hợp nhất → rerank
        ↓
Tinh chỉnh frame và kiểm duyệt kết quả
        ↓
Validator → submission.zip
```

## Yêu cầu môi trường

Khuyến nghị chạy các notebook trên **Kaggle Notebook** vì pipeline cần GPU, bộ nhớ lớn và một số mô hình có kích thước đáng kể.

### Thành phần cần chuẩn bị

- Tài khoản Kaggle có quyền truy cập các dataset cần thiết.
- GPU Kaggle cho các bước embedding, rerank hoặc xử lý video.
- Bật Internet khi notebook cần tải mô hình hoặc gọi OpenRouter API.
- Python/Jupyter Notebook theo môi trường Kaggle.
- Dataset video gốc, feature dataset và bộ đề được mount vào `/kaggle/input`.

Notebook có cơ chế tự dò thư mục dataset trong `/kaggle/input`. Khi cần, có thể ghi đè đường dẫn bằng các biến cấu hình như `feature_root`, `dataset_root`, `query_dir`, `FEATURE_ROOT`, `ART_INPUT` hoặc `PKG_INPUT` tùy pipeline.

## Bắt đầu nhanh

### 1. Lấy mã nguồn

```bash
git clone https://github.com/Kietnehi/AI-HCM-Challenge-2026.git
cd AI-HCM-Challenge-2026
```

### 2. Chọn pipeline

- Dùng **Pipeline API** nếu muốn tận dụng embedding, rerank và phân tích truy vấn qua API.
- Dùng **Pipeline Local** nếu muốn chạy nhiều mô hình trực tiếp trên GPU và kiểm duyệt kết quả trong notebook.
- Dùng `Code-Extract-Input/` nếu cần tạo hoặc cập nhật feature từ video đầu vào.

### 3. Chạy notebook theo thứ tự

Không nên chạy ngẫu nhiên các notebook trong cùng một pipeline. Mỗi giai đoạn tạo artifact làm đầu vào cho giai đoạn tiếp theo. Hãy đọc README và hướng dẫn Kaggle tương ứng trước khi chạy toàn bộ dataset; nên chạy smoke test với giới hạn nhỏ trước.

## Pipeline API

Hướng dẫn chi tiết nằm tại [`Code-ThuNghiem-AIC/Pipeline-API/README.md`](Code-ThuNghiem-AIC/Pipeline-API/README.md) và [`HUONG_DAN_CHAY_KAGGLE.md`](Code-ThuNghiem-AIC/Pipeline-API/HUONG_DAN_CHAY_KAGGLE.md).

Chạy các notebook chính theo thứ tự:

1. **`01-build-indices-api.ipynb`**
   - Dò và kiểm tra dữ liệu.
   - Tạo canonical records.
   - Xây dựng FAISS, BM25, object index và text-embedding index.
   - Sinh artifact tại `/kaggle/working/artifacts/` cùng `artifact_manifest.json`.

2. **`02-retrieve-refine-candidates-api.ipynb`**
   - Phân tích truy vấn và truy hồi đa nhánh.
   - Fusion, rerank, video prior, temporal NMS và frame refinement.
   - Tạo review package tại `/kaggle/working/review_package/`.
   - Cell xuất submission tạo `/kaggle/working/submission.zip`.

3. **`03-time-to-frameindex.ipynb`**
   - Xử lý và kiểm tra frame index theo pipeline hiện có trong repository.

> **Ghi chú:** Kế hoạch ban đầu có tham chiếu tới một notebook human-review/submit độc lập, nhưng notebook đó hiện không còn trong thư mục Pipeline API. Vì vậy, pipeline API hiện tại có bước xuất submission tích hợp trong notebook truy hồi để tạo gói kết quả cơ bản.

### Dataset tham khảo trên Kaggle

Pipeline API được thiết kế để sử dụng các dataset sau:

- `fatle542/AIC-Dataset`
- `kitnehi1211/feature-AIC-2026`
- `kitnehi1211/dethithunghiem`

Tên slug hoặc đường dẫn mount thực tế có thể thay đổi theo phiên bản dataset. Nếu notebook không tự dò được dữ liệu, hãy chỉ định lại đường dẫn trong `CFG`.

## Pipeline Local

Hướng dẫn chi tiết nằm tại [`Code-ThuNghiem-AIC/Local/README.md`](Code-ThuNghiem-AIC/Local/README.md) và [`KAGGLE_RUN_GUIDE.md`](Code-ThuNghiem-AIC/Local/KAGGLE_RUN_GUIDE.md).

Pipeline Local gồm ba notebook:

1. **`01_build_indices_local.ipynb`**: kiểm tra schema/coverage, tạo canonical records, FAISS cho SigLIP2/BGE-M3/CLIP, BM25 đa trường và object inverted index.
2. **`02_retrieve_refine_candidates_local.ipynb`**: dịch/mở rộng truy vấn, retrieval song ngữ, weighted RRF, rerank, frame refinement và tạo review package.
3. **`03_human_review_submit_local.ipynb`**: review bằng `ipywidgets`, lưu quyết định, chạy validator và tạo `submission.zip`.

### Một số artifact quan trọng

- NB01: `/kaggle/working/artifacts/` và `manifest.json`.
- NB02: `artifacts02_mimo/review_package/`, gồm `candidates.parquet`, `frame_catalog.csv`, `frames/`, `queries_parsed.json` và `sheets/`.
- NB03: thư mục submit chứa review decision, báo cáo validation và `submission.zip`.

### Ghi chú thiết kế

- Không trộn các embedding space: SigLIP2, BGE-M3 và CLIP dùng các FAISS index riêng.
- GPU được giải phóng sau mỗi stage để giảm nguy cơ hết VRAM.
- Checkpoint theo stage và theo query cho phép tiếp tục sau khi notebook bị gián đoạn.
- Thiếu một modality không làm loại bỏ ứng viên; ứng viên chỉ mất đóng góp từ nhánh tương ứng.

## Cấu hình API và chi phí

### API key

Chỉ cung cấp khóa thông qua biến môi trường `OPENROUTER_API_KEY` hoặc Kaggle Secret có cùng tên. Không ghi khóa trực tiếp vào notebook, README, log hoặc artifact.

Ví dụ tên biến cần tạo:

```text
OPENROUTER_API_KEY=<khóa-của-bạn>
```

Không đưa giá trị thật của khóa vào GitHub.

### Chế độ chạy an toàn

- `CFG["DRY_RUN"] = True` là mặc định trong các notebook API: chỉ ước tính token/chi phí và không gọi API thật.
- Đặt `DRY_RUN = False` chỉ khi đã kiểm tra dữ liệu, cấu hình và chi phí.
- Hard cap dùng chung là `MAX_TOTAL_COST_USD = 2.00`.
- Cache content-addressed cho embedding, rerank và LLM giúp hạn chế chi phí khi chạy lại.

### Mô hình chính của Pipeline API

- Visual: `google/siglip2-giant-opt-patch16-384`.
- Text embedding: `openai/text-embedding-3-small` qua OpenRouter.
- Text reranker: `voyageai/rerank-2.5-lite`.
- Phân tích, dịch và mở rộng truy vấn: `xiaomi/mimo-v2.5`.
- Lexical retrieval: BM25-Okapi trên `scipy.sparse`.

## Định dạng submission

Chi tiết định dạng phải được đối chiếu với [`TheLeCuocThi-DeThi/sotuyenAIC.md`](TheLeCuocThi-DeThi/sotuyenAIC.md). Các nguyên tắc kiểm tra chính gồm:

- Mỗi truy vấn có một tệp CSV tương ứng.
- Tệp dùng UTF-8, phân tách bằng dấu phẩy và không có header.
- Mỗi tệp có tối đa 100 dòng.
- `video_id` không có đuôi `.mp4`.
- `frame_id` phải là số nguyên hợp lệ.
- Q&A phải đúng số cột và câu trả lời không quá 100 ký tự.
- TRAKE phải có đúng số frame yêu cầu và frame được sắp xếp tăng dần.
- File ZIP phải chứa thư mục `submission/`.

Với Q&A cần người kiểm duyệt điền câu trả lời, pipeline có thể sử dụng tệp JSON dạng:

```json
{
  "query-p1-15-qa": "12"
}
```

Sau khi review, hãy chạy lại bước export/validator để tạo ZIP cuối cùng. Không nên nộp ZIP khi còn lỗi P0 trong báo cáo validation.

## Dữ liệu không được đưa lên GitHub

Các mục sau được loại khỏi repository bằng `.gitignore`:

- `Feature_Dataset/`
- `Feature_Dataset.zip`
- `Kiet-Prompt/`
- `Link.txt`
- `THUNGHIEM-bo-de-thi/` và các bộ dữ liệu thử nghiệm tương tự
- `.env` và các tệp chứa secret
- Các thư mục cache, virtual environment, artifact và output tạm thời

Các tệp bị ignore vẫn có thể tồn tại trên máy local hoặc trong môi trường Kaggle. Nếu clone repository mới, cần chuẩn bị lại các dữ liệu này theo quyền truy cập hợp lệ của bạn.

## Lưu ý bảo mật

- Không commit API key, mật khẩu, token, cookie hoặc đường dẫn chứa thông tin riêng tư.
- Kiểm tra `git status` và `git check-ignore` trước khi commit dữ liệu mới.
- Nếu một khóa đã từng xuất hiện trong lịch sử Git, hãy thu hồi khóa đó trên nhà cung cấp và tạo khóa mới; chỉ xóa chuỗi khóa khỏi file hiện tại là chưa đủ.
- Không tải dataset hoặc tài liệu nội bộ lên repository công khai nếu chưa có quyền phân phối.

## Trạng thái dự án

Repository đang phục vụ mục đích nghiên cứu, thử nghiệm và chuẩn bị submission cho AI Challenge 2026. Khi thay đổi cấu hình hoặc notebook, nên ghi lại phiên bản mô hình, dataset, tham số và kết quả validation để có thể tái lập thí nghiệm.
