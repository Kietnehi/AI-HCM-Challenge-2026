# AI-HCM-Challenge-2026

Ma nguon va tai lieu phuc vu AI Challenge 2026.

## Noi dung

- `Code-Extract-Input/`: notebook va ma nguon trich xuat dac trung tu video.
- `Code-ThuNghiem-AIC/`: pipeline local va pipeline API cho retrieval, refinement va tao submission.
- `TheLeCuocThi-DeThi/`: the le, dinh dang bai thi va cac truy van mau.
- `THUNGHIEM-bo-de-thi/`: bo de dung cho thu nghiem.
- `submission_example_AIC26/`: cac file submission mau.
- `Information.txt`: ghi chu ve du lieu va tai nguyen lien quan.

## Pipeline API

Huong dan chay pipeline nam trong [`Code-ThuNghiem-AIC/Pipeline-API/README.md`](Code-ThuNghiem-AIC/Pipeline-API/README.md).
Thu tu notebook chinh:

1. `01-build-indices-api.ipynb` - xay dung index.
2. `02-retrieve-refine-candidates-api.ipynb` - truy hoi, xep hang va tao goi submission.
3. `03-time-to-frameindex.ipynb` - xu ly frame index va kiem tra ket qua.

## Luu y

Dataset lon, file lien ket noi bo, prompt noi bo va thong tin bi mat khong duoc dua vao repository. Hay xem `.gitignore` truoc khi them du lieu moi.

Khong commit API key hoac file `.env` vao GitHub.
