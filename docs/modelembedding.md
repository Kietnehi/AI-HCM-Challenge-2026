Đúng gần đúng, nhưng chính xác là:

  - Jina-CLIP v2: embed ảnh keyframe và cũng embed text query để tìm ảnh bằng câu chữ.
  - BGE-M3: embed text như OCR, STT, caption, media-info; cũng embed text query để tìm các text đó.

  vision.faiss
    data:  ảnh keyframe → Jina encode_image
    query: câu hỏi      → Jina encode_text

  text.faiss
    data:  OCR/STT/caption/meta → BGE encode
    query: câu hỏi              → BGE encode

  Vì vậy query được encode hai lần, bằng hai model khác nhau:

  query
   ├─ Jina-CLIP → tìm frame ảnh giống mô tả
   └─ BGE-M3    → tìm transcript/OCR/caption/meta liên quan

  Sau đó gộp hai kết quả lại.
