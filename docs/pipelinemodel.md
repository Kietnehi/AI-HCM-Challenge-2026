 ## Vai trò từng model

   Model           Dùng trên dữ liệu nào                     Ai chạy                           Chạy lúc nào
  ━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Jina-CLIP v2    Ảnh keyframe và query                     Bạn của bạn + pipeline của bạn    Index ảnh trước; encode query khi search
  ──────────────  ────────────────────────────────────────  ────────────────────────────────  ───────────────────────────────────────────
   BGE-M3          OCR, STT, caption, media-info và query    Bạn của bạn + pipeline của bạn    Index text trước; encode query khi search
  ──────────────  ────────────────────────────────────────  ────────────────────────────────  ───────────────────────────────────────────
   BGE reranker    Query + top text candidate                Pipeline của bạn                  Chỉ lúc search
  ──────────────  ────────────────────────────────────────  ────────────────────────────────  ───────────────────────────────────────────
   Qwen VL         Query + top frame/video                   Pipeline của bạn                  Chỉ Q&A/TRAKE hoặc rerank cuối
