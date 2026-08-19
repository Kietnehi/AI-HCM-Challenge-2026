"""Dịch transcript tiếng Việt (PhoWhisper) sang tiếng Anh cho text retrieval.

Chiến lược: xếp tầng (cascade) nhiều backend, ưu tiên thứ miễn phí trước, thứ
tính tiền chỉ gánh phần các backend trước không dịch nổi. Nhờ vậy chi phí thực
tế gần bằng 0 mà vẫn không bỏ sót segment nào.

Thứ tự mặc định (đổi bằng --chain):

  1. gemini          Gemini API, model Flash đang free of charge. Chất lượng tốt
                     nhất: khôi phục dấu câu, đổi số viết chữ thành chữ số, sửa
                     lỗi ASR theo ngữ cảnh. Dùng nhiều key trong .env, xoay vòng
                     khi bị rate-limit. Hết quota thì nhường tầng dưới.
  2. googletrans     Google Translate qua googletrans. Free, không key.
  3. deep-translator Cũng Google Translate nhưng endpoint khác, làm tầng dự
                     phòng khi googletrans bị chặn.
  4. openrouter      DeepSeek qua OpenRouter. TỐN TIỀN nên xếp cuối, chỉ chạm
                     tới khi cả ba tầng trên đều thất bại. Có --max-cost để
                     chặn cháy credit.

Mỗi tầng đều gộp nhiều segment vào một request rồi tách lại; nếu số phần tử trả
về không khớp thì nhóm tự chia đôi và thử lại. Timestamp giữ nguyên. Kết quả ghi
ra cây thư mục song song với input nên dừng giữa chừng rồi chạy lại là tiếp tục.

Ví dụ:
    python Code/Translation/translate_transcripts.py --limit 2      # chạy thử
    python Code/Translation/translate_transcripts.py                # toàn bộ
    python Code/Translation/translate_transcripts.py --chain gemini,googletrans
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

VERBOSE = False


def _trace(message: str) -> None:
    """In lý do một tầng bị bỏ qua. Tắt mặc định vì cascade tự xử lý được."""
    if VERBOSE:
        print(f"    · {message}", file=sys.stderr, flush=True)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SRC = REPO_ROOT / "Feature_Dataset" / "Transcript_Extract"
DEFAULT_OUT = REPO_ROOT / "Feature_Dataset" / "Transcript_Translated"

# Giới hạn dùng chung cho mọi tầng: một nhóm phải chạy được trên cả Google
# (chặn ~5000 ký tự) lẫn LLM (batch quá lớn dễ trả thiếu phần tử).
MAX_ITEMS = 25
MAX_CHARS = 4000

SYSTEM_PROMPT = """Bạn dịch phụ đề tin tức tiếng Việt sang tiếng Anh cho một hệ thống tìm kiếm video.

Input là các đoạn nhận dạng giọng nói: KHÔNG có dấu câu, toàn chữ thường, số viết bằng chữ, đôi khi sai chính tả do nhận dạng.

Yêu cầu:
- Dịch sang tiếng Anh tự nhiên, có dấu câu và viết hoa đúng chuẩn.
- Đổi số, ngày tháng, đơn vị viết bằng chữ thành chữ số: "năm phẩy bảy xăngtimét" -> "5.7 cm"; "ba mươi mốt tháng bảy" -> "July 31"; "hai mươi lăm điểm phần trăm" -> "25 basis points".
- Sửa lỗi nhận dạng hiển nhiên dựa vào ngữ cảnh: "đồng băng sông cửu long" -> "Mekong Delta".
- Giữ đúng tên người, địa danh, tổ chức; dùng tên tiếng Anh phổ biến nếu có.
- KHÔNG thêm, bớt, tóm tắt hay bình luận. Chỉ dịch.
- Trả về đúng một phần tử cho mỗi đoạn input, giữ nguyên id và thứ tự.
- Nếu một đoạn tối nghĩa, vẫn dịch sát nhất có thể; tuyệt đối không để trống."""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "translations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "en": {"type": "string"},
                },
                "required": ["id", "en"],
            },
        }
    },
    "required": ["translations"],
}


# ---------------------------------------------------------------- tiện ích


def iter_transcripts(src: Path) -> list[Path]:
    return sorted(path for path in src.rglob("*.json") if path.is_file())


def is_complete(out_path: Path, expected_segments: int) -> bool:
    """Output đã dịch xong và khớp số segment với input?"""
    if not out_path.is_file():
        return False
    try:
        data = json.loads(out_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    segments = data.get("segments")
    if not isinstance(segments, list) or len(segments) != expected_segments:
        return False
    # PhoWhisper sinh segment rỗng ở đoạn không có tiếng nói; những segment đó
    # hợp lệ khi text_en cũng rỗng, nếu không video sẽ bị dịch lại mãi.
    return all(
        str(segment.get("text_en", "")).strip()
        or not str(segment.get("text", "")).strip()
        for segment in segments
    )


def write_atomic(path: Path, payload: dict) -> None:
    """Ghi qua file tạm để lần chạy bị ngắt không để lại JSON hỏng."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


def split_oversized(text: str, limit: int) -> list[str]:
    """Cắt segment dài quá giới hạn thành nhiều mảnh, cắt ở ranh giới từ."""
    parts: list[str] = []
    current: list[str] = []
    size = 0
    for word in text.split():
        if current and size + len(word) + 1 > limit:
            parts.append(" ".join(current))
            current, size = [], 0
        current.append(word)
        size += len(word) + 1
    if current:
        parts.append(" ".join(current))
    return parts or [""]


def group_pieces(texts: list[str]) -> list[list[int]]:
    """Gom index các mảnh thành nhóm theo cả số lượng lẫn tổng ký tự."""
    groups: list[list[int]] = []
    current: list[int] = []
    size = 0
    for index, text in enumerate(texts):
        cost = len(text) + 1
        if current and (len(current) >= MAX_ITEMS or size + cost > MAX_CHARS):
            groups.append(current)
            current, size = [], 0
        current.append(index)
        size += cost
    if current:
        groups.append(current)
    return groups


# ---------------------------------------------------------------- các tầng
#
# Mọi tầng có chung giao diện try_group(texts) -> list[str] | None.
# None nghĩa là "tầng này không dịch được, mời tầng sau" chứ không phải thất bại
# vĩnh viễn, nhờ vậy cascade tự trượt xuống mà không mất segment.


class GeminiTier:
    """Gemini API, nhiều key xoay vòng. Các model Flash hiện free of charge."""

    name = "gemini"
    paid = False

    def __init__(self, model: str, api_keys: list[str], min_interval: float,
                 max_wait: float, quota_cooldown: float = 60.0) -> None:
        from google import genai

        self.model = model
        self.max_wait = max_wait
        self.quota_cooldown = quota_cooldown
        self._clients = [genai.Client(api_key=key) for key in api_keys]
        # Free tier giới hạn theo từng key nên mỗi key có nhịp riêng.
        self._ready_at = [0.0] * len(self._clients)
        self._min_interval = min_interval
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        if not self._clients:
            return False
        # Mọi key đều đang phải chờ quá lâu -> coi như hết quota, nhường tầng sau
        # thay vì ngồi ngủ trong khi có backend miễn phí khác đang rảnh.
        with self._lock:
            soonest = min(self._ready_at) - time.monotonic()
        return soonest <= self.max_wait

    def _acquire(self) -> int | None:
        with self._lock:
            index = min(range(len(self._clients)), key=lambda i: self._ready_at[i])
            wait = self._ready_at[index] - time.monotonic()
            if wait > self.max_wait:
                return None
            self._ready_at[index] = (
                max(self._ready_at[index], time.monotonic()) + self._min_interval
            )
        if wait > 0:
            time.sleep(wait)
        return index

    def _penalize(self, index: int, seconds: float) -> None:
        with self._lock:
            self._ready_at[index] = max(
                self._ready_at[index], time.monotonic() + seconds
            )

    def try_group(self, texts: list[str]) -> list[str] | None:
        from google.genai import types

        if not self._clients:
            return None
        payload = json.dumps(
            [{"id": i, "vi": text} for i, text in enumerate(texts)], ensure_ascii=False
        )
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0,
            response_mime_type="application/json",
            response_schema=RESPONSE_SCHEMA,
            max_output_tokens=16384,
            # Dịch thuần, không cần suy luận; bật thinking sẽ đốt hết output
            # budget trước khi kịp sinh JSON.
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )

        for _ in range(len(self._clients)):
            index = self._acquire()
            if index is None:
                return None
            try:
                response = self._clients[index].models.generate_content(
                    model=self.model, contents=payload, config=config
                )
            except Exception as error:
                message = str(error)
                if any(m in message for m in ("RESOURCE_EXHAUSTED", "429")):
                    self._penalize(index, self.quota_cooldown)
                    _trace(f"gemini key #{index} hết quota, nghỉ {self.quota_cooldown:.0f}s")
                    continue
                if any(m in message for m in ("API_KEY_INVALID", "not valid", "PERMISSION")):
                    self._penalize(index, 10**6)
                    print(f"  ! gemini key #{index} hỏng: {message[:100]}",
                          file=sys.stderr)
                    continue
                if any(m in message for m in ("UNAVAILABLE", "503", "500")):
                    self._penalize(index, 15.0)
                    _trace(f"gemini key #{index} tạm lỗi server")
                    continue
                _trace(f"gemini lỗi khác: {message[:120]}")
                return None

            result = _parse_llm_json(response.text, response.parsed, len(texts))
            if result is not None:
                return result
            _trace("gemini trả JSON không khớp số phần tử")
        return None


class GoogletransTier:
    """googletrans - Google Translate free, không cần key. API của bản 3.4 là async."""

    name = "googletrans"
    paid = False

    def __init__(self, cooldown: float = 90.0) -> None:
        from googletrans import Translator

        self._make = Translator
        self._blocked_until = 0.0
        self._cooldown = cooldown

    @property
    def available(self) -> bool:
        return time.monotonic() >= self._blocked_until

    async def _translate(self, payload: str) -> str | None:
        async with self._make() as translator:
            result = await translator.translate(payload, src="vi", dest="en")
        return getattr(result, "text", None)

    def try_group(self, texts: list[str]) -> list[str] | None:
        if not self.available:
            return None
        try:
            text = asyncio.run(self._translate("\n".join(texts)))
        except Exception as error:
            # Bị chặn hoặc endpoint đổi -> nghỉ một lúc rồi thử lại sau.
            self._blocked_until = time.monotonic() + self._cooldown
            _trace(f"googletrans lỗi ({str(error)[:100]}), nghỉ {self._cooldown:.0f}s")
            return None
        return _split_lines(text, len(texts))


class DeepTranslatorTier:
    """deep-translator - cũng Google Translate nhưng đường khác, làm dự phòng."""

    name = "deep-translator"
    paid = False

    def __init__(self, delay: float = 0.2, cooldown: float = 90.0) -> None:
        from deep_translator import GoogleTranslator

        self._make = lambda: GoogleTranslator(source="vi", target="en")
        self.delay = delay
        self._blocked_until = 0.0
        self._cooldown = cooldown

    @property
    def available(self) -> bool:
        return time.monotonic() >= self._blocked_until

    def try_group(self, texts: list[str]) -> list[str] | None:
        if not self.available:
            return None
        try:
            text = self._make().translate("\n".join(texts))
            time.sleep(self.delay)
        except Exception as error:
            self._blocked_until = time.monotonic() + self._cooldown
            _trace(f"deep-translator lỗi ({str(error)[:100]}), nghỉ {self._cooldown:.0f}s")
            return None
        return _split_lines(text, len(texts))


class OpenRouterTier:
    """DeepSeek qua OpenRouter. Tầng duy nhất tốn tiền nên luôn xếp cuối."""

    name = "openrouter"
    paid = True

    def __init__(self, model: str, api_keys: list[str], max_cost: float) -> None:
        import requests

        self._requests = requests
        self.model = model
        self._keys = api_keys
        self.max_cost = max_cost
        self.cost = 0.0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        return bool(self._keys) and self.cost < self.max_cost

    def try_group(self, texts: list[str]) -> list[str] | None:
        result = self._attempt(texts)
        if result is not None or len(texts) == 1:
            return result
        # Câu trả lời hỏng thường do output bị cắt; chia đôi rồi thử lại ngay
        # trong tầng này thay vì đẩy xuống tầng free, vì tiền đã trả rồi.
        left = self.try_group(texts[: len(texts) // 2])
        if left is None:
            return None
        right = self.try_group(texts[len(texts) // 2 :])
        return None if right is None else left + right

    def _attempt(self, texts: list[str]) -> list[str] | None:
        if not self.available:
            return None
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(
                        [{"id": i, "vi": t} for i, t in enumerate(texts)],
                        ensure_ascii=False,
                    ),
                },
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
            # deepseek-v4-flash là reasoning model: để mặc định thì nó đốt sạch
            # token budget vào suy luận nội bộ và trả content rỗng. Dịch thuần
            # không cần suy luận nên tắt hẳn.
            "reasoning": {"enabled": False},
            # Không đặt trần thì output dài bị cắt giữa chừng, JSON hỏng mà tiền
            # vẫn bị tính. Bản dịch tiếng Anh dài cỡ bản gốc nên ước theo đó.
            "max_tokens": min(32000, 400 + sum(len(t) for t in texts)),
            # Xin OpenRouter trả chi phí thật để cộng dồn thay vì ước lượng.
            "usage": {"include": True},
        }
        for key in self._keys:
            try:
                response = self._requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {key}",
                        "Content-Type": "application/json",
                        "X-Title": "AIC 2026 Transcript Translation",
                    },
                    json=payload,
                    timeout=120,
                )
            except self._requests.RequestException as error:
                _trace(f"openrouter không kết nối được: {str(error)[:100]}")
                continue
            if response.status_code in (401, 402, 429):
                _trace(f"openrouter từ chối: HTTP {response.status_code}")
                continue
            if not response.ok:
                _trace(f"openrouter lỗi HTTP {response.status_code}")
                continue
            try:
                body = response.json()
                choice = body["choices"][0]
                content = choice["message"]["content"]
            except (KeyError, IndexError, ValueError):
                _trace("openrouter trả body không đọc được")
                continue

            usage = body.get("usage") or {}
            with self._lock:
                self.cost += float(usage.get("cost") or 0.0)
                self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
                self.completion_tokens += int(usage.get("completion_tokens") or 0)

            result = _parse_llm_json(content, None, len(texts))
            if result is not None:
                return result
            _trace(
                f"openrouter JSON không dùng được (finish={choice.get('finish_reason')}, "
                f"{len(texts)} đoạn)"
            )
        return None


def _split_lines(text: str | None, expected: int) -> list[str] | None:
    """Tách kết quả Google theo dòng, chỉ nhận khi đúng số dòng mong đợi."""
    if not isinstance(text, str) or not text.strip():
        return None
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if len(lines) != expected:
        return None
    return lines


def _parse_llm_json(raw, parsed, expected: int) -> list[str] | None:
    """Đọc JSON {translations:[{id,en}]} và sắp lại đúng thứ tự id.

    Chấp nhận cả hai biến thể model hay trả ngoài schema: mảng trần [{id,en}],
    và object đơn {id,en} khi nhóm chỉ có một phần tử.
    """
    if parsed is None:
        try:
            parsed = json.loads(raw or "")
        except (json.JSONDecodeError, TypeError):
            return None
    if isinstance(parsed, list):
        items = parsed
    elif isinstance(parsed, dict) and "translations" in parsed:
        items = parsed.get("translations")
    elif isinstance(parsed, dict) and "en" in parsed:
        items = [parsed]
    else:
        return None
    if not isinstance(items, list) or len(items) != expected:
        return None
    result = [""] * expected
    for item in items:
        if not isinstance(item, dict):
            return None
        position = item.get("id")
        if not isinstance(position, int) or not 0 <= position < expected:
            return None
        result[position] = " ".join(str(item.get("en", "")).split())
    return result if all(result) else None


MIN_RATIO = 0.55
RATIO_MIN_LEN = 80
MAX_TEXT_SPLIT = 3


def _looks_truncated(vi: str, en: str) -> bool:
    """Bản dịch ngắn bất thường so với bản gốc -> Google đã nuốt mất nội dung.

    Google Translate im lặng bỏ bớt phần sau của những đoạn ASR dài không dấu
    câu: trả về câu đầu rồi nhảy thẳng tới câu cuối, không báo lỗi gì. Số dòng
    vẫn khớp nên kiểm tra theo dòng không phát hiện được, phải so độ dài.

    Tiếng Anh dịch từ tiếng Việt thường dài xấp xỉ hoặc hơn bản gốc (trung vị
    đo được là 1.04). Dưới 0.55 gần như chắc chắn là mất chữ chứ không phải
    hành văn gọn. Chỉ soi đoạn đủ dài vì đoạn ngắn có tỉ lệ dao động rất mạnh.
    """
    return len(vi) >= RATIO_MIN_LEN and len(en) < MIN_RATIO * len(vi)


class Cascade:
    """Thử lần lượt từng tầng; nhóm nào cả dàn đều chịu thì chia đôi thử lại."""

    def __init__(self, tiers: list) -> None:
        self.tiers = tiers
        self.stats: dict[str, int] = {tier.name: 0 for tier in tiers}
        self.failed = 0
        self.repaired = 0
        self.truncated = 0

    def _try_tiers(self, texts: list[str]) -> tuple[list[str] | None, str | None, bool]:
        """Trả (kết quả, tên tầng, có sạch không).

        Kết quả bị nghi cắt cụt vẫn được giữ lại làm phương án chót — bản dịch
        thiếu vẫn hơn ô trống khi mọi cách chữa đều thất bại. Giữ bản dài nhất
        trong số các bản cụt.
        """
        best: tuple[list[str], str] | None = None
        for tier in self.tiers:
            if not tier.available:
                continue
            result = tier.try_group(texts)
            if result is None:
                continue
            if not any(
                _looks_truncated(vi, en) for vi, en in zip(texts, result)
            ):
                return result, tier.name, True
            _trace(f"{tier.name} trả bản dịch ngắn bất thường, nghi bị cắt")
            if best is None or sum(map(len, result)) > sum(map(len, best[0])):
                best = (result, tier.name)
        return (best[0], best[1], False) if best else (None, None, False)

    def translate_group(self, texts: list[str], depth: int = 0) -> list[str]:
        if not texts:
            return []
        result, tier_name, clean = self._try_tiers(texts)
        if clean:
            self.stats[tier_name] += len(texts)
            return result

        # Nhiều đoạn: chia đôi nhóm, đoạn hỏng sẽ bị cô lập dần.
        if len(texts) > 1:
            middle = len(texts) // 2
            return self.translate_group(texts[:middle], depth) + self.translate_group(
                texts[middle:], depth
            )

        # Còn đúng một đoạn mà vẫn cụt: chia chính văn bản đó. Cắt ngắn lại thì
        # Google bám được ngữ cảnh và dịch trọn, đo thực tế cứu được 12/12 mẫu.
        text = texts[0]
        cut = text.rfind(" ", 0, len(text) // 2)
        if depth < MAX_TEXT_SPLIT and cut > 0:
            left = self.translate_group([text[:cut]], depth + 1)
            right = self.translate_group([text[cut + 1 :]], depth + 1)
            joined = " ".join(part for part in (left + right) if part).strip()
            if joined and (result is None or len(joined) > len(result[0])):
                if depth == 0:
                    self.repaired += 1
                return [joined]

        if result is not None:
            self.truncated += 1
            self.stats[tier_name] += 1
            return result
        self.failed += 1
        return [""]


# ---------------------------------------------------------------- xử lý


def translate_video(cascade: Cascade, data: dict) -> dict:
    """Dịch toàn bộ segment của một video, giữ nguyên timestamp."""
    segments = data.get("segments") or []

    # Phẳng hoá: mỗi segment thành 0+ mảnh. Segment rỗng (đoạn không có tiếng
    # nói) không sinh mảnh nào nên không tốn request.
    pieces: list[str] = []
    spans: list[tuple[int, int]] = []
    for segment in segments:
        text = " ".join(str(segment.get("text", "")).split())
        if not text:
            parts: list[str] = []
        elif len(text) > MAX_CHARS:
            parts = split_oversized(text, MAX_CHARS)
        else:
            parts = [text]
        spans.append((len(pieces), len(pieces) + len(parts)))
        pieces.extend(parts)

    # Chạy tuần tự: các tầng free đều bị rate-limit theo IP/key nên bắn song
    # song chỉ làm chúng bị chặn sớm hơn.
    translated: list[str] = [""] * len(pieces)
    for group in group_pieces(pieces):
        values = cascade.translate_group([pieces[i] for i in group])
        for index, value in zip(group, values):
            translated[index] = value

    out_segments = []
    for segment, (start, end) in zip(segments, spans):
        text_en = " ".join(part for part in translated[start:end] if part).strip()
        out_segments.append(
            {
                "id": segment.get("id"),
                "start": segment.get("start"),
                "end": segment.get("end"),
                "video_start": segment.get("video_start"),
                "video_end": segment.get("video_end"),
                "text": segment.get("text"),
                "text_en": text_en,
            }
        )

    return {
        "video_id": data.get("video_id"),
        "source": data.get("source"),
        "asr_model": data.get("model"),
        "translator": "+".join(tier.name for tier in cascade.tiers),
        "language": "vi",
        "target_language": "en",
        "text": data.get("text"),
        "text_en": " ".join(s["text_en"] for s in out_segments if s["text_en"]),
        "segments": out_segments,
    }


def _norm(text: str) -> str:
    return " ".join(str(text).lower().split())


def find_untranslated(out_dir: Path) -> list[tuple[Path, dict, list[int]]]:
    """Tìm segment mà bản dịch trùng khít bản gốc, tức là chưa dịch thật.

    Google Translate đôi khi trả nguyên văn với đoạn ASR khó, và những file đó
    vẫn qua được is_complete nên resume thường không đụng tới.
    """
    jobs: list[tuple[Path, dict, list[int]]] = []
    for path in sorted(out_dir.rglob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        indices = [
            index
            for index, segment in enumerate(data.get("segments") or [])
            if str(segment.get("text", "")).strip()
            and _norm(segment.get("text", "")) == _norm(segment.get("text_en", ""))
        ]
        if indices:
            jobs.append((path, data, indices))
    return jobs


def run_fix(cascade: Cascade, out_dir: Path) -> int:
    """Dịch lại riêng những segment còn nguyên tiếng Việt rồi ghi đè tại chỗ."""
    jobs = find_untranslated(out_dir)
    total = sum(len(indices) for _, _, indices in jobs)
    print(f"Tìm thấy {total} segment chưa dịch trong {len(jobs)} video.")
    if not jobs:
        return 0

    repaired = stubborn = 0
    for position, (path, data, indices) in enumerate(jobs, start=1):
        segments = data["segments"]
        pieces = [" ".join(str(segments[i]["text"]).split()) for i in indices]
        results: list[str] = [""] * len(pieces)
        for group in group_pieces(pieces):
            values = cascade.translate_group([pieces[i] for i in group])
            for slot, value in zip(group, values):
                results[slot] = value

        changed = False
        for slot, index in enumerate(indices):
            new = results[slot].strip()
            if new and _norm(new) != _norm(segments[index]["text"]):
                segments[index]["text_en"] = new
                repaired += 1
                changed = True
            else:
                stubborn += 1
        if changed:
            data["text_en"] = " ".join(
                s["text_en"] for s in segments if str(s.get("text_en", "")).strip()
            )
            write_atomic(path, data)
        print(
            f"[{position}/{len(jobs)}] {data.get('video_id')} "
            f"{len(indices)} segment | đã vá {repaired} | vẫn nguyên {stubborn}",
            flush=True,
        )

    print(f"\nVá được {repaired}/{total} segment. Còn {stubborn} segment không dịch nổi.")
    return 0


def find_truncated(out_dir: Path) -> list[tuple[Path, dict, list[int]]]:
    """Tìm segment mà bản dịch ngắn bất thường, tức là bị Google cắt mất đuôi."""
    jobs: list[tuple[Path, dict, list[int]]] = []
    for path in sorted(out_dir.rglob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        indices = [
            index
            for index, segment in enumerate(data.get("segments") or [])
            if str(segment.get("text_en", "")).strip()
            and _looks_truncated(
                " ".join(str(segment.get("text", "")).split()),
                " ".join(str(segment.get("text_en", "")).split()),
            )
        ]
        if indices:
            jobs.append((path, data, indices))
    return jobs


def run_repair(cascade: Cascade, out_dir: Path) -> int:
    """Dịch lại các segment bị cắt cụt bằng cách chia nhỏ, rồi ghi đè tại chỗ.

    Chỉ nhận bản mới khi nó dài hơn bản cũ, nên lần chạy này không bao giờ làm
    dữ liệu tệ đi so với trước.
    """
    jobs = find_truncated(out_dir)
    total = sum(len(indices) for _, _, indices in jobs)
    print(f"Tìm thấy {total} segment nghi bị cắt cụt trong {len(jobs)} video.")
    if not jobs:
        return 0

    better = unchanged = 0
    for position, (path, data, indices) in enumerate(jobs, start=1):
        segments = data["segments"]
        changed = False
        for index in indices:
            source = " ".join(str(segments[index]["text"]).split())
            old = str(segments[index]["text_en"]).strip()
            new = cascade.translate_group([source])[0].strip()
            if len(new) > len(old):
                segments[index]["text_en"] = new
                better += 1
                changed = True
            else:
                unchanged += 1
        if changed:
            data["text_en"] = " ".join(
                s["text_en"] for s in segments if str(s.get("text_en", "")).strip()
            )
            write_atomic(path, data)
        print(
            f"[{position}/{len(jobs)}] {data.get('video_id')} "
            f"{len(indices)} segment | tốt hơn {better} | giữ nguyên {unchanged}",
            flush=True,
        )

    print(f"\nCải thiện {better}/{total} segment. {unchanged} segment không dài thêm.")
    return 0


def load_keys(name: str) -> list[str]:
    """Đọc key từ .env gốc; hỗ trợ JSON list hoặc chuỗi phân tách bằng dấu phẩy."""
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
    raw = os.getenv(name, "").strip()
    if not raw:
        return []
    if raw.startswith("["):
        try:
            values = json.loads(raw)
        except json.JSONDecodeError:
            return []
        return [str(value).strip() for value in values if str(value).strip()]
    return [value.strip() for value in raw.split(",") if value.strip()]


TIER_NAMES = ("gemini", "googletrans", "deep-translator", "openrouter")
FREE_NAMES = ("gemini", "googletrans", "deep-translator")


def parse_assign(spec: str) -> list[tuple[str, float]]:
    """Đọc 'gemini=35,googletrans=25,...' thành [(tên, trọng số)]."""
    pairs: list[tuple[str, float]] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        name, _, weight = part.partition("=")
        name = name.strip()
        if name not in TIER_NAMES:
            raise SystemExit(f"Nguồn dịch không hợp lệ: {name}")
        try:
            value = float(weight) if weight.strip() else 1.0
        except ValueError:
            raise SystemExit(f"Trọng số không hợp lệ cho {name}: {weight!r}")
        if value > 0:
            pairs.append((name, value))
    if not pairs:
        raise SystemExit("--assign không có nguồn nào hợp lệ.")
    return pairs


def make_tier(name: str, args):
    """Tạo một tầng; trả None nếu thiếu key."""
    if name == "gemini":
        keys = load_keys("GEMINI_API_KEY")
        if not keys:
            print("  ! bỏ gemini: thiếu GEMINI_API_KEY", file=sys.stderr)
            return None
        print(f"  gemini          : {args.gemini_model} | {len(keys)} key | free")
        return GeminiTier(args.gemini_model, keys, args.min_interval, args.max_wait)
    if name == "googletrans":
        print("  googletrans     : free")
        return GoogletransTier()
    if name == "deep-translator":
        print("  deep-translator : free")
        return DeepTranslatorTier()
    if name == "openrouter":
        keys = load_keys("OPENROUTER_API_KEY")
        if not keys:
            print("  ! bỏ openrouter: thiếu OPENROUTER_API_KEY", file=sys.stderr)
            return None
        print(
            f"  openrouter      : {args.openrouter_model} | TỐN TIỀN, "
            f"trần ${args.max_cost:.2f}"
        )
        return OpenRouterTier(args.openrouter_model, keys, args.max_cost)
    return None


def split_by_weight(items: list, assign: list[tuple[str, float]]) -> list[tuple[str, list]]:
    """Chia danh sách video thành các phần liên tục theo trọng số.

    Cắt liên tục (không xen kẽ) nên mỗi nguồn nhận trọn các thư mục liền nhau,
    dễ đối chiếu khi cần chạy lại đúng một phần.
    """
    total = sum(weight for _, weight in assign)
    shards: list[tuple[str, list]] = []
    start = 0
    for index, (name, weight) in enumerate(assign):
        if index == len(assign) - 1:
            end = len(items)
        else:
            end = min(len(items), start + round(len(items) * weight / total))
        shards.append((name, items[start:end]))
        start = end
    return shards


def split_by_folder(
    items: list, folders: list[str], assign: list[tuple[str, float]]
) -> list[tuple[str, list]]:
    """Như split_by_weight nhưng không bao giờ cắt ngang một thư mục con.

    Mỗi nguồn nhận trọn vài thư mục (Videos_L21_a, Videos_L22_a, ...). Thư mục
    to nhỏ khác nhau nên chia tham lam theo số video còn thiếu của từng nguồn,
    bám sát trọng số nhất có thể mà vẫn giữ nguyên ranh giới thư mục.
    """
    buckets: dict[str, list] = {}
    for item, folder in zip(items, folders):
        buckets.setdefault(folder, []).append(item)

    total_weight = sum(weight for _, weight in assign)
    quota = {
        name: len(items) * weight / total_weight for name, weight in assign
    }
    taken = {name: 0 for name, _ in assign}
    shards: dict[str, list] = {name: [] for name, _ in assign}

    # Thư mục lớn xếp trước để phần dư cuối cùng chỉ là các thư mục nhỏ, sai số
    # so với trọng số nhờ vậy nhỏ hơn hẳn so với chia theo thứ tự tên.
    for folder in sorted(buckets, key=lambda key: -len(buckets[key])):
        group = buckets[folder]
        name = max(assign, key=lambda pair: quota[pair[0]] - taken[pair[0]])[0]
        shards[name].extend(group)
        taken[name] += len(group)

    return [(name, shards[name]) for name, _ in assign]


# ---------------------------------------------------------------- CLI


def main() -> int:
    # Console Windows mặc định cp1252, mọi dòng log tiếng Việt sẽ ném
    # UnicodeEncodeError khi output bị chuyển hướng ra file hoặc pipe.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--assign",
        default="gemini=35,googletrans=25,deep-translator=25,openrouter=15",
        help="chia dataset cho từng nguồn theo trọng số, ví dụ 'gemini=50,googletrans=50'",
    )
    # Chọn theo kết quả dò key thực tế: gemini-3.6-flash 429 ngay, còn
    # gemini-2.5-flash trả 404 với các key mới. gemini-3.5-flash chạy được với
    # cả ba key nên tổng quota free gấp ba.
    parser.add_argument("--gemini-model", default="gemini-3.5-flash")
    parser.add_argument("--openrouter-model", default="deepseek/deepseek-v4-flash")
    parser.add_argument(
        "--max-cost",
        type=float,
        default=0.50,
        help="trần chi phí OpenRouter (USD); chạm trần thì tầng này tự tắt",
    )
    parser.add_argument(
        "--min-interval",
        type=float,
        default=4.0,
        help="giãn cách tối thiểu giữa 2 request của cùng một key Gemini (giây)",
    )
    parser.add_argument(
        "--max-wait",
        type=float,
        default=10.0,
        help="chờ Gemini quá ngần này giây thì nhường tầng free phía dưới",
    )
    parser.add_argument(
        "--by-folder",
        action="store_true",
        help="chia theo thư mục con: mỗi nguồn nhận trọn vài thư mục, không cắt ngang",
    )
    parser.add_argument("--limit", type=int, default=0, help="chỉ xử lý N video đầu")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--verbose", action="store_true", help="in lý do mỗi tầng bị bỏ qua"
    )
    parser.add_argument(
        "--fix-untranslated",
        action="store_true",
        help="quét thư mục output, dịch lại các segment còn nguyên tiếng Việt",
    )
    parser.add_argument(
        "--repair-truncated",
        action="store_true",
        help="quét thư mục output, dịch lại các segment bị Google cắt mất đuôi",
    )
    args = parser.parse_args()

    global VERBOSE
    VERBOSE = args.verbose

    if args.fix_untranslated or args.repair_truncated:
        if not args.out.is_dir():
            print(f"Chưa có thư mục kết quả: {args.out}", file=sys.stderr)
            return 1
        print("Nguồn dịch:")
        tiers = []
        for name, _ in parse_assign(args.assign):
            tier = make_tier(name, args)
            if tier is not None:
                tiers.append(tier)
        if not tiers:
            raise SystemExit("Không dựng được nguồn dịch nào.")
        cascade = Cascade(tiers)
        if args.fix_untranslated:
            code = run_fix(cascade, args.out)
            if code or not args.repair_truncated:
                return code
        return run_repair(cascade, args.out)

    if not args.src.is_dir():
        print(f"Không tìm thấy thư mục transcript: {args.src}", file=sys.stderr)
        return 1

    files = iter_transcripts(args.src)
    if args.limit:
        files = files[: args.limit]
    if not files:
        print(f"Không có file .json nào trong {args.src}", file=sys.stderr)
        return 1

    pending: list[tuple[Path, dict]] = []
    pending_folders: list[str] = []
    skipped = 0
    for path in files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as error:
            print(f"  ! đọc lỗi {path.name}: {error}", file=sys.stderr)
            continue
        relative = path.relative_to(args.src)
        out_path = args.out / relative
        if not args.overwrite and is_complete(out_path, len(data.get("segments") or [])):
            skipped += 1
            continue
        pending.append((out_path, data))
        pending_folders.append(relative.parts[0] if len(relative.parts) > 1 else ".")

    total_segments = sum(len(item[1].get("segments") or []) for item in pending)
    print(
        f"{len(files)} video | đã xong {skipped} | cần dịch {len(pending)} "
        f"({total_segments} segment)"
    )
    if not pending:
        print("Không còn gì để dịch.")
        return 0

    assign = parse_assign(args.assign)
    print("Nguồn dịch:")
    registry = {}
    for name in dict.fromkeys(name for name, _ in assign):
        tier = make_tier(name, args)
        if tier is not None:
            registry[name] = tier
    if not registry:
        raise SystemExit("Không dựng được nguồn dịch nào.")
    assign = [(name, weight) for name, weight in assign if name in registry]

    # Các tầng free còn lại làm lưới đỡ: nếu nguồn chính của một phần chết
    # (hết quota, bị chặn IP) thì phần đó vẫn chạy tiếp thay vì bỏ trống.
    # OpenRouter không bao giờ tự động đỡ vì nó tốn tiền.
    if args.by_folder:
        shards = split_by_folder(pending, pending_folders, assign)
    else:
        shards = split_by_weight(pending, assign)
    folder_of = dict(zip((id(item) for item in pending), pending_folders))
    print("\nPhân chia:")
    plans = []
    for name, items in shards:
        if not items:
            continue
        backups = [
            registry[other]
            for other in FREE_NAMES
            if other != name and other in registry
        ]
        segments = sum(len(data.get("segments") or []) for _, data in items)
        print(f"  {name:16s} {len(items):4d} video | {segments:6d} segment")
        if args.by_folder:
            names = sorted({folder_of[id(item)] for item in items})
            print(f"  {'':16s} {', '.join(names)}")
        plans.append((name, items, Cascade([registry[name]] + backups)))

    started = time.time()
    progress = {"done": 0}
    progress_lock = threading.Lock()

    def run_shard(name: str, items: list, cascade: Cascade) -> None:
        for position, (out_path, data) in enumerate(items, start=1):
            count = len(data.get("segments") or [])
            payload = translate_video(cascade, data)
            write_atomic(out_path, payload)

            with progress_lock:
                progress["done"] += count
                done = progress["done"]
            elapsed = time.time() - started
            rate = done / elapsed if elapsed else 0
            remaining = (total_segments - done) / rate if rate else 0
            print(
                f"[{name}] {position}/{len(items)} {data.get('video_id')} "
                f"{count} seg | tổng {done}/{total_segments} | {rate:.1f} seg/s | "
                f"còn ~{remaining / 60:.0f} phút",
                flush=True,
            )

    with ThreadPoolExecutor(max_workers=len(plans)) as pool:
        futures = [pool.submit(run_shard, *plan) for plan in plans]
        for future in futures:
            future.result()

    totals: dict[str, int] = {}
    failed = repaired = truncated = 0
    for _, _, cascade in plans:
        failed += cascade.failed
        repaired += cascade.repaired
        truncated += cascade.truncated
        for name, value in cascade.stats.items():
            totals[name] = totals.get(name, 0) + value

    print("\nThống kê nguồn dịch (theo mảnh thực tế):")
    for name, value in sorted(totals.items(), key=lambda item: -item[1]):
        print(f"  {name:16s} {value}")
    for name, tier in registry.items():
        if getattr(tier, "paid", False):
            print(
                f"\n{name}: {tier.prompt_tokens} token vào, "
                f"{tier.completion_tokens} token ra, chi phí thực ${tier.cost:.4f}"
            )
    if repaired:
        print(f"\nĐã cứu {repaired} đoạn bị Google cắt cụt bằng cách chia nhỏ.")
    if truncated:
        print(
            f"Còn {truncated} đoạn chia nhỏ vẫn cụt, đã giữ bản dài nhất.",
            file=sys.stderr,
        )
    if failed:
        print(f"\nCó {failed} đoạn dịch thất bại (để trống).", file=sys.stderr)
        print("Chạy lại script để dịch tiếp những video đó.", file=sys.stderr)
    print(f"\nXong sau {(time.time() - started) / 60:.1f} phút. Kết quả ở {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
