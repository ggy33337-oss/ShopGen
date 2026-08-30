# -*- coding: utf-8 -*-

import base64
from dataclasses import dataclass
from urllib import parse, request

from llm.qwen_client import (
    QwenGateway,
    extract_chat_content,
    open_model_request,
    strip_reasoning_tags,
)
from infra.redis_image_cache import RedisImageCache, extract_image_id
from poster.multimodal_analyzer import ANALYZER_MODEL_FALLBACK


MAX_REFERENCE_IMAGE_BYTES = 10 * 1024 * 1024
REFERENCE_IMAGE_TIMEOUT = 20
MAX_REFERENCE_IMAGES = 1
REFERENCE_ANALYSIS_TIMEOUT = 80
SUPPORTED_IMAGE_TYPES = {
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/gif",
}


@dataclass(frozen=True)
class ReferenceImage:
    content: bytes
    content_type: str
    source_url: str


def attach_visual_reference_analysis(values, user_input, visual_history, limit=MAX_REFERENCE_IMAGES):
    enriched_history = []
    analysis_count = 0
    items = list(visual_history or [])
    selected_items = [item for item in items if item.get("selected_for_edit")]
    fallback_items = items[-limit:] if not selected_items else []
    target_turn_ids = {
        item.get("turn_id")
        for item in [*selected_items, *fallback_items]
        if item.get("turn_id")
    }
    for index, item in enumerate(items, start=1):
        record = dict(item)
        should_analyze = record.get("turn_id") in target_turn_ids and analysis_count < limit
        if should_analyze:
            image_data_url = load_image_data_url(values, record.get("image_url", ""))
            if image_data_url:
                image_analysis = analyze_reference_image(values, user_input, record, image_data_url, index)
                if image_analysis:
                    record["image_analysis"] = image_analysis
                analysis_count += 1
        enriched_history.append(record)
    return enriched_history


def analyze_reference_image(values, user_input, record, image_data_url, index):
    model_name = (
        values.get("QWEN_VL_MODEL")
        or ANALYZER_MODEL_FALLBACK
    )
    system_message = (
        "你是历史参考图视觉分析器。"
        "请直接观察图片内容，结合当前用户修改要求，输出一段可用于重新生成图片的中文视觉摘要。"
        "重点包含主体、构图、背景、色调、材质、光线、可见文字/元素、需要保留的细节和本轮需要修改的点。"
        "不要输出 Markdown，不要输出 JSON，不要超过 700 字。"
    )
    user_text = (
        f"历史参考图序号：{index}\n"
        f"当前用户要求：{user_input}\n"
        f"历史用户需求：{record.get('user_input', '')}\n"
        f"历史助手说明：{record.get('assistant_text', '')}\n"
        f"历史生图提示词：{record.get('image_prompt', '')[:1200]}\n"
        "请观察下方图片，并给出重新生成时应该保留和修改的视觉要点。"
    )
    messages = [
        {"role": "system", "content": system_message},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ],
        },
    ]
    try:
        gateway = QwenGateway(values)
        response = gateway.chat_completion(
            messages=messages,
            model=model_name,
            temperature=0.0,
            max_tokens=900,
            timeout=REFERENCE_ANALYSIS_TIMEOUT,
            error_label="历史参考图视觉分析模型",
        )
    except Exception:
        return ""
    return strip_reasoning_tags(extract_chat_content(response))[:1200]


def load_image_data_url(values, image_url):
    reference_image = load_reference_image(values, image_url)
    if not reference_image:
        return ""

    encoded = base64.b64encode(reference_image.content).decode("ascii")
    return f"data:{reference_image.content_type};base64,{encoded}"


def load_reference_image(values, image_url):
    image_url = str(image_url or "").strip()
    if not image_url:
        return None
    cached_image_id = extract_image_id(image_url)
    if cached_image_id:
        cached = RedisImageCache(values).get_image(cached_image_id)
        if not cached:
            return None
        return ReferenceImage(
            content=cached.content,
            content_type=cached.content_type,
            source_url=image_url,
        )
    if image_url.startswith("data:image/"):
        return parse_data_url(image_url)

    parsed_url = parse.urlparse(image_url)
    if parsed_url.scheme not in {"http", "https"}:
        return None

    try:
        req = request.Request(
            image_url,
            headers={"User-Agent": "ecommerce-visual-assistant/1.0"},
            method="GET",
        )
        with open_model_request(values, req, REFERENCE_IMAGE_TIMEOUT) as response:
            content_type = normalize_content_type(response.headers.get("Content-Type", ""))
            content = response.read(MAX_REFERENCE_IMAGE_BYTES + 1)
    except Exception:
        return None

    if len(content) > MAX_REFERENCE_IMAGE_BYTES:
        return None
    if content_type not in SUPPORTED_IMAGE_TYPES:
        content_type = infer_content_type(image_url)
    if content_type not in SUPPORTED_IMAGE_TYPES:
        return None

    return ReferenceImage(
        content=content,
        content_type=content_type,
        source_url=image_url,
    )


def parse_data_url(data_url):
    header, separator, encoded = str(data_url or "").partition(",")
    if separator != "," or ";base64" not in header:
        return None
    content_type = normalize_content_type(header.removeprefix("data:").replace(";base64", ""))
    if content_type not in SUPPORTED_IMAGE_TYPES:
        return None
    try:
        content = base64.b64decode(encoded, validate=True)
    except Exception:
        return None
    if not content or len(content) > MAX_REFERENCE_IMAGE_BYTES:
        return None
    return ReferenceImage(
        content=content,
        content_type=content_type,
        source_url="data-url",
    )


def normalize_content_type(content_type):
    return str(content_type or "").split(";", 1)[0].strip().lower()


def infer_content_type(image_url):
    path = parse.urlparse(str(image_url or "")).path.lower()
    if path.endswith(".png"):
        return "image/png"
    if path.endswith((".jpg", ".jpeg")):
        return "image/jpeg"
    if path.endswith(".webp"):
        return "image/webp"
    if path.endswith(".gif"):
        return "image/gif"
    return ""
