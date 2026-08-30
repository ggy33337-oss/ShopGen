# -*- coding: utf-8 -*-

from llm.qwen_client import QwenGateway, extract_chat_content, strip_reasoning_tags
from poster.json_parser import load_strict_json_object
from poster.models.analysis import AnalyzerResult


ANALYZER_MODEL_FALLBACK = "qwen3-vl-plus"


def build_analyzer_messages(user_text, uploaded_content=None):
    system_message = (
        "你是 MultiModal Analyzer，负责分析用户上传内容和文字需求。"
        "支持 png、jpg、jpeg、pdf、docx 和 plain text。"
        "必须只返回严格 JSON 对象，不允许返回自然语言、Markdown 或代码块。"
        "JSON 字段固定为：product、style、layout、color_scheme、selling_points、reference_summary。"
        "selling_points 必须是字符串数组；无法判断的字段返回空字符串或空数组。"
    )

    text_parts = [f"用户原始需求：{user_text}"]
    if uploaded_content and uploaded_content.text:
        text_parts.append(
            f"上传文件名：{uploaded_content.filename}\n"
            f"上传文件提取文本：\n{uploaded_content.text}"
        )
    elif uploaded_content:
        text_parts.append(f"上传文件名：{uploaded_content.filename}")

    user_payload = "\n\n".join(text_parts)
    data_urls = tuple(uploaded_content.data_urls) if uploaded_content else ()
    if data_urls:
        content = [{"type": "text", "text": user_payload}]
        for data_url in data_urls:
            content.append({"type": "image_url", "image_url": {"url": data_url}})
    else:
        content = user_payload

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": content},
    ]


def analyze_multimodal(values, user_text, uploaded_content=None):
    model_name = (
        values.get("QWEN_VL_MODEL")
        or ANALYZER_MODEL_FALLBACK
    )
    gateway = QwenGateway(values)
    response = gateway.chat_completion(
        messages=build_analyzer_messages(user_text, uploaded_content),
        model=model_name,
        temperature=0.0,
        max_tokens=1200,
        response_format={"type": "json_object"},
        error_label="百炼多模态分析模型",
    )
    raw_content = strip_reasoning_tags(extract_chat_content(response))
    return AnalyzerResult.model_validate(load_strict_json_object(raw_content, "MultiModal Analyzer"))
