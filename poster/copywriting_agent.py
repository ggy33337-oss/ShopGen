import json

from llm.openai_client import extract_chat_content, post_chat_completion, strip_reasoning_tags
from poster.json_parser import load_strict_json_object
from poster.models.copywriting import CopywritingResult
from poster.models.poster_schema import PosterSchema


def build_copywriting_messages(poster_schema: PosterSchema):
    system_message = (
        "你是 Copywriting Agent，只接收 Poster Schema。"
        "你的任务是同时生成营销文案和结构化生图提示词字段。"
        "必须只返回严格 JSON 对象，不允许返回自然语言、Markdown 或代码块。"
        "不要直接生成最终 Prompt。image_prompt 必须保持结构化。"
        "JSON 格式固定为："
        '{"copywriting":{"headline":"","subheadline":"","cta":""},'
        '"image_prompt":{"subject":"","visual_style":"","lighting":"","composition":"","quality":""}}'
    )
    return [
        {"role": "system", "content": system_message},
        {
            "role": "user",
            "content": json.dumps(poster_schema.model_dump(), ensure_ascii=False),
        },
    ]


def generate_copywriting(values, poster_schema: PosterSchema):
    model_name = values.get("POSTER_COPYWRITING_MODEL_NAME") or values["MODEL_NAME"]
    response = post_chat_completion(
        values,
        build_copywriting_messages(poster_schema),
        {"temperature": 0.5, "max_tokens": 1200},
        model_name=model_name,
        response_format={"type": "json_object"},
    )
    raw_content = strip_reasoning_tags(extract_chat_content(response))
    return CopywritingResult.model_validate(load_strict_json_object(raw_content, "Copywriting Agent"))
