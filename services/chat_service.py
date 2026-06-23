import json
import re
import time

from pydantic import ValidationError

from core.config import read_config
from infra.conversation_store import (
    append_visual_history,
    append_messages,
    get_history_messages,
    get_recent_visual_history,
    normalize_conversation_id,
)
from infra.logger import write_log
from llm.openai_client import (
    classify_intent,
    generate_image,
    generate_image_edit,
    generate_text,
    get_chat_options,
)
from llm.models import VisualModelResponse
from llm.prompt_builder import build_message
from services.visual_reference_loader import (
    attach_visual_reference_analysis,
    load_reference_image,
)


VALID_INTENTS = ["text", "image", "mixed"]
IMAGE_REFERENCE_KEYWORDS = [
    "图",
    "图片",
    "海报",
    "上一张",
    "上面",
    "刚才",
    "上一版",
    "这张",
    "那个",
    "细节",
    "优化",
    "重生成",
    "重新生成",
    "参考",
]
IMAGE_EDIT_KEYWORDS = [
    "背景颜色",
    "背景色",
    "换背景",
    "改背景",
    "背景换",
    "背景改",
    "换成",
    "改成",
    "颜色换",
    "换颜色",
    "改颜色",
    "色调换",
    "只把",
    "仅将",
    "保持",
    "保留",
]
BATCH_REFERENCE_KEYWORDS = [
    "这些图片",
    "这些图",
    "全部",
    "一起",
    "所有",
]
NUMBER_WORDS = {
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "1": 1,
    "2": 2,
    "3": 3,
}


class IntentClassificationError(Exception):
    pass


class IntentClassificationServiceError(Exception):
    pass


class HandledResponse(Exception):
    pass


def normalize_intent(raw_intent):
    text = str(raw_intent or "").strip().lower()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    tokens = re.findall(r"[a-z]+", text)
    valid_tokens = [token for token in tokens if token in VALID_INTENTS]
    if valid_tokens:
        return valid_tokens[-1]
    return text


def build_visual_history_hint(visual_history):
    if not visual_history:
        return ""

    lines = ["当前会话最近生成图片记录："]
    for index, item in enumerate(visual_history, start=1):
        lines.append(
            f"{index}. 用户需求：{item.get('user_input', '')[:200]}；"
            f"助手说明：{item.get('assistant_text', '')[:200]}；"
            f"图片提示词：{item.get('image_prompt', '')[:300]}"
        )
    return "\n".join(lines)


def taxonomy_classification(user_input, values, history_messages=None, visual_history=None):
    try:
        classified_input = user_input
        visual_history_hint = build_visual_history_hint(visual_history)
        if visual_history_hint:
            classified_input = f"{visual_history_hint}\n\n当前用户输入：{user_input}"
        intent = normalize_intent(classify_intent(classified_input, values, history_messages))
    except Exception as e:
        raise IntentClassificationServiceError("意图识别接口调用失败") from e

    if intent in VALID_INTENTS:
        return intent
    raise IntentClassificationError(f"意图识别结果无效：{intent}")


def should_load_visual_reference_images(user_input, intent):
    if intent in ["image", "mixed"]:
        return True
    text = str(user_input or "")
    return any(keyword in text for keyword in IMAGE_REFERENCE_KEYWORDS)


def should_edit_existing_image(user_input, intent, visual_history):
    if intent not in ["image", "mixed"] or not visual_history:
        return False
    text = str(user_input or "")
    has_edit_keyword = any(keyword in text for keyword in IMAGE_EDIT_KEYWORDS)
    has_image_reference = any(
        keyword in text
        for keyword in ["图", "图片", "上面", "上一张", "这张", "那个", "刚才"]
    )
    return has_edit_keyword and has_image_reference


def is_batch_image_edit_request(user_input):
    text = str(user_input or "")
    return any(keyword in text for keyword in BATCH_REFERENCE_KEYWORDS)


def get_visual_records_with_images(visual_history):
    return [
        item
        for item in visual_history or []
        if str(item.get("image_url") or "").strip()
    ]


def select_visual_record(user_input, visual_history):
    records = get_visual_records_with_images(visual_history)
    if not records:
        return {}, 0

    explicit_index = extract_requested_visual_index(user_input)
    if explicit_index:
        if explicit_index > len(records):
            return {}, explicit_index
        return records[explicit_index - 1], explicit_index

    return records[-1], len(records)


def extract_requested_visual_index(user_input):
    text = str(user_input or "")
    patterns = [
        r"第\s*([123一二两三])\s*(?:张|个|版|幅)",
        r"(?<![上前后下])([123一二两三])\s*(?:张|个|版|幅)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        return NUMBER_WORDS.get(match.group(1), 0)
    return 0


def mark_selected_visual_record(visual_history, selected_record, selected_index):
    if not selected_record:
        return visual_history
    enriched = []
    selected_turn_id = selected_record.get("turn_id")
    for index, item in enumerate(visual_history or [], start=1):
        record = dict(item)
        if selected_turn_id and record.get("turn_id") == selected_turn_id:
            record["selected_for_edit"] = True
            record["selected_index"] = selected_index
        enriched.append(record)
    return enriched


def build_image_edit_unavailable_response(reason):
    return (
        "这类需求属于“基于上一张图做局部编辑”，例如只换背景颜色。"
        "我不会再用纯文字生图去硬替代，因为那样很容易生成另一张图。\n\n"
        f"当前没有完成图片编辑：{reason}\n\n"
        "请确认图片编辑接口可用后再试，或重新上传原图让我按图处理。"
    )


def build_image_edit_fallback_visual_response(user_input, selected_record):
    previous_prompt = str(selected_record.get("image_prompt") or "").strip()
    image_analysis = str(selected_record.get("image_analysis") or "").strip()
    context_parts = []
    if image_analysis:
        context_parts.append(f"参考图视觉摘要：{image_analysis[:900]}")
    if previous_prompt:
        context_parts.append(f"上一轮生图提示词：{previous_prompt[:900]}")
    context = "\n".join(context_parts) or "参考图内容以本次上传/传入的图片为准。"
    image_prompt = (
        "基于提供的参考图片进行局部图片编辑，严格保留原图主体、产品外形、摆放位置、"
        "构图比例、画面层次、材质细节和商业商品展示质感。"
        f"本轮只执行用户要求：{user_input}。"
        "如果用户要求修改背景颜色或整体背景色，只修改背景和环境色调，主体颜色、结构、"
        "纹理、透明部件、金属高光和阴影关系尽量保持原样。"
        "保持高清写实商业摄影质感，背景干净有层次，光线自然协调。"
        "避免主体变形、产品结构改变、主体缺失、背景杂乱、文字乱码、多余文字、水印、"
        "logo错乱、过度饱和、低清晰度、噪点、畸变。"
        f"\n{context}"
    )
    return {
        "reply_text": "我会基于上一张图继续编辑，尽量保留主体和构图，只按你的要求调整背景颜色。",
        "image_prompt": image_prompt[:2000],
    }


def build_batch_image_edit_response():
    return (
        "你这句话指向多张图片批量编辑，但当前链路一次只能稳定编辑一张原图。"
        "为了避免我随便选一张导致结果跑偏，请明确说：\n\n"
        "第1张背景换成绿色\n"
        "第2张背景换成绿色\n"
        "上一张背景换成绿色"
    )


def parse_visual_response(raw_response):
    raw_response = unwrap_json_string(str(raw_response or "").strip())
    try:
        parsed = json.loads(raw_response)
    except json.JSONDecodeError:
        json_start = raw_response.find("{")
        json_end = raw_response.rfind("}")
        if json_start != -1 and json_end != -1 and json_start < json_end:
            try:
                parsed = json.loads(raw_response[json_start:json_end + 1])
            except json.JSONDecodeError:
                parsed = None
        else:
            parsed = None

    if parsed is None or not isinstance(parsed, dict):
        return {
            "reply_text": "已根据你的需求生成图片。",
            "image_prompt": raw_response,
        }
    nested_prompt = parsed.get("image_prompt")
    if isinstance(nested_prompt, str):
        nested_text = unwrap_json_string(nested_prompt.strip())
        if nested_text.startswith("{") and nested_text.endswith("}"):
            try:
                nested = json.loads(nested_text)
                if isinstance(nested, dict) and any(key in nested for key in ["reply_text", "image_prompt"]):
                    parsed = nested
            except json.JSONDecodeError:
                pass

    try:
        visual_response = VisualModelResponse.model_validate(parsed)
    except ValidationError:
        visual_response = VisualModelResponse()

    reply_text = visual_response.reply_text
    image_prompt = visual_response.image_prompt
    if not reply_text:
        reply_text = "已根据你的需求生成图片。"

    return {
        "reply_text": reply_text,
        "image_prompt": image_prompt,
    }


def unwrap_json_string(text):
    current = str(text or "").strip()
    for _ in range(2):
        if len(current) < 2:
            break
        try:
            decoded = json.loads(current)
        except json.JSONDecodeError:
            break
        if not isinstance(decoded, str):
            break
        current = decoded.strip()
    return current


def build_missing_image_prompt_response(user_input, visual_history):
    if visual_history:
        latest = visual_history[-1]
        return (
            "我理解你是想基于上一张图继续生成，但这轮模型没有返回可用的生图提示词。"
            "当前最近一张图的记录是：\n"
            f"用户需求：{latest.get('user_input', '')}\n"
            f"图片提示词：{latest.get('image_prompt', '')[:800]}\n\n"
            "你可以直接说“按这张图再生成一版”，我会继续基于这条记录处理。"
        )
    return (
        "我理解你是想生成图片，但这轮模型没有返回可用的生图提示词。"
        f"请补充一下图片主体、场景或风格。当前输入：{user_input}"
    )


def is_transient_model_error(error):
    message = str(error)
    transient_markers = [
        "timed out",
        "timeout",
        "超时",
        "连接中断",
        "Connection reset",
        "WinError 10054",
        "远程主机强迫关闭",
    ]
    return any(marker in message for marker in transient_markers)


def extract_assistant_text(response):
    if isinstance(response, dict):
        return str(response.get("text") or "").strip()
    return str(response or "").strip()


def extract_image_url(response):
    if not isinstance(response, dict):
        return ""

    choices = response.get("image", {}).get("output", {}).get("choices", [])
    for choice in choices:
        content = choice.get("message", {}).get("content", [])
        for item in content:
            image_url = item.get("image")
            if image_url:
                return str(image_url).strip()
    return ""


def ask(user_input, conversation_id=None):
    start_time = time.perf_counter()
    conversation_id = normalize_conversation_id(conversation_id)

    if user_input == "":
        response = (
            "您好！您输入了空内容，请问您是想：\n\n"
            "生成一张产品图？例如：商品主图、详情页、海报。\n"
            "为具体商品撰写营销文案或卖点解析？\n"
            "获取电商视觉设计建议？\n"
            "还是需要优化已有文案或图片描述？"
        )
        latency_ms = int((time.perf_counter() - start_time) * 1000)
        return response, "", latency_ms, conversation_id

    values = read_config(".env")
    intent = None
    image_prompt = ""
    visual_history = []
    selected_visual_record = {}
    selected_visual_index = 0
    try:
        history_messages = get_history_messages(conversation_id)
        visual_history = get_recent_visual_history(conversation_id)
        intent = taxonomy_classification(user_input, values, history_messages, visual_history)
        is_edit_request = should_edit_existing_image(user_input, intent, visual_history)
        if is_edit_request and is_batch_image_edit_request(user_input):
            response = build_batch_image_edit_response()
            raise HandledResponse
        if is_edit_request:
            selected_visual_record, selected_visual_index = select_visual_record(user_input, visual_history)
            if selected_visual_index and not selected_visual_record:
                response = build_image_edit_unavailable_response(
                    f"最近三张图里没有第 {selected_visual_index} 张可编辑图片。"
                )
                raise HandledResponse
            visual_history = mark_selected_visual_record(
                visual_history,
                selected_visual_record,
                selected_visual_index,
            )
        if should_load_visual_reference_images(user_input, intent):
            visual_history = attach_visual_reference_analysis(values, user_input, visual_history)
        messages = build_message(user_input, intent, history_messages, visual_history)
        visual_response = None
        try:
            response = generate_text(values, messages, intent)
        except Exception as e:
            if (
                intent in ["image", "mixed"]
                and is_edit_request
                and selected_visual_record
                and is_transient_model_error(e)
            ):
                visual_response = build_image_edit_fallback_visual_response(user_input, selected_visual_record)
                response = visual_response["reply_text"]
            else:
                raise
        if intent in ["image", "mixed"]:
            if visual_response is None:
                visual_response = parse_visual_response(response)
            image_prompt = visual_response["image_prompt"]
            if not image_prompt.strip():
                if is_edit_request and selected_visual_record:
                    visual_response = build_image_edit_fallback_visual_response(user_input, selected_visual_record)
                    image_prompt = visual_response["image_prompt"]
                    response = visual_response["reply_text"]
                else:
                    response = build_missing_image_prompt_response(user_input, visual_history)
            if image_prompt.strip():
                if is_edit_request:
                    if not selected_visual_record:
                        response = build_image_edit_unavailable_response("最近三轮里没有可用的历史图片。")
                    else:
                        reference_image = load_reference_image(values, selected_visual_record.get("image_url", ""))
                        if not reference_image:
                            response = build_image_edit_unavailable_response("没有读取到选中的历史图片文件。")
                        else:
                            image_result = generate_image_edit(
                                values,
                                image_prompt,
                                reference_image.content,
                                reference_image.content_type,
                            )
                            response = {
                                "text": visual_response["reply_text"],
                                "image": image_result,
                            }
                else:
                    image_result = generate_image(values, image_prompt)
                    response = {
                        "text": visual_response["reply_text"],
                        "image": image_result,
                    }
    except IntentClassificationError:
        response = (
            "我还没有判断清楚你的需求。你是想只要文字内容，"
            "还是重新生成图片，或者同时生成图片和文案？请再明确一下。"
        )
    except IntentClassificationServiceError:
        response = "意图识别失败，请稍后重试。"
    except HandledResponse:
        pass
    except Exception as e:
        response = f"抱歉，发生了一个错误：{str(e)}"
    finally:
        assistant_text = extract_assistant_text(response)
        assistant_message = {"role": "assistant", "content": assistant_text}
        image_url = extract_image_url(response)
        if image_url:
            assistant_message["image_url"] = image_url
            append_visual_history(conversation_id, {
                "image_url": image_url,
                "user_input": user_input,
                "assistant_text": assistant_text,
                "image_prompt": image_prompt,
            })

        append_messages(conversation_id, [
            {"role": "user", "content": user_input},
            assistant_message,
        ])

    latency_ms = int((time.perf_counter() - start_time) * 1000)
    write_log({
        "conversation_id": conversation_id,
        "user_input": user_input,
        "response": response,
        "latency_ms": latency_ms,
        "model": values.get("MODEL_NAME", ""),
        "image_model": values.get("IMAGE_MODEL_NAME", ""),
        "intent": intent or "unknown",
        "image_prompt": image_prompt,
        "chat_options": get_chat_options(intent or "text"),
    })
    return response, values.get("MODEL_NAME", ""), latency_ms, conversation_id
