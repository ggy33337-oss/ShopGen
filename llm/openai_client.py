import json
import re
import socket
import time
import uuid
from urllib import error, request


CHAT_REQUEST_TIMEOUT = 120
INTENT_REQUEST_TIMEOUT = 8
IMAGE_REQUEST_TIMEOUT = 420
IMAGE_RETRY_HTTP_CODES = {524}
IMAGE_MAX_ATTEMPTS = 2


def get_endpoint(values, endpoint_key, suffix):
    endpoint = str(values.get(endpoint_key, "")).strip()
    if endpoint:
        return endpoint

    base_url = values.get("OPENAI_COMPAT_BASE_URL", "").rstrip("/")
    if base_url:
        return f"{base_url}{suffix}"

    raise RuntimeError(f"缺少模型接口配置：{endpoint_key}")


def get_base_endpoint(values, base_url_key, suffix):
    base_url = str(values.get(base_url_key, "")).strip().rstrip("/")
    if not base_url:
        raise RuntimeError(f"缺少模型接口配置：{base_url_key}")
    return f"{base_url}{suffix}"


def get_required_value(values, key):
    value = str(values.get(key, "")).strip()
    if not value:
        raise RuntimeError(f"缺少模型配置：{key}")
    return value


def get_dashscope_api_key(values, specific_key=None):
    if specific_key:
        api_key = str(values.get(specific_key, "")).strip()
        if api_key:
            return api_key
    return get_required_value(values, "DASHSCOPE_API_KEY")


def get_image_api_key(values):
    return values.get("IMAGE_API_KEY") or values["OPENAI_API_KEY"]


def get_image_edit_endpoint(values):
    endpoint = str(values.get("IMAGE_EDIT_BASE_URL", "")).strip()
    if endpoint:
        return endpoint

    image_endpoint = str(values.get("IMAGE_BASE_URL", "")).strip()
    if image_endpoint.endswith("/images/generations"):
        return image_endpoint[:-len("/generations")] + "/edits"

    base_url = values.get("OPENAI_COMPAT_BASE_URL", "").rstrip("/")
    if base_url:
        return f"{base_url}/images/edits"

    raise RuntimeError("缺少图片编辑接口配置：IMAGE_EDIT_BASE_URL")


def get_proxy_url(values):
    return str(values.get("MODEL_PROXY_URL", "")).strip()


def open_model_request(values, req, timeout):
    proxy_url = get_proxy_url(values)
    if not proxy_url:
        return request.urlopen(req, timeout=timeout)

    opener = request.build_opener(request.ProxyHandler({
        "http": proxy_url,
        "https": proxy_url,
    }))
    return opener.open(req, timeout=timeout)


def get_chat_options(intent):
    options = {
        "text": {
            "temperature": 0.7,
            "max_tokens": 2048,
        },
        "image": {
            "temperature": 0.4,
            "max_tokens": 1200,
        },
        "mixed": {
            "temperature": 0.6,
            "max_tokens": 2600,
        },
    }
    return options.get(intent, options["text"])


def strip_reasoning_tags(text):
    return re.sub(r"<think>.*?</think>", "", str(text or ""), flags=re.DOTALL).strip()


def post_chat_completion(
    values,
    messages,
    options,
    model_name=None,
    response_format=None,
    timeout=CHAT_REQUEST_TIMEOUT,
):
    endpoint = get_endpoint(values, "OPENAI_BASE_URL", "/chat/completions")
    return post_chat_completion_to_endpoint(
        values=values,
        endpoint=endpoint,
        api_key=values["OPENAI_API_KEY"],
        messages=messages,
        options=options,
        model_name=model_name or values["MODEL_NAME"],
        response_format=response_format,
        timeout=timeout,
        error_label="文本模型",
    )


def post_chat_completion_to_endpoint(
    values,
    endpoint,
    api_key,
    messages,
    options,
    model_name,
    response_format=None,
    timeout=CHAT_REQUEST_TIMEOUT,
    error_label="文本模型",
):
    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": options["temperature"],
        "max_tokens": options["max_tokens"],
    }
    if response_format:
        payload["response_format"] = response_format

    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        endpoint,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with open_model_request(values, req, timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{error_label}接口请求失败：HTTP {e.code} {error_body}") from e
    except (ConnectionResetError, TimeoutError, socket.timeout, error.URLError) as e:
        raise RuntimeError(
            f"{error_label}响应超时或连接中断，请稍后重试。"
            f"当前单次等待上限：{timeout}秒。"
        ) from e


def post_dashscope_chat_completion(
    values,
    messages,
    options,
    model_name,
    response_format=None,
    timeout=CHAT_REQUEST_TIMEOUT,
    api_key_name=None,
    error_label="百炼模型",
):
    endpoint = get_base_endpoint(
        values,
        "DASHSCOPE_COMPAT_BASE_URL",
        "/chat/completions",
    )
    return post_chat_completion_to_endpoint(
        values=values,
        endpoint=endpoint,
        api_key=get_dashscope_api_key(values, api_key_name),
        messages=messages,
        options=options,
        model_name=model_name,
        response_format=response_format,
        timeout=timeout,
        error_label=error_label,
    )


def extract_chat_content(response):
    choices = response.get("choices", [])
    if not choices:
        return ""
    message = choices[0].get("message", {})
    return message.get("content", "")


def generate_text(values, messages, intent="text"):
    options = get_chat_options(intent)
    response = post_chat_completion(values, messages, options)
    return strip_reasoning_tags(extract_chat_content(response))


def normalize_image_response(data):
    image_url = ""
    images = data.get("data", [])
    if images:
        first_image = images[0]
        image_url = first_image.get("url") or first_image.get("image") or ""
        b64_json = first_image.get("b64_json") or ""
        if not image_url and b64_json:
            image_url = f"data:image/png;base64,{b64_json}"

    return {
        "output": {
            "choices": [
                {
                    "message": {
                        "content": [
                            {
                                "image": image_url,
                                "raw": data,
                            }
                        ]
                    }
                }
            ]
        }
    }


def build_image_payload(values, prompt):
    normalized_prompt = " ".join(str(prompt or "").split())

    return {
        "model": values["IMAGE_MODEL_NAME"],
        "prompt": normalized_prompt,
        "n": 1,
        "size": "1024x1024",
        "response_format": "url",
    }


def generate_image(values, prompt):
    payload = build_image_payload(values, prompt)
    data = json.dumps(payload).encode("utf-8")
    endpoint = get_endpoint(values, "IMAGE_BASE_URL", "/images/generations")
    headers = {
        "Authorization": f"Bearer {get_image_api_key(values)}",
        "Content-Type": "application/json",
    }
    last_error = None
    for attempt in range(IMAGE_MAX_ATTEMPTS):
        req = request.Request(endpoint, data=data, headers=headers, method="POST")
        try:
            with open_model_request(values, req, IMAGE_REQUEST_TIMEOUT) as response:
                return normalize_image_response(json.loads(response.read().decode("utf-8")))
        except error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(f"图片模型接口请求失败：HTTP {e.code} {error_body}")
            if e.code not in IMAGE_RETRY_HTTP_CODES or attempt == IMAGE_MAX_ATTEMPTS - 1:
                raise last_error from e
            time.sleep(2)
        except (ConnectionResetError, TimeoutError, socket.timeout, error.URLError) as e:
            last_error = RuntimeError(
                "图片生成耗时过长或服务商连接被中断，请稍后重试，"
                "或减少画面要求、改用更快的图片模型。"
                f"当前单次等待上限：{IMAGE_REQUEST_TIMEOUT}秒。"
            )
            if attempt == IMAGE_MAX_ATTEMPTS - 1:
                raise last_error from e
            time.sleep(2)
    raise last_error


def generate_image_edit(values, prompt, image_content, image_content_type="image/png"):
    data, content_type = build_multipart_form_data(
        fields={
            "model": values["IMAGE_MODEL_NAME"],
            "prompt": " ".join(str(prompt or "").split()),
            "n": "1",
            "size": "1024x1024",
            "response_format": "url",
        },
        files={
            "image": {
                "filename": "reference.png",
                "content_type": image_content_type or "image/png",
                "content": image_content,
            },
        },
    )
    endpoint = get_image_edit_endpoint(values)
    headers = {
        "Authorization": f"Bearer {get_image_api_key(values)}",
        "Content-Type": content_type,
    }
    last_error = None
    for attempt in range(IMAGE_MAX_ATTEMPTS):
        req = request.Request(endpoint, data=data, headers=headers, method="POST")
        try:
            with open_model_request(values, req, IMAGE_REQUEST_TIMEOUT) as response:
                return normalize_image_response(json.loads(response.read().decode("utf-8")))
        except error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(f"图片编辑接口请求失败：HTTP {e.code} {error_body}")
            if e.code not in IMAGE_RETRY_HTTP_CODES or attempt == IMAGE_MAX_ATTEMPTS - 1:
                raise last_error from e
            time.sleep(2)
        except (ConnectionResetError, TimeoutError, socket.timeout, error.URLError) as e:
            last_error = RuntimeError(
                "图片编辑耗时过长或服务商连接被中断，请稍后重试。"
                f"当前单次等待上限：{IMAGE_REQUEST_TIMEOUT}秒。"
            )
            if attempt == IMAGE_MAX_ATTEMPTS - 1:
                raise last_error from e
            time.sleep(2)
    raise last_error


def build_multipart_form_data(fields, files):
    boundary = f"----codexform{uuid.uuid4().hex}"
    chunks = []
    for name, value in fields.items():
        chunks.extend([
            f"--{boundary}\r\n".encode("utf-8"),
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"),
            str(value).encode("utf-8"),
            b"\r\n",
        ])
    for name, file_info in files.items():
        filename = file_info.get("filename") or "file"
        content_type = file_info.get("content_type") or "application/octet-stream"
        content = file_info.get("content") or b""
        chunks.extend([
            f"--{boundary}\r\n".encode("utf-8"),
            (
                f'Content-Disposition: form-data; name="{name}"; '
                f'filename="{filename}"\r\n'
            ).encode("utf-8"),
            f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"),
            content,
            b"\r\n",
        ])
    chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def build_recent_history_text(history_messages, max_messages=6):
    if not history_messages:
        return "无"

    recent_messages = history_messages[-max_messages:]
    lines = []
    for message in recent_messages:
        role = message.get("role", "")
        content = str(message.get("content", "")).strip()
        if role in ["user", "assistant"] and content:
            lines.append(f"{role}: {content[:500]}")
    return "\n".join(lines) if lines else "无"


def classify_intent(user_input, values, history_messages=None):
    model_name = (
        values.get("INTENT_MODEL_NAME")
        or values.get("DASHSCOPE_TEXT_MODEL_NAME")
        or "qwen-plus"
    )
    system_message = (
        "你是一个调用链意图分类器，只判断本轮需要调用哪类模型。"
        "只能返回 text、image、mixed 三个词之一。"
        "text 表示只需要文字回答，包括普通聊天、解释、追问、评价上一版、指出不足、优化建议、只要文案或标题；"
        "image 表示本轮明确只需要生成图片；"
        "mixed 表示本轮明确同时需要图片和文字解释或文案。"
        "如果用户只是说“不够大气”“不够高级”“不好看”“哪里不对”“帮我优化一下思路”，返回 text。"
        "如果用户说“根据这个图片/参考这张海报/基于上一张图写文案”，这是基于已有素材写文字，返回 text，不要返回 image 或 mixed。"
        "只有用户本轮明确要求生成新的图片、海报、主图、配图时，才返回 image 或 mixed。"
        "你可以参考最近历史判断“这个图片”“上一版”“这张图”指的是什么，但最终只判断本轮是否需要调用生图模型。"
        "不要输出解释，不要输出标点，不要输出 JSON。"
    )
    messages = [
        {"role": "system", "content": system_message},
        {
            "role": "user",
            "content": (
                f"最近历史：\n{build_recent_history_text(history_messages)}\n\n"
                f"当前用户输入：{user_input}"
            ),
        },
    ]
    response = post_dashscope_chat_completion(
        values,
        messages,
        {
            "temperature": 0.0,
            "max_tokens": 10,
        },
        model_name=model_name,
        timeout=INTENT_REQUEST_TIMEOUT,
        api_key_name="INTENT_API_KEY",
        error_label="百炼意图识别模型",
    )
    return extract_chat_content(response).strip().lower()
