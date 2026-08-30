# -*- coding: utf-8 -*-

import json
import re
import socket
import time
from urllib import error, request

from runtime.models import Copywriting, ImagePromptPlan, IntentDecision


CHAT_REQUEST_TIMEOUT = 120
INTENT_REQUEST_TIMEOUT = 20
IMAGE_REQUEST_TIMEOUT = 420
MAX_IMAGE_ATTEMPTS = 2
RETRYABLE_HTTP_CODES = {429, 500, 502, 503, 504, 524}
DEFAULT_COMPAT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_API_BASE_URL = "https://dashscope.aliyuncs.com/api/v1"
DEFAULT_TEXT_MODEL = "qwen-plus"
DEFAULT_VL_MODEL = "qwen3-vl-plus"
DEFAULT_IMAGE_MODEL = "qwen-image-3.0-pro"


def strip_reasoning_tags(text):
    return re.sub(r"<think>.*?</think>", "", str(text or ""), flags=re.DOTALL).strip()


def extract_chat_content(response):
    choices = response.get("choices", []) if isinstance(response, dict) else []
    if not choices:
        return ""
    return str(choices[0].get("message", {}).get("content", "") or "")


def load_json_object(text, label):
    cleaned = strip_reasoning_tags(text).strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise RuntimeError(f"{label}未返回有效 JSON 对象。") from exc
        try:
            value = json.loads(cleaned[start:end + 1])
        except json.JSONDecodeError as nested_exc:
            raise RuntimeError(f"{label}未返回有效 JSON 对象。") from nested_exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{label}未返回 JSON 对象。")
    return value


def get_required_value(values, key):
    value = str(values.get(key, "") or "").strip()
    if not value:
        raise RuntimeError(f"缺少千问模型配置：{key}")
    return value


def get_proxy_url(values):
    return str(values.get("MODEL_PROXY_URL", "") or "").strip()


def open_model_request(values, req, timeout):
    proxy_url = get_proxy_url(values)
    if not proxy_url:
        return request.urlopen(req, timeout=timeout)
    opener = request.build_opener(request.ProxyHandler({"http": proxy_url, "https": proxy_url}))
    return opener.open(req, timeout=timeout)


def normalize_qwen_model(model_name, fallback, capability):
    model = str(model_name or fallback).strip()
    if not model.lower().startswith("qwen"):
        raise RuntimeError(f"{capability}模型必须使用千问模型，当前配置为：{model}")
    return model


class QwenGateway:
    def __init__(self, values):
        self.values = values
        self.api_key = get_required_value(values, "DASHSCOPE_API_KEY")
        self.compat_base_url = str(
            values.get("DASHSCOPE_COMPAT_BASE_URL") or DEFAULT_COMPAT_BASE_URL
        ).rstrip("/")
        self.api_base_url = str(values.get("DASHSCOPE_API_BASE_URL") or DEFAULT_API_BASE_URL).rstrip("/")
        self.text_model = normalize_qwen_model(
            values.get("QWEN_TEXT_MODEL"), DEFAULT_TEXT_MODEL, "文本"
        )
        self.vl_model = normalize_qwen_model(
            values.get("QWEN_VL_MODEL"), DEFAULT_VL_MODEL, "视觉理解"
        )
        self.image_model = normalize_qwen_model(
            values.get("QWEN_IMAGE_MODEL"), DEFAULT_IMAGE_MODEL, "图片生成"
        )

    def classify_request(
        self,
        user_input,
        history_messages,
        visual_history,
        has_uploaded_image=False,
        force_image=False,
    ):
        history_text = self._history_text(history_messages)
        visual_text = self._visual_history_text(visual_history)
        system_message = (
            "你是图片生成系统的第一层意图识别器。只返回严格 JSON 对象，字段为 "
            '{"intent":"text|image|mixed","use_previous_image":false,"reason":""}。'
            "intent=text 表示只需文字；image 表示只需图片；mixed 表示图片和配套文字。"
            "use_previous_image 仅当用户明确要求修改、延续、重做会话中的历史图片时为 true。"
            "若本轮上传了新图片但没有要求修改历史图片，use_previous_image 必须为 false。"
        )
        if force_image:
            system_message += "当前入口是海报生成入口，intent 必须为 image 或 mixed。"
        messages = [
            {"role": "system", "content": system_message},
            {
                "role": "user",
                "content": (
                    f"最近对话：\n{history_text}\n\n"
                    f"历史图片：\n{visual_text}\n\n"
                    f"本轮是否上传新图片：{'是' if has_uploaded_image else '否'}\n\n"
                    f"当前请求：{user_input}"
                ),
            },
        ]
        response = self.chat_completion(
            messages,
            model=self.text_model,
            temperature=0.0,
            max_tokens=200,
            response_format={"type": "json_object"},
            timeout=INTENT_REQUEST_TIMEOUT,
            error_label="千问意图识别模型",
        )
        payload = load_json_object(extract_chat_content(response), "千问意图识别模型")
        intent = str(payload.get("intent") or "").strip().lower()
        if force_image and intent == "text":
            intent = "mixed"
        if intent not in {"text", "image", "mixed"}:
            raise RuntimeError(f"千问意图识别结果无效：{intent or '空'}")
        return IntentDecision(
            intent=intent,
            use_previous_image=bool(payload.get("use_previous_image")),
            reason=str(payload.get("reason") or "").strip()[:500],
        )

    def generate_text(self, messages, intent="text"):
        options = {
            "text": (0.7, 2048),
            "image": (0.4, 1200),
            "mixed": (0.6, 2600),
        }
        temperature, max_tokens = options.get(intent, options["text"])
        response = self.chat_completion(
            messages,
            model=self.text_model,
            temperature=temperature,
            max_tokens=max_tokens,
            error_label="千问文本模型",
        )
        return strip_reasoning_tags(extract_chat_content(response))

    def plan_image_task(self, user_input, reference_images, context="", intent="image"):
        system_message = (
            "你是电商图片任务规划器，必须观察参考图并返回严格 JSON 对象。"
            "字段固定为 reply_text、image_prompt、copywriting。"
            "copywriting 固定包含 headline、subheadline、cta 三个字符串字段。"
            "image_prompt 供千问图片模型执行，应明确主体、保留项、修改项、场景、构图、光线、"
            "色彩、文字内容和负面约束；不得包含 URL、Base64 或内部编号。"
            "reply_text 是给用户看的简短说明，不得泄露完整提示词。"
        )
        content = [
            {
                "type": "text",
                "text": (
                    f"用户请求：{user_input}\n"
                    f"任务意图：{intent}\n"
                    f"补充上下文：{context or '无'}"
                ),
            }
        ]
        for image_url in reference_images[:3]:
            content.append({"type": "image_url", "image_url": {"url": image_url}})
        response = self.chat_completion(
            [{"role": "system", "content": system_message}, {"role": "user", "content": content}],
            model=self.vl_model,
            temperature=0.3,
            max_tokens=1800,
            response_format={"type": "json_object"},
            error_label="千问视觉规划模型",
        )
        payload = load_json_object(extract_chat_content(response), "千问视觉规划模型")
        prompt = str(payload.get("image_prompt") or "").strip()
        if not prompt:
            raise RuntimeError("千问视觉规划模型未返回可用的图片提示词。")
        copy_payload = payload.get("copywriting")
        if not isinstance(copy_payload, dict):
            copy_payload = {}
        return ImagePromptPlan(
            reply_text=str(payload.get("reply_text") or "已根据参考图生成新图片。").strip(),
            image_prompt=prompt[:6000],
            copywriting=Copywriting(
                headline=str(copy_payload.get("headline") or "").strip()[:400],
                subheadline=str(copy_payload.get("subheadline") or "").strip()[:400],
                cta=str(copy_payload.get("cta") or "").strip()[:400],
            ),
        )

    def generate_image(self, prompt, reference_images=None):
        normalized_prompt = " ".join(str(prompt or "").split())
        if not normalized_prompt:
            raise RuntimeError("千问图片模型缺少生图提示词。")
        content = [{"image": image} for image in (reference_images or [])[:3]]
        content.append({"text": normalized_prompt})
        parameters = {
            "n": 1,
            "negative_prompt": str(
                self.values.get("QWEN_IMAGE_NEGATIVE_PROMPT")
                or "低分辨率，低画质，主体畸形，结构错误，文字乱码，多余文字，水印，错误标志"
            )[:500],
            "prompt_extend": True,
            "watermark": False,
        }
        if not reference_images:
            parameters["size"] = str(self.values.get("QWEN_IMAGE_SIZE") or "2048*2048")
        payload = {
            "model": self.image_model,
            "input": {"messages": [{"role": "user", "content": content}]},
            "parameters": parameters,
        }
        endpoint = str(
            self.values.get("DASHSCOPE_IMAGE_ENDPOINT")
            or f"{self.api_base_url}/services/aigc/multimodal-generation/generation"
        ).strip()
        response = self._post_json(
            endpoint,
            payload,
            timeout=IMAGE_REQUEST_TIMEOUT,
            error_label="千问图片模型",
            attempts=MAX_IMAGE_ATTEMPTS,
        )
        image_url = self.extract_image_url(response)
        if not image_url:
            code = str(response.get("code") or "").strip()
            message = str(response.get("message") or "").strip()
            detail = f"{code} {message}".strip() or "响应中没有图片 URL"
            raise RuntimeError(f"千问图片模型生成失败：{detail}")
        return image_url

    def chat_completion(
        self,
        messages,
        model,
        temperature,
        max_tokens,
        response_format=None,
        timeout=CHAT_REQUEST_TIMEOUT,
        error_label="千问模型",
        attempts=1,
    ):
        payload = {
            "model": normalize_qwen_model(model, self.text_model, error_label),
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            payload["response_format"] = response_format
        return self._post_json(
            f"{self.compat_base_url}/chat/completions",
            payload,
            timeout=timeout,
            error_label=error_label,
            attempts=attempts,
        )

    def _post_json(self, endpoint, payload, timeout, error_label, attempts=1):
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        last_error = None
        for attempt in range(attempts):
            req = request.Request(
                endpoint,
                data=data,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json; charset=utf-8",
                },
                method="POST",
            )
            try:
                with open_model_request(self.values, req, timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                last_error = RuntimeError(f"{error_label}接口请求失败：HTTP {exc.code} {body}")
                if exc.code not in RETRYABLE_HTTP_CODES or attempt == attempts - 1:
                    raise last_error from exc
            except (ConnectionResetError, TimeoutError, socket.timeout, error.URLError) as exc:
                last_error = RuntimeError(
                    f"{error_label}响应超时或连接中断。当前单次等待上限：{timeout}秒。"
                )
                if attempt == attempts - 1:
                    raise last_error from exc
            time.sleep(2)
        raise last_error

    @staticmethod
    def extract_image_url(response):
        choices = response.get("output", {}).get("choices", []) if isinstance(response, dict) else []
        for choice in choices:
            content = choice.get("message", {}).get("content", [])
            for item in content:
                image_url = item.get("image") if isinstance(item, dict) else ""
                if image_url:
                    return str(image_url).strip()
        return ""

    @staticmethod
    def _history_text(history_messages, max_messages=8):
        lines = []
        for item in (history_messages or [])[-max_messages:]:
            role = str(item.get("role") or "").strip()
            content = str(item.get("content") or "").strip()
            if role in {"user", "assistant"} and content:
                lines.append(f"{role}: {content[:600]}")
        return "\n".join(lines) or "无"

    @staticmethod
    def _visual_history_text(visual_history):
        lines = []
        for index, item in enumerate(visual_history or [], start=1):
            if not str(item.get("image_url") or "").strip():
                continue
            lines.append(
                f"{index}. 用户需求：{str(item.get('user_input') or '')[:300]}；"
                f"助手说明：{str(item.get('assistant_text') or '')[:300]}"
            )
        return "\n".join(lines) or "无"
