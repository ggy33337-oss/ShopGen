# -*- coding: utf-8 -*-

import base64
import binascii
import json
import re
import socket
import ssl
import time
import uuid
from urllib import error, request
from urllib.parse import urlsplit

from infra.logger import (
    generation_log_context,
    generation_stage,
    get_generation_deadline,
    write_generation_event,
)
from runtime.models import Copywriting, ImagePromptPlan, IntentDecision


CHAT_REQUEST_TIMEOUT = 120
INTENT_REQUEST_TIMEOUT = 20
IMAGE_REQUEST_TIMEOUT = 420
DEFAULT_IMAGE_TOTAL_TIMEOUT = 420
MAX_IMAGE_ATTEMPTS = 2
RETRYABLE_HTTP_CODES = {429, 500, 502, 503, 504, 524}
DEFAULT_COMPAT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_API_BASE_URL = "https://dashscope.aliyuncs.com/api/v1"
DEFAULT_TEXT_MODEL = "qwen3.8-max"
DEFAULT_VL_MODEL = "qwen3-vl-plus"
DEFAULT_IMAGE_MODEL = "wanx2.1-imageedit"
WANX_IMAGE_EDIT_MODEL = "wanx2.1-imageedit"
DEFAULT_GPT_IMAGE_MODEL = "gpt-image-2"
WANX_IMAGE_POLL_INTERVAL = 2


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


def coerce_bool(value, default=False):
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y", "是", "真"}
    return bool(value)


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


def normalize_image_model(model_name, fallback, capability):
    model = str(model_name or fallback).strip()
    if not model.lower().startswith(("qwen", "wan", "wanx", "gpt-image")):
        raise RuntimeError(
            f"{capability}模型必须使用千问、Wanx 或 GPT Image 模型，当前配置为：{model}"
        )
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
        self.image_model = normalize_image_model(
            values.get("QWEN_IMAGE_MODEL") or values.get("IMAGE_MODEL"),
            DEFAULT_IMAGE_MODEL,
            "图片生成",
        )
        self.openai_image_api_key = str(
            values.get("OPENAI_IMAGE_API_KEY") or values.get("OPENAI_API_KEY") or ""
        ).strip()
        self.openai_image_base_url = str(
            values.get("OPENAI_IMAGE_BASE_URL")
            or values.get("OPENAI_BASE_URL")
            or "https://api.openai.com/v1"
        ).rstrip("/")
        self.openai_image_endpoint = str(
            values.get("OPENAI_IMAGE_ENDPOINT")
            or f"{self.openai_image_base_url}/images/edits"
        ).strip()
        self.image_total_timeout = max(
            1,
            int(values.get("QWEN_IMAGE_TOTAL_TIMEOUT") or DEFAULT_IMAGE_TOTAL_TIMEOUT),
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
            "你是图片生成系统的第一层意图推理器。必须结合最近对话、历史图片记录、"
            "当前请求已经过应用层图片检查；没有上传图片时才调用本模型判断文字意图，"
            "不要只靠固定关键词判断。"
            "用户可能用‘上一张’‘刚才生成的’‘上一版’，也可能用较模糊的指代表达；"
            "你需要根据完整会话判断其实际指向。只返回严格 JSON 对象，字段为 "
            '{"intent":"text|image|mixed","use_previous_image":false,"needs_tool":false,"is_clear":true,"reason":""}。'
            "intent=text 表示文本层请求；image 表示图片层请求；mixed 表示图片和配套文字。"
            "use_previous_image 表示用户实际要求修改、延续或重做会话中的历史图片；"
            "needs_tool 表示仅靠当前文本模型无法完成、需要进入工具链路三的请求；"
            "is_clear 表示用户是否给出了明确、可执行的任务目标。"
            "如果用户只是寒暄、表达不完整、只说‘做一个’或没有具体生图目标，is_clear 必须为 false；"
            "表达可以明确也可以模糊，由你结合上下文判断。"
            "没有上传图片时，只有明确基于历史图片修改且存在历史图片线索时，use_previous_image 才为 true。"
            "reason 必须简洁说明判断依据。"
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
        with generation_log_context(component="intent_model", model=self.text_model):
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
            use_previous_image=coerce_bool(payload.get("use_previous_image")),
            needs_tool=coerce_bool(payload.get("needs_tool")),
            is_clear=coerce_bool(payload.get("is_clear"), default=True),
            reason=str(payload.get("reason") or "").strip()[:500],
        )

    def generate_text(self, messages, intent="text"):
        options = {
            "text": (0.7, 2048),
            "image": (0.4, 1200),
            "mixed": (0.6, 2600),
        }
        temperature, max_tokens = options.get(intent, options["text"])
        with generation_log_context(component="text_model", model=self.text_model):
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
            "处理信息冲突时必须遵守以下优先级：系统硬约束高于用户本轮执行目标；"
            "用户执行目标决定修改内容；参考图决定用户未要求修改的视觉事实；"
            "业务参数只影响未明确指定的表现方式；知识库只补充空白信息；"
            "通用美化和负面约束优先级最低。不得让低优先级信息覆盖高优先级信息。"
            "image_prompt 供图片模型执行，应明确主体、保留项、修改项、场景、构图、光线、"
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
        with generation_log_context(
            component="image_prompt_planner",
            model=self.vl_model,
            reference_image_count=len(reference_images[:3]),
        ):
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
            raise RuntimeError("图片模型缺少生图提示词。")
        if self.image_model.lower().startswith("gpt-image"):
            with generation_log_context(
                component="image_model",
                model=self.image_model,
                provider="openai-compatible",
                reference_image_count=len((reference_images or [])[:16]),
                prompt_chars=len(normalized_prompt),
            ):
                with generation_stage(
                    "image_model.generate",
                    timeout_seconds=self.image_total_timeout,
                    max_attempts=MAX_IMAGE_ATTEMPTS,
                ):
                    return self._generate_openai_image(
                        normalized_prompt[:32000], reference_images or []
                    )
        if self.image_model.lower().startswith("wan"):
            if self.image_model.lower() != WANX_IMAGE_EDIT_MODEL:
                raise RuntimeError(
                    "当前 Wanx 模型仅支持 wanx2.1-imageedit；"
                    "wanx2.1-t2i-turbo 不支持当前参考图编辑链路。"
                )
            with generation_log_context(
                component="image_model",
                model=self.image_model,
                reference_image_count=len((reference_images or [])[:1]),
                prompt_chars=len(normalized_prompt[:800]),
            ):
                with generation_stage(
                    "image_model.generate",
                    timeout_seconds=self.image_total_timeout,
                    max_attempts=MAX_IMAGE_ATTEMPTS,
                ):
                    return self._generate_wanx_image_edit(
                        normalized_prompt[:800], reference_images or []
                    )
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
        with generation_log_context(
            component="image_model",
            model=self.image_model,
            reference_image_count=len(content) - 1,
            prompt_chars=len(normalized_prompt),
        ):
            with generation_stage(
                "image_model.generate",
                timeout_seconds=self.image_total_timeout,
                max_attempts=MAX_IMAGE_ATTEMPTS,
            ):
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

    def _generate_openai_image(self, prompt, reference_images):
        if not self.openai_image_api_key:
            raise RuntimeError(
                "已选择 GPT Image，但缺少 OPENAI_IMAGE_API_KEY 配置。"
            )
        if not reference_images:
            raise RuntimeError("GPT Image 当前编辑链必须提供参考图片。")

        boundary = f"----codex-{uuid.uuid4().hex}"
        body = bytearray()
        self._append_multipart_field(body, boundary, "model", self.image_model)
        self._append_multipart_field(body, boundary, "prompt", prompt)
        self._append_multipart_field(body, boundary, "n", "1")
        self._append_multipart_field(body, boundary, "input_fidelity", "high")
        for index, image_url in enumerate(reference_images[:16], start=1):
            image_bytes, content_type = self._read_image_reference(image_url)
            extension = content_type.split("/", 1)[-1] or "png"
            self._append_multipart_file(
                body,
                boundary,
                "image",
                f"reference-{index}.{extension}",
                content_type,
                image_bytes,
            )
        body.extend(f"--{boundary}--\r\n".encode("ascii"))
        req = request.Request(
            self.openai_image_endpoint,
            data=bytes(body),
            headers={
                "Authorization": f"Bearer {self.openai_image_api_key}",
                "Content-Type": f"multipart/form-data; boundary={boundary}",
            },
            method="POST",
        )
        try:
            with open_model_request(self.values, req, self.image_total_timeout) as response:
                response_body = response.read().decode("utf-8")
                payload = json.loads(response_body)
        except error.HTTPError as exc:
            response_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"GPT Image 接口请求失败：HTTP {exc.code} {response_body[:500]}"
            ) from exc
        except (error.URLError, TimeoutError, socket.timeout, ConnectionResetError) as exc:
            raise RuntimeError("GPT Image 接口响应超时或连接中断。") from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("GPT Image 接口返回了无法解析的 JSON 响应。") from exc

        image_url = self._extract_openai_image_url(payload)
        if not image_url:
            detail = str(payload.get("error", {}).get("message") or "响应中没有图片 URL").strip()
            raise RuntimeError(f"GPT Image 生成失败：{detail}")
        return image_url

    @staticmethod
    def _append_multipart_field(body, boundary, name, value):
        body.extend(
            (
                f"--{boundary}\r\n"
                f"Content-Disposition: form-data; name=\"{name}\"\r\n\r\n"
                f"{value}\r\n"
            ).encode("utf-8")
        )

    @staticmethod
    def _append_multipart_file(body, boundary, name, filename, content_type, content):
        body.extend(
            (
                f"--{boundary}\r\n"
                f"Content-Disposition: form-data; name=\"{name}\"; filename=\"{filename}\"\r\n"
                f"Content-Type: {content_type}\r\n\r\n"
            ).encode("utf-8")
        )
        body.extend(content)
        body.extend(b"\r\n")

    def _read_image_reference(self, image_url):
        image_url = str(image_url or "").strip()
        if image_url.startswith("data:"):
            header, encoded = image_url.split(",", 1)
            content_type = header[5:].split(";", 1)[0] or "image/png"
            try:
                return base64.b64decode(encoded), content_type
            except (ValueError, binascii.Error) as exc:
                raise RuntimeError("GPT Image 参考图片 data URL 无法解码。") from exc
        if image_url.startswith(("http://", "https://")):
            try:
                with open_model_request(
                    self.values,
                    request.Request(image_url, method="GET"),
                    IMAGE_REQUEST_TIMEOUT,
                ) as response:
                    return response.read(), response.headers.get_content_type() or "image/png"
            except (error.URLError, TimeoutError, socket.timeout) as exc:
                raise RuntimeError("GPT Image 参考图片下载失败。") from exc
        raise RuntimeError("GPT Image 参考图片必须是 HTTP URL 或 data URL。")

    @staticmethod
    def _extract_openai_image_url(payload):
        items = payload.get("data", []) if isinstance(payload, dict) else []
        for item in items:
            if not isinstance(item, dict):
                continue
            image_url = str(item.get("url") or "").strip()
            if image_url:
                return image_url
            encoded = str(item.get("b64_json") or "").strip()
            if encoded:
                return f"data:image/png;base64,{encoded}"
        return ""

    def _generate_wanx_image_edit(self, prompt, reference_images):
        if not reference_images:
            raise RuntimeError("Wanx 2.1 图片编辑必须提供参考图片。")
        endpoint = str(
            self.values.get("DASHSCOPE_WANX_IMAGE_ENDPOINT")
            or f"{self.api_base_url}/services/aigc/image2image/image-synthesis"
        ).strip()
        payload = {
            "model": self.image_model,
            "input": {
                "function": "description_edit",
                "prompt": prompt,
                "base_image_url": str(reference_images[0]).strip(),
            },
            "parameters": {"n": 1, "watermark": False},
        }
        deadline = get_generation_deadline()
        own_deadline = time.perf_counter() + self.image_total_timeout
        deadline = min(deadline, own_deadline) if deadline is not None else own_deadline
        with generation_log_context(deadline_monotonic=deadline):
            response = self._post_json(
                endpoint,
                payload,
                timeout=IMAGE_REQUEST_TIMEOUT,
                error_label="万相图片模型",
                attempts=MAX_IMAGE_ATTEMPTS,
            )
            task_id = str(response.get("output", {}).get("task_id") or "").strip()
            if not task_id:
                code = str(response.get("code") or "").strip()
                message = str(response.get("message") or "").strip()
                detail = f"{code} {message}".strip() or "创建任务响应中没有 task_id"
                raise RuntimeError(f"万相图片模型创建任务失败：{detail}")
            write_generation_event(
                "image_model.task_submitted",
                "completed",
                task_id=task_id,
                provider_request_id=str(response.get("request_id") or "").strip(),
            )
            return self._poll_wanx_task(task_id, deadline)

    def _poll_wanx_task(self, task_id, deadline):
        task_endpoint = f"{self.api_base_url}/tasks/{task_id}"
        while True:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                raise RuntimeError("万相图片模型任务等待超时。")
            poll_timeout = min(30.0, remaining)
            started_at = time.perf_counter()
            write_generation_event(
                "image_model.task_poll",
                "started",
                task_id=task_id,
                timeout_seconds=poll_timeout,
            )
            try:
                req = request.Request(
                    task_endpoint,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    method="GET",
                )
                with open_model_request(self.values, req, poll_timeout) as response:
                    body = response.read().decode("utf-8")
                    payload = json.loads(body)
                output = payload.get("output") if isinstance(payload, dict) else {}
                output = output if isinstance(output, dict) else {}
                status = str(output.get("task_status") or "UNKNOWN").upper()
                write_generation_event(
                    "image_model.task_poll",
                    "completed",
                    task_id=task_id,
                    task_status=status,
                    latency_ms=int((time.perf_counter() - started_at) * 1000),
                )
            except error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                write_generation_event(
                    "image_model.task_poll",
                    "failed",
                    task_id=task_id,
                    latency_ms=int((time.perf_counter() - started_at) * 1000),
                    http_status=exc.code,
                    error_message=body[:2000],
                )
                raise RuntimeError(f"万相图片模型查询任务失败：HTTP {exc.code}") from exc
            except (TimeoutError, socket.timeout, error.URLError, json.JSONDecodeError) as exc:
                write_generation_event(
                    "image_model.task_poll",
                    "failed",
                    task_id=task_id,
                    latency_ms=int((time.perf_counter() - started_at) * 1000),
                    error_category="timeout" if isinstance(exc, (TimeoutError, socket.timeout)) else "connection_error",
                    error_type=type(exc).__name__,
                    error_message=str(exc)[:2000],
                )
                raise RuntimeError("万相图片模型查询任务超时或连接中断。") from exc

            if status == "SUCCEEDED":
                for item in output.get("results", []):
                    image_url = str(item.get("url") or item.get("image_url") or "").strip()
                    if image_url:
                        return image_url
                raise RuntimeError("万相图片模型任务成功，但响应中没有图片 URL。")
            if status in {"FAILED", "CANCELED", "UNKNOWN"}:
                message = str(payload.get("message") or output.get("message") or "任务未成功完成").strip()
                raise RuntimeError(f"万相图片模型任务{status}：{message}")
            time.sleep(min(WANX_IMAGE_POLL_INTERVAL, max(0, deadline - time.perf_counter())))

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
        endpoint_name = self._safe_endpoint(endpoint)
        total_timeout = None
        if error_label in {"千问图片模型", "万相图片模型"}:
            total_timeout = self.image_total_timeout
        request_deadline = get_generation_deadline()
        if total_timeout is not None:
            request_deadline = min(
                request_deadline or (time.perf_counter() + total_timeout),
                time.perf_counter() + total_timeout,
            )
        idempotency_key = uuid.uuid4().hex
        for attempt in range(attempts):
            attempt_number = attempt + 1
            attempt_started_at = time.perf_counter()
            remaining = request_deadline - attempt_started_at if request_deadline else None
            if remaining is not None and remaining <= 0:
                last_error = RuntimeError(
                    f"{error_label}已超过图片生成总等待预算：{total_timeout}秒。"
                    if total_timeout is not None
                    else f"{error_label}已超过请求总 deadline。"
                )
                write_generation_event(
                    "model.deadline_exceeded",
                    "failed",
                    operation=error_label,
                    total_timeout_seconds=total_timeout,
                    idempotency_key=idempotency_key,
                )
                break
            effective_timeout = min(float(timeout), remaining) if remaining is not None else float(timeout)
            event_details = {
                "provider": "dashscope",
                "operation": error_label,
                "endpoint": endpoint_name,
                "attempt": attempt_number,
                "max_attempts": attempts,
                "timeout_seconds": effective_timeout,
                "configured_timeout_seconds": timeout,
                "total_timeout_seconds": total_timeout,
                "request_bytes": len(data),
                "proxy_enabled": bool(get_proxy_url(self.values)),
                "idempotency_key": idempotency_key,
            }
            write_generation_event("model.request", "started", **event_details)
            req = request.Request(
                endpoint,
                data=data,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json; charset=utf-8",
                    "Idempotency-Key": idempotency_key,
                    **(
                        {"X-DashScope-Async": "enable"}
                        if error_label == "万相图片模型"
                        else {}
                    ),
                },
                method="POST",
            )
            try:
                write_generation_event(
                    "network.request",
                    "started",
                    **event_details,
                    phase="connect_and_read",
                )
                with open_model_request(self.values, req, effective_timeout) as response:
                    response_body = response.read().decode("utf-8")
                    write_generation_event(
                        "network.request",
                        "completed",
                        **event_details,
                        phase="connect_and_read",
                        latency_ms=int((time.perf_counter() - attempt_started_at) * 1000),
                        http_status=getattr(response, "status", 200),
                    )
                    response_payload = json.loads(response_body)
                    write_generation_event(
                        "model.request",
                        "completed",
                        **event_details,
                        latency_ms=int((time.perf_counter() - attempt_started_at) * 1000),
                        http_status=getattr(response, "status", 200),
                        provider_request_id=self._response_request_id(response),
                        response_bytes=len(response_body.encode("utf-8")),
                    )
                    return response_payload
            except error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                last_error = RuntimeError(f"{error_label}接口请求失败：HTTP {exc.code} {body}")
                will_retry = self._can_retry(
                    exc.code in RETRYABLE_HTTP_CODES,
                    attempt,
                    attempts,
                    request_deadline,
                )
                write_generation_event(
                    "network.request",
                    "failed",
                    **event_details,
                    phase="connect_and_read",
                    latency_ms=int((time.perf_counter() - attempt_started_at) * 1000),
                    error_category=self._http_error_category(exc.code),
                    error_type=type(exc).__name__,
                    http_status=exc.code,
                    error_message=body[:2000],
                )
                write_generation_event(
                    "model.request",
                    "failed",
                    **event_details,
                    latency_ms=int((time.perf_counter() - attempt_started_at) * 1000),
                    error_category=self._http_error_category(exc.code),
                    error_type=type(exc).__name__,
                    http_status=exc.code,
                    provider_request_id=self._header_request_id(exc.headers),
                    error_message=body[:2000],
                    retry_scheduled=will_retry,
                )
                if not will_retry:
                    raise last_error from exc
            except (ConnectionResetError, TimeoutError, socket.timeout, error.URLError) as exc:
                last_error = RuntimeError(
                    f"{error_label}响应超时或连接中断。图片生成总等待预算：{total_timeout or timeout}秒。"
                )
                will_retry = self._can_retry(
                    True,
                    attempt,
                    attempts,
                    request_deadline,
                )
                write_generation_event(
                    "network.request",
                    "failed",
                    **event_details,
                    phase="connect_and_read",
                    latency_ms=int((time.perf_counter() - attempt_started_at) * 1000),
                    error_category=self._network_error_category(exc),
                    error_type=type(exc).__name__,
                    error_message=str(exc)[:2000],
                )
                write_generation_event(
                    "model.request",
                    "failed",
                    **event_details,
                    latency_ms=int((time.perf_counter() - attempt_started_at) * 1000),
                    error_category=self._network_error_category(exc),
                    error_type=type(exc).__name__,
                    error_message=str(exc)[:2000],
                    retry_scheduled=will_retry,
                )
                if not will_retry:
                    raise last_error from exc
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                last_error = RuntimeError(f"{error_label}返回了无法解析的 JSON 响应。")
                write_generation_event(
                    "model.request",
                    "failed",
                    **event_details,
                    latency_ms=int((time.perf_counter() - attempt_started_at) * 1000),
                    error_category="invalid_response",
                    error_type=type(exc).__name__,
                    error_message=str(exc)[:2000],
                    retry_scheduled=False,
                )
                raise last_error from exc
            write_generation_event(
                "model.retry_wait",
                "scheduled",
                operation=error_label,
                failed_attempt=attempt_number,
                next_attempt=attempt_number + 1,
                delay_seconds=2,
            )
            remaining_after_failure = (
                request_deadline - time.perf_counter() if request_deadline else None
            )
            if remaining_after_failure is not None and remaining_after_failure <= 2:
                break
            time.sleep(min(2, max(0, remaining_after_failure))) if remaining_after_failure is not None else time.sleep(2)
        raise last_error

    @staticmethod
    def _can_retry(retryable, attempt, attempts, deadline):
        if not retryable or attempt >= attempts - 1:
            return False
        return deadline is None or deadline - time.perf_counter() > 2

    @staticmethod
    def _safe_endpoint(endpoint):
        parsed = urlsplit(str(endpoint or ""))
        return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"

    @staticmethod
    def _header_request_id(headers):
        if not headers:
            return ""
        for name in ("x-request-id", "x-dashscope-request-id", "request-id"):
            value = headers.get(name)
            if value:
                return str(value).strip()[:200]
        return ""

    @classmethod
    def _response_request_id(cls, response):
        return cls._header_request_id(getattr(response, "headers", None))

    @staticmethod
    def _network_error_category(exc):
        reason = getattr(exc, "reason", None)
        if isinstance(exc, (TimeoutError, socket.timeout)) or isinstance(reason, (TimeoutError, socket.timeout)):
            return "timeout"
        if isinstance(reason, socket.gaierror):
            return "dns_error"
        if isinstance(exc, ssl.SSLError) or isinstance(reason, ssl.SSLError):
            return "tls_error"
        if isinstance(exc, ConnectionRefusedError) or isinstance(reason, ConnectionRefusedError):
            return "connection_refused"
        if isinstance(exc, ConnectionResetError) or isinstance(reason, ConnectionResetError):
            return "connection_reset"
        return "connection_error"

    @staticmethod
    def _http_error_category(status_code):
        if status_code == 429:
            return "rate_limited"
        if status_code in RETRYABLE_HTTP_CODES:
            return "provider_server_error"
        if 400 <= status_code < 500:
            return "request_rejected"
        return "http_error"

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
