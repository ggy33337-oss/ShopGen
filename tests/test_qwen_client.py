# -*- coding: utf-8 -*-

import json
import unittest
from unittest.mock import patch

from llm.qwen_client import QwenGateway


class CapturingQwenGateway(QwenGateway):
    def __init__(self, values=None):
        super().__init__(values or {"DASHSCOPE_API_KEY": "test-key"})
        self.request_data = {}

    def _post_json(self, endpoint, payload, timeout, error_label, attempts=1):
        self.request_data = {
            "endpoint": endpoint,
            "payload": payload,
            "timeout": timeout,
            "error_label": error_label,
            "attempts": attempts,
        }
        return {
            "output": {
                "choices": [
                    {"message": {"content": [{"image": "https://example.com/result.png"}]}}
                ]
            }
        }


class FailingEndpointGateway(QwenGateway):
    def __init__(self):
        super().__init__({
            "DASHSCOPE_API_KEY": "test-key",
            "DASHSCOPE_COMPAT_BASE_URL": "https://private.example/v1",
        })
        self.endpoints = []

    def _post_json(self, endpoint, payload, timeout, error_label, attempts=1):
        self.endpoints.append(endpoint)
        if len(self.endpoints) == 1:
            raise RuntimeError(f"{error_label}响应超时或连接中断。")
        return {"choices": [{"message": {"content": "ok"}}]}


class QwenImagePayloadTests(unittest.TestCase):
    def test_chat_completion_falls_back_from_unavailable_custom_endpoint(self):
        gateway = FailingEndpointGateway()

        response = gateway.chat_completion(
            [{"role": "user", "content": "你好"}],
            model="qwen3.8-max",
            temperature=0,
            max_tokens=20,
        )

        self.assertEqual("ok", response["choices"][0]["message"]["content"])
        self.assertEqual("https://private.example/v1/chat/completions", gateway.endpoints[0])
        self.assertEqual(
            "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
            gateway.endpoints[1],
        )

    def test_intent_classifier_parses_layer_tool_and_clarity_fields(self):
        gateway = QwenGateway({"DASHSCOPE_API_KEY": "test-key"})
        gateway.chat_completion = lambda *args, **kwargs: {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"intent":"image","use_previous_image":true,'
                            '"needs_tool":"true","is_clear":"false","reason":"needs detail"}'
                        )
                    }
                }
            ]
        }

        decision = gateway.classify_request(
            user_input="做一个",
            history_messages=[],
            visual_history=[],
        )

        self.assertEqual("image", decision.intent)
        self.assertTrue(decision.use_previous_image)
        self.assertTrue(decision.needs_tool)
        self.assertFalse(decision.is_clear)

    def test_default_text_model_is_qwen38_max(self):
        gateway = QwenGateway({"DASHSCOPE_API_KEY": "test-key"})

        self.assertEqual("qwen3.8-max", gateway.text_model)

    def test_default_image_model_is_qwen_image(self):
        gateway = QwenGateway({"DASHSCOPE_API_KEY": "test-key"})

        self.assertEqual("qwen-image-3.0", gateway.image_model)

    def test_reference_generation_uses_qwen_native_multimodal_payload(self):
        gateway = CapturingQwenGateway({
            "DASHSCOPE_API_KEY": "test-key",
            "QWEN_IMAGE_MODEL": "qwen-image-3.0-pro",
        })

        image_url = gateway.generate_image("生成商品图", ["data:image/png;base64,aW1hZ2U="])

        payload = gateway.request_data["payload"]
        content = payload["input"]["messages"][0]["content"]
        self.assertEqual("qwen-image-3.0-pro", payload["model"])
        self.assertTrue(gateway.request_data["endpoint"].endswith("/multimodal-generation/generation"))
        self.assertEqual("data:image/png;base64,aW1hZ2U=", content[0]["image"])
        self.assertEqual("生成商品图", content[1]["text"])
        self.assertNotIn("size", payload["parameters"])
        self.assertEqual("https://example.com/result.png", image_url)

    def test_gpt_image_edit_uses_openai_compatible_multipart_payload(self):
        gateway = QwenGateway(
            {
                "DASHSCOPE_API_KEY": "test-key",
                "QWEN_IMAGE_MODEL": "gpt-image-2",
                "OPENAI_IMAGE_API_KEY": "openai-test-key",
                "OPENAI_IMAGE_BASE_URL": "https://api.example.com/v1",
            }
        )

        class FakeResponse:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return b'{"data":[{"b64_json":"aW1hZ2U="}]}'

        with patch("llm.qwen_client.open_model_request", return_value=FakeResponse()) as open_request:
            image_url = gateway.generate_image(
                "把背景改成红色", ["data:image/png;base64,aW1hZ2U="]
            )

        self.assertEqual("data:image/png;base64,aW1hZ2U=", image_url)
        req = open_request.call_args.args[1]
        self.assertEqual("https://api.example.com/v1/images/edits", req.full_url)
        self.assertIn(b"name=\"model\"", req.data)
        self.assertIn(b"gpt-image-2", req.data)
        self.assertIn(b"name=\"image\"", req.data)

    def test_gpt_image_generation_uses_generations_endpoint_without_reference(self):
        gateway = QwenGateway(
            {
                "DASHSCOPE_API_KEY": "test-key",
                "QWEN_IMAGE_MODEL": "gpt-image-2",
                "OPENAI_IMAGE_API_KEY": "openai-test-key",
                "OPENAI_IMAGE_BASE_URL": "https://api.example.com/v1",
            }
        )

        class FakeResponse:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return b'{"data":[{"b64_json":"aW1hZ2U="}]}'

        with patch("llm.qwen_client.open_model_request", return_value=FakeResponse()) as open_request:
            image_url = gateway.generate_image("生成一个滑板", [])

        self.assertEqual("data:image/png;base64,aW1hZ2U=", image_url)
        req = open_request.call_args.args[1]
        self.assertEqual("https://api.example.com/v1/images/generations", req.full_url)
        self.assertEqual(
            {"model": "gpt-image-2", "prompt": "生成一个滑板", "n": 1},
            json.loads(req.data.decode("utf-8")),
        )
        self.assertEqual("Bearer openai-test-key", req.get_header("Authorization"))

    def test_non_qwen_model_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "必须使用千问模型"):
            QwenGateway(
                {
                    "DASHSCOPE_API_KEY": "test-key",
                    "QWEN_TEXT_MODEL": "other-model",
                }
            )

if __name__ == "__main__":
    unittest.main()
