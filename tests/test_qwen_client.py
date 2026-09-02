# -*- coding: utf-8 -*-

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


class QwenImagePayloadTests(unittest.TestCase):
    def test_default_text_model_is_qwen38_max(self):
        gateway = QwenGateway({"DASHSCOPE_API_KEY": "test-key"})

        self.assertEqual("qwen3.8-max", gateway.text_model)

    def test_default_image_model_is_wanx_image_edit(self):
        gateway = QwenGateway({"DASHSCOPE_API_KEY": "test-key"})

        self.assertEqual("wanx2.1-imageedit", gateway.image_model)

    def test_wanx_image_edit_uses_async_task_payload(self):
        gateway = CapturingQwenGateway()
        gateway._poll_wanx_task = lambda task_id, deadline: "https://example.com/result.png"
        gateway._post_json = lambda endpoint, payload, timeout, error_label, attempts=1: (
            gateway.request_data.update(
                endpoint=endpoint,
                payload=payload,
                timeout=timeout,
                error_label=error_label,
                attempts=attempts,
            )
            or {"output": {"task_id": "task-1"}, "request_id": "request-1"}
        )

        image_url = gateway.generate_image("把背景改成红色", ["data:image/png;base64,aW1hZ2U="])

        payload = gateway.request_data["payload"]
        self.assertEqual("wanx2.1-imageedit", payload["model"])
        self.assertTrue(gateway.request_data["endpoint"].endswith("/image2image/image-synthesis"))
        self.assertEqual("description_edit", payload["input"]["function"])
        self.assertEqual("把背景改成红色", payload["input"]["prompt"])
        self.assertEqual("data:image/png;base64,aW1hZ2U=", payload["input"]["base_image_url"])
        self.assertEqual("https://example.com/result.png", image_url)

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
