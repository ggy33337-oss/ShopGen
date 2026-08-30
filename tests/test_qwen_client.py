# -*- coding: utf-8 -*-

import unittest

from llm.qwen_client import QwenGateway


class CapturingQwenGateway(QwenGateway):
    def __init__(self):
        super().__init__({"DASHSCOPE_API_KEY": "test-key"})
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
    def test_reference_generation_uses_qwen_native_multimodal_payload(self):
        gateway = CapturingQwenGateway()

        image_url = gateway.generate_image("生成商品图", ["data:image/png;base64,aW1hZ2U="])

        payload = gateway.request_data["payload"]
        content = payload["input"]["messages"][0]["content"]
        self.assertEqual("qwen-image-3.0-pro", payload["model"])
        self.assertTrue(gateway.request_data["endpoint"].endswith("/multimodal-generation/generation"))
        self.assertEqual("data:image/png;base64,aW1hZ2U=", content[0]["image"])
        self.assertEqual("生成商品图", content[1]["text"])
        self.assertNotIn("size", payload["parameters"])
        self.assertEqual("https://example.com/result.png", image_url)

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
