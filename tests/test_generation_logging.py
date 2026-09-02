# -*- coding: utf-8 -*-

import json
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from infra.logger import generation_log_context, write_generation_event, write_log
from llm.qwen_client import QwenGateway


class GenerationLoggerTests(unittest.TestCase):
    def test_events_are_disabled_without_a_request_context(self):
        with patch("infra.logger.write_log") as mocked_write_log:
            write_generation_event("image.generate", "started")

        mocked_write_log.assert_not_called()

    def test_context_and_sensitive_values_are_safely_written(self):
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "generation.jsonl"
            with generation_log_context(
                log_path=log_path,
                task_id="task-123",
                conversation_id="会话-一",
            ):
                write_generation_event(
                    "image.generate",
                    "failed",
                    authorization="Bearer secret-token",
                    image="data:image/png;base64,aW1hZ2U=",
                )

            record = json.loads(log_path.read_text(encoding="utf-8"))
            self.assertEqual("task-123", record["task_id"])
            self.assertEqual("会话-一", record["conversation_id"])
            self.assertEqual("[REDACTED]", record["authorization"])
            self.assertEqual("[base64 image omitted]", record["image"])

    def test_write_log_does_not_mutate_callers_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = {"status": "completed"}
            write_log(payload, Path(directory) / "llm.jsonl")

            self.assertNotIn("created_at", payload)


class QwenAttemptLoggingTests(unittest.TestCase):
    def test_each_timeout_attempt_and_final_failure_are_logged(self):
        gateway = QwenGateway(
            {
                "DASHSCOPE_API_KEY": "test-key",
                "QWEN_IMAGE_MODEL": "qwen-image-3.0-pro",
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "generation.jsonl"
            with generation_log_context(log_path=log_path, task_id="timeout-task"):
                with patch("llm.qwen_client.open_model_request", side_effect=socket.timeout("timed out")):
                    with patch("llm.qwen_client.time.sleep"):
                        with self.assertRaisesRegex(RuntimeError, "响应超时或连接中断"):
                            gateway.generate_image(
                                "把背景换成蓝色",
                                ["data:image/png;base64,aW1hZ2U="],
                            )

            records = [
                json.loads(line)
                for line in log_path.read_text(encoding="utf-8").splitlines()
            ]
            failed_attempts = [
                record
                for record in records
                if record["stage"] == "model.request" and record["status"] == "failed"
            ]
            self.assertEqual([1, 2], [record["attempt"] for record in failed_attempts])
            self.assertEqual(["timeout", "timeout"], [record["error_category"] for record in failed_attempts])
            self.assertTrue(failed_attempts[0]["retry_scheduled"])
            self.assertFalse(failed_attempts[1]["retry_scheduled"])
            self.assertTrue(
                any(
                    record["stage"] == "image_model.generate" and record["status"] == "failed"
                    for record in records
                )
            )
            raw_log = log_path.read_text(encoding="utf-8")
            self.assertNotIn("aW1hZ2U=", raw_log)
            self.assertNotIn("test-key", raw_log)

    def test_missing_image_url_is_logged_as_generation_failure(self):
        gateway = QwenGateway(
            {
                "DASHSCOPE_API_KEY": "test-key",
                "QWEN_IMAGE_MODEL": "qwen-image-3.0-pro",
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "generation.jsonl"
            with generation_log_context(log_path=log_path, task_id="invalid-output-task"):
                with patch.object(gateway, "_post_json", return_value={"output": {"choices": []}}):
                    with self.assertRaisesRegex(RuntimeError, "响应中没有图片 URL"):
                        gateway.generate_image("生成商品图")

            records = [
                json.loads(line)
                for line in log_path.read_text(encoding="utf-8").splitlines()
            ]
            failed = [
                record
                for record in records
                if record["stage"] == "image_model.generate" and record["status"] == "failed"
            ]
            self.assertEqual(1, len(failed))
            self.assertIn("响应中没有图片 URL", failed[0]["error_message"])

    def test_image_timeout_respects_request_deadline_without_second_attempt(self):
        gateway = QwenGateway({"DASHSCOPE_API_KEY": "test-key"})
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "generation.jsonl"
            with generation_log_context(
                log_path=log_path,
                task_id="deadline-task",
                deadline_seconds=0.01,
            ):
                with patch(
                    "llm.qwen_client.open_model_request",
                    side_effect=socket.timeout("timed out"),
                ):
                    with self.assertRaisesRegex(RuntimeError, "总等待预算"):
                        gateway.generate_image(
                            "把背景换成蓝色",
                            ["data:image/png;base64,aW1hZ2U="],
                        )

            records = [
                json.loads(line)
                for line in log_path.read_text(encoding="utf-8").splitlines()
            ]
            attempts = [
                record
                for record in records
                if record["stage"] == "model.request"
                and record["status"] == "failed"
            ]
            self.assertEqual([1], [record["attempt"] for record in attempts])
            self.assertFalse(any(record["stage"] == "model.retry_wait" for record in records))
            self.assertTrue(any(record["stage"] == "network.request" for record in records))


if __name__ == "__main__":
    unittest.main()
