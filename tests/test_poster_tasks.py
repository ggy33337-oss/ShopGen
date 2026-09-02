# -*- coding: utf-8 -*-

import asyncio
import unittest

from poster.models.request import PosterGenerationRequest
from services.poster_tasks import PosterTaskManager


class FakePosterService:
    def __init__(self, values):
        self.values = values

    async def generate(self, request, task_id=None, **kwargs):
        await asyncio.sleep(0)
        return {"task_id": task_id, "user_text": request.user_text}


class PosterTaskManagerTests(unittest.IsolatedAsyncioTestCase):
    async def test_submit_returns_immediately_and_tracks_completion(self):
        manager = PosterTaskManager(service_factory=FakePosterService)
        request = PosterGenerationRequest(user_text="生成图片")

        task = await manager.submit(request, {}, "task-1")
        self.assertEqual("queued", task.status)

        for _ in range(20):
            current = await manager.get("task-1")
            if current.status == "completed":
                break
            await asyncio.sleep(0.01)

        self.assertEqual("completed", current.status)
        self.assertEqual("task-1", current.result["task_id"])

    async def test_failed_task_keeps_error_for_status_polling(self):
        class FailingService(FakePosterService):
            async def generate(self, request, task_id=None, **kwargs):
                raise RuntimeError("图片模型失败")

        manager = PosterTaskManager(service_factory=FailingService)
        await manager.submit(PosterGenerationRequest(user_text="生成图片"), {}, "task-2")

        for _ in range(20):
            current = await manager.get("task-2")
            if current.status == "failed":
                break
            await asyncio.sleep(0.01)

        self.assertEqual("failed", current.status)
        self.assertEqual("图片模型失败", current.error)


if __name__ == "__main__":
    unittest.main()
