# -*- coding: utf-8 -*-

import asyncio
import time
from dataclasses import dataclass
from typing import Any

from poster.models.request import PosterGenerationRequest
from services.poster_service import PosterService


TASK_RETENTION_SECONDS = 60 * 60


@dataclass
class PosterTask:
    task_id: str
    conversation_id: str
    status: str = "queued"
    result: Any = None
    error: str = ""
    created_at: float = 0.0
    updated_at: float = 0.0


class PosterTaskManager:
    """Small in-process task registry for long-running poster generations."""

    def __init__(self, service_factory=None):
        self._tasks: dict[str, PosterTask] = {}
        self._running: set[asyncio.Task] = set()
        self._lock = asyncio.Lock()
        self._service_factory = service_factory or (lambda values: PosterService(values))

    async def submit(
        self,
        request: PosterGenerationRequest,
        values: dict[str, str],
        task_id: str,
        deadline_monotonic=None,
        deadline_at=None,
    ):
        now = time.time()
        task = PosterTask(
            task_id=task_id,
            conversation_id=request.conversation_id,
            created_at=now,
            updated_at=now,
        )
        async with self._lock:
            self._purge_expired(now)
            self._tasks[task_id] = task
        background = asyncio.create_task(
            self._run(task, request, values, deadline_monotonic, deadline_at)
        )
        self._running.add(background)
        background.add_done_callback(self._running.discard)
        return task

    async def get(self, task_id: str):
        async with self._lock:
            self._purge_expired(time.time())
            return self._tasks.get(task_id)

    async def _run(
        self,
        task: PosterTask,
        request: PosterGenerationRequest,
        values: dict[str, str],
        deadline_monotonic=None,
        deadline_at=None,
    ):
        task.status = "running"
        task.updated_at = time.time()
        try:
            task.result = await self._service_factory(values).generate(
                request,
                task_id=task.task_id,
                deadline_monotonic=deadline_monotonic,
                deadline_at=deadline_at,
            )
            task.status = "completed"
        except asyncio.CancelledError:
            task.status = "cancelled"
            task.error = "任务已取消。"
            raise
        except Exception as exc:
            task.status = "failed"
            task.error = str(exc)[:2000]
        finally:
            task.updated_at = time.time()

    def _purge_expired(self, now):
        expired = [
            task_id
            for task_id, task in self._tasks.items()
            if task.status in {"completed", "failed", "cancelled"}
            and now - task.updated_at > TASK_RETENTION_SECONDS
        ]
        for task_id in expired:
            self._tasks.pop(task_id, None)


_MANAGER = PosterTaskManager()


def get_poster_task_manager():
    return _MANAGER
