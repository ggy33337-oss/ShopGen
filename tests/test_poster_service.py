# -*- coding: utf-8 -*-

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from poster.models.request import PosterGenerationRequest
from services.poster_service import PosterService


class PosterServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_generate_persists_root_image_with_storage_parameter_name(self):
        result = SimpleNamespace(
            image_url="/api/images/result",
            reply_text="海报已生成",
            image_prompt="",
            route="chain_2_uploaded_image",
            status="completed",
            knowledge_status="not_used",
            task_id="task-1",
            actions=(),
            copywriting=SimpleNamespace(
                headline="金黄绵羊 · 独特魅力",
                subheadline="经典黑白线条与明亮色彩的完美融合",
                cta="立即体验",
            ),
        )

        class FakeOrchestrator:
            def run(self, state):
                return result

        service = PosterService({}, orchestrator=FakeOrchestrator())
        service._parse_uploaded_content = lambda request: None
        service._cache_root_reference = lambda uploaded_content: "/api/images/root"
        request = PosterGenerationRequest(user_text="生成一张海报", conversation_id="test")

        with patch("services.poster_service.get_history_messages", new=AsyncMock(return_value=[])), \
             patch("services.poster_service.get_recent_visual_history", new=AsyncMock(return_value=[])), \
             patch("services.poster_service.append_visual_history", new=AsyncMock()), \
             patch("services.poster_service.append_messages", new=AsyncMock()) as append_messages, \
             patch("services.poster_service.save_visual_edit_session", new=AsyncMock()) as save_session, \
             patch("services.poster_service.write_generation_event"), \
             patch("services.poster_service.write_log"):
            await service.generate(request)

        save_session.assert_awaited_once_with(
            "test",
            root_image_url="/api/images/root",
            latest_result_url="/api/images/result",
            revision=1,
        )
        saved_messages = append_messages.await_args.args[1]
        self.assertEqual(
            "主标题：金黄绵羊 · 独特魅力\n副标题：经典黑白线条与明亮色彩的完美融合\n行动语：立即体验",
            saved_messages[1]["content"],
        )


if __name__ == "__main__":
    unittest.main()
