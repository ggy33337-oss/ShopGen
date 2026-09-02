# -*- coding: utf-8 -*-

import unittest
from types import SimpleNamespace

from knowledge.models import KnowledgeContext
from runtime.models import (
    Copywriting,
    ImagePromptPlan,
    ImageTaskState,
    IntentDecision,
    ROUTE_CHAIN_ONE,
    ROUTE_CHAIN_THREE,
    ROUTE_CHAIN_TWO,
    ROUTE_TEXT,
)
from runtime.orchestrator import ImageGenerationOrchestrator
from runtime.orchestrator import CLARIFICATION_REQUIRED_MESSAGE, REUPLOAD_REQUIRED_MESSAGE


REFERENCE_DATA_URL = "data:image/png;base64,aW1hZ2U="
HISTORY_DATA_URL = "data:image/png;base64,aGlzdG9yeQ=="
UPLOAD_DATA_URL = "data:image/png;base64,dXBsb2FkZWQ="


class FakeGateway:
    image_model = "qwen-image-3.0-pro"

    def __init__(self, decision):
        self.decision = decision
        self.generated_references = []
        self.generated_prompts = []
        self.planned_requests = []

    def classify_request(self, **kwargs):
        del kwargs
        return self.decision

    def generate_text(self, messages, intent="text"):
        del messages, intent
        return "千问文字回复"

    def plan_image_task(self, **kwargs):
        self.planned_requests.append(kwargs)
        return ImagePromptPlan(
            reply_text="图片已生成",
            image_prompt="保留商品主体，仅调整背景",
            copywriting=Copywriting(headline="新品上市"),
        )

    def generate_image(self, prompt, reference_images=None):
        self.generated_prompts.append(prompt)
        self.generated_references = list(reference_images or [])
        return REFERENCE_DATA_URL


class FakeImageCache:
    def __init__(self):
        self.images = []

    def put_image(self, content, content_type):
        self.images.append((content, content_type))
        return SimpleNamespace(public_url="/api/images/0123456789abcdef0123456789abcdef")


class FakeKnowledgeBase:
    def __init__(self):
        self.queries = []

    def search(self, query, limit=3, category=""):
        self.queries.append((query, limit, category))
        return KnowledgeContext(
            query=query,
            status="matched",
            context_text="[知识库匹配 1]\n标题：蓝色运动鞋\n内容：主色为海洋蓝。",
            matches=({"title": "蓝色运动鞋", "score": 0.96},),
        )


class OrchestratorRoutingTests(unittest.TestCase):
    def test_text_request_stays_outside_image_chains(self):
        gateway = FakeGateway(IntentDecision(intent="text"))
        state = ImageTaskState(user_input="写一句文案", conversation_id="test")

        result = ImageGenerationOrchestrator(
            {}, gateway=gateway, image_cache=FakeImageCache()
        ).run(state)

        self.assertEqual(ROUTE_TEXT, result.route)
        self.assertEqual("千问文字回复", result.reply_text)
        self.assertEqual("", result.image_url)

    def test_request_image_model_overrides_gateway_default(self):
        gateway = FakeGateway(IntentDecision(intent="text"))
        state = ImageTaskState(
            user_input="写一句文案",
            conversation_id="test",
            image_model="wanx2.1-imageedit",
        )

        ImageGenerationOrchestrator(
            {}, gateway=gateway, image_cache=FakeImageCache()
        ).run(state)

        self.assertEqual("wanx2.1-imageedit", gateway.image_model)

    def test_uploaded_image_has_priority_over_history_model_decision(self):
        gateway = FakeGateway(IntentDecision(intent="image", use_previous_image=True))
        state = ImageTaskState(
            user_input="把上一张图的背景换成绿色",
            conversation_id="test",
            visual_history=[{"image_url": HISTORY_DATA_URL, "user_input": "生成商品图"}],
            uploaded_content=SimpleNamespace(data_urls=(UPLOAD_DATA_URL,), text=""),
        )

        cache = FakeImageCache()
        result = ImageGenerationOrchestrator(
            {},
            gateway=gateway,
            knowledge_base=FakeKnowledgeBase(),
            image_cache=cache,
        ).run(state)

        self.assertEqual(ROUTE_CHAIN_TWO, result.route)
        self.assertEqual([UPLOAD_DATA_URL], gateway.generated_references)
        self.assertEqual("/api/images/0123456789abcdef0123456789abcdef", result.image_url)
        self.assertEqual(1, len(cache.images))

    def test_uploaded_image_bypasses_intent_model_and_goes_to_chain_two(self):
        class NoClassifierGateway(FakeGateway):
            def classify_request(self, **kwargs):
                raise AssertionError("uploaded image should bypass intent classification")

        gateway = NoClassifierGateway(IntentDecision(intent="text"))
        state = ImageTaskState(
            user_input="参考这张图生成商品图",
            conversation_id="test",
            uploaded_content=SimpleNamespace(data_urls=(UPLOAD_DATA_URL,), text=""),
        )

        result = ImageGenerationOrchestrator(
            {}, gateway=gateway, knowledge_base=FakeKnowledgeBase(), image_cache=FakeImageCache()
        ).run(state)

        self.assertEqual(ROUTE_CHAIN_TWO, result.route)
        self.assertEqual("image", state.intent)

    def test_model_can_route_ambiguous_request_to_history_without_upload(self):
        gateway = FakeGateway(IntentDecision(intent="image", use_previous_image=True))
        state = ImageTaskState(
            user_input="刚才那个再高级一点",
            conversation_id="test",
            visual_history=[{"image_url": HISTORY_DATA_URL, "user_input": "生成商品图"}],
            edit_session={
                "root_image_url": REFERENCE_DATA_URL,
                "latest_result_url": HISTORY_DATA_URL,
                "revision": 1,
            },
        )

        result = ImageGenerationOrchestrator(
            {}, gateway=gateway, image_cache=FakeImageCache()
        ).run(state)

        self.assertEqual(ROUTE_CHAIN_ONE, result.route)
        self.assertEqual([REFERENCE_DATA_URL], gateway.generated_references)
        self.assertEqual(
            [HISTORY_DATA_URL, REFERENCE_DATA_URL],
            gateway.planned_requests[0]["reference_images"],
        )
        self.assertIn(
            "参考图1是第一次生成结果，仅用于识别本轮需要优化的方向和范围",
            gateway.planned_requests[0]["context"],
        )
        self.assertIn(
            "最终生图模型只能以根参考图为唯一基准",
            gateway.planned_requests[0]["context"],
        )
        self.assertEqual(2, result.edit_revision)
        self.assertEqual(REFERENCE_DATA_URL, result.root_reference_url)

    def test_third_edit_requires_reupload_without_planning_or_generation(self):
        gateway = FakeGateway(IntentDecision(intent="image", use_previous_image=True))
        state = ImageTaskState(
            user_input="再把主体颜色调亮一点",
            conversation_id="test",
            visual_history=[{"image_url": HISTORY_DATA_URL, "user_input": "第二次生成"}],
            edit_session={
                "root_image_url": REFERENCE_DATA_URL,
                "latest_result_url": HISTORY_DATA_URL,
                "revision": 2,
            },
        )

        result = ImageGenerationOrchestrator(
            {}, gateway=gateway, image_cache=FakeImageCache()
        ).run(state)

        self.assertEqual(ROUTE_CHAIN_ONE, result.route)
        self.assertEqual("requires_reupload", result.status)
        self.assertEqual(REUPLOAD_REQUIRED_MESSAGE, result.reply_text)
        self.assertEqual([], gateway.planned_requests)
        self.assertEqual([], gateway.generated_references)
        self.assertEqual("generation_refused", result.actions[-1]["action"])

    def test_history_edit_without_root_reference_requires_reupload(self):
        gateway = FakeGateway(IntentDecision(intent="image", use_previous_image=True))
        state = ImageTaskState(
            user_input="继续修改上一张图",
            conversation_id="test",
            visual_history=[{"image_url": HISTORY_DATA_URL}],
        )

        result = ImageGenerationOrchestrator(
            {}, gateway=gateway, image_cache=FakeImageCache()
        ).run(state)

        self.assertEqual("requires_reupload", result.status)
        self.assertEqual(REUPLOAD_REQUIRED_MESSAGE, result.reply_text)
        self.assertEqual([], gateway.planned_requests)
        self.assertEqual([], gateway.generated_references)

    def test_uploaded_image_uses_chain_two_and_retrieved_knowledge(self):
        gateway = FakeGateway(IntentDecision(intent="mixed", use_previous_image=False))
        state = ImageTaskState(
            user_input="参考这张图生成促销海报",
            conversation_id="test",
            uploaded_content=SimpleNamespace(data_urls=(REFERENCE_DATA_URL,), text="商品说明"),
            force_image=True,
        )

        cache = FakeImageCache()
        knowledge_base = FakeKnowledgeBase()
        result = ImageGenerationOrchestrator(
            {},
            gateway=gateway,
            knowledge_base=knowledge_base,
            image_cache=cache,
        ).run(state)

        self.assertEqual(ROUTE_CHAIN_TWO, result.route)
        self.assertEqual("matched", result.knowledge_status)
        self.assertEqual(1, len(knowledge_base.queries))
        self.assertEqual("参考这张图生成促销海报", result.image_prompt.split("\n", 1)[0])
        self.assertEqual([], gateway.planned_requests)
        self.assertEqual(1, len(gateway.generated_prompts))
        self.assertEqual(1, len(cache.images))
        self.assertIn("商品说明", result.image_prompt)
        self.assertIn("主色为海洋蓝", result.image_prompt)

    def test_no_image_uses_chain_three_placeholder_without_generation(self):
        gateway = FakeGateway(IntentDecision(intent="image", use_previous_image=False))
        state = ImageTaskState(user_input="生成一张新品海报", conversation_id="test")

        result = ImageGenerationOrchestrator(
            {},
            gateway=gateway,
            knowledge_base=FakeKnowledgeBase(),
            image_cache=FakeImageCache(),
        ).run(state)

        self.assertEqual(ROUTE_CHAIN_THREE, result.route)
        self.assertEqual("placeholder", result.status)
        self.assertEqual("matched", result.knowledge_status)
        self.assertEqual("", result.image_url)
        self.assertEqual([], gateway.generated_references)

    def test_unclear_text_request_returns_clarification_without_generation(self):
        gateway = FakeGateway(IntentDecision(intent="text", is_clear=False))
        state = ImageTaskState(user_input="做一个", conversation_id="test")

        result = ImageGenerationOrchestrator(
            {}, gateway=gateway, knowledge_base=FakeKnowledgeBase(), image_cache=FakeImageCache()
        ).run(state)

        self.assertEqual(ROUTE_TEXT, result.route)
        self.assertEqual("clarification_required", result.status)
        self.assertEqual(CLARIFICATION_REQUIRED_MESSAGE, result.reply_text)

    def test_tool_request_uses_chain_three(self):
        gateway = FakeGateway(IntentDecision(intent="image", needs_tool=True))
        state = ImageTaskState(user_input="调用外部工具完成任务", conversation_id="test")

        result = ImageGenerationOrchestrator(
            {}, gateway=gateway, knowledge_base=FakeKnowledgeBase(), image_cache=FakeImageCache()
        ).run(state)

        self.assertEqual(ROUTE_CHAIN_THREE, result.route)
        self.assertEqual("placeholder", result.status)


if __name__ == "__main__":
    unittest.main()
