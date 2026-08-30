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


REFERENCE_DATA_URL = "data:image/png;base64,aW1hZ2U="


class FakeGateway:
    image_model = "qwen-image-3.0-pro"

    def __init__(self, decision):
        self.decision = decision
        self.generated_references = []

    def classify_request(self, **kwargs):
        del kwargs
        return self.decision

    def generate_text(self, messages, intent="text"):
        del messages, intent
        return "千问文字回复"

    def plan_image_task(self, **kwargs):
        del kwargs
        return ImagePromptPlan(
            reply_text="图片已生成",
            image_prompt="保留商品主体，仅调整背景",
            copywriting=Copywriting(headline="新品上市"),
        )

    def generate_image(self, prompt, reference_images=None):
        del prompt
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

    def test_history_image_has_priority_over_uploaded_image(self):
        gateway = FakeGateway(IntentDecision(intent="image", use_previous_image=True))
        state = ImageTaskState(
            user_input="把上一张图的背景换成绿色",
            conversation_id="test",
            visual_history=[{"image_url": REFERENCE_DATA_URL, "user_input": "生成商品图"}],
            uploaded_content=SimpleNamespace(data_urls=(REFERENCE_DATA_URL,), text=""),
        )

        cache = FakeImageCache()
        result = ImageGenerationOrchestrator({}, gateway=gateway, image_cache=cache).run(state)

        self.assertEqual(ROUTE_CHAIN_ONE, result.route)
        self.assertEqual(1, len(gateway.generated_references))
        self.assertEqual("/api/images/0123456789abcdef0123456789abcdef", result.image_url)
        self.assertEqual(1, len(cache.images))

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
        self.assertEqual("新品上市", result.copywriting.headline)
        self.assertEqual(1, len(cache.images))

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


if __name__ == "__main__":
    unittest.main()
