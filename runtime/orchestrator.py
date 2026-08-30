# -*- coding: utf-8 -*-

import base64
import json

from llm.prompt_builder import build_message
from llm.qwen_client import QwenGateway
from knowledge.qdrant_knowledge_base import QdrantKnowledgeBase
from infra.redis_image_cache import RedisImageCache
from runtime.models import (
    ImageTaskState,
    OrchestrationResult,
    ROUTE_CHAIN_ONE,
    ROUTE_CHAIN_THREE,
    ROUTE_CHAIN_TWO,
    ROUTE_TEXT,
)
from runtime.placeholders import PlaceholderTextToImageChain
from services.visual_reference_loader import load_reference_image


MAX_QWEN_REFERENCE_IMAGES = 3


class ImageGenerationOrchestrator:
    """图片生成架构中唯一负责链路选择与执行的流程控制者。"""

    def __init__(
        self,
        values,
        gateway=None,
        knowledge_base=None,
        chain_three=None,
        image_cache=None,
    ):
        self.values = values
        self.gateway = gateway or QwenGateway(values)
        self.knowledge_base = knowledge_base or QdrantKnowledgeBase(values)
        self.chain_three = chain_three or PlaceholderTextToImageChain()
        self.image_cache = image_cache or RedisImageCache(values)

    def run(self, state: ImageTaskState) -> OrchestrationResult:
        decision = self.gateway.classify_request(
            user_input=state.user_input,
            history_messages=state.history_messages,
            visual_history=state.visual_history,
            has_uploaded_image=bool(self._uploaded_images(state)),
            force_image=state.force_image,
        )
        state.intent = decision.intent
        state.record(
            "intent_decided",
            intent=decision.intent,
            use_previous_image=decision.use_previous_image,
            reason=decision.reason,
        )

        if decision.intent == "text" and not state.force_image:
            return self._run_text(state)

        if decision.use_previous_image and self._has_history_image(state):
            return self._run_chain_one(state)

        if self._uploaded_images(state):
            return self._run_chain_two(state)

        return self._run_chain_three(state)

    def _run_text(self, state):
        state.route = ROUTE_TEXT
        messages = build_message(
            state.user_input,
            "text",
            state.history_messages,
            state.visual_history,
        )
        reply_text = self.gateway.generate_text(messages, intent="text")
        state.record("text_generated")
        return self._result(state, reply_text=reply_text, status="completed")

    def _run_chain_one(self, state):
        state.route = ROUTE_CHAIN_ONE
        selected = self._select_history_image(state.visual_history)
        reference = load_reference_image(self.values, selected.get("image_url", ""))
        if not reference:
            raise RuntimeError("链路一无法读取会话中的历史图片，请重新上传原图后再试。")

        reference_data_url = self._reference_to_data_url(reference)
        context = self._build_history_context(selected)
        plan = self.gateway.plan_image_task(
            user_input=state.user_input,
            reference_images=[reference_data_url],
            context=context,
            intent=state.intent,
        )
        state.record("image_prompt_planned", reference_source="history")
        generated_url = self.gateway.generate_image(
            prompt=plan.image_prompt,
            reference_images=[reference_data_url],
        )
        image_url = self._cache_generated_image(generated_url)
        state.record("image_generated", model=self.gateway.image_model, storage="redis")
        return self._result(
            state,
            reply_text=plan.reply_text,
            status="completed",
            image_url=image_url,
            image_prompt=plan.image_prompt,
            copywriting=plan.copywriting,
        )

    def _run_chain_two(self, state):
        state.route = ROUTE_CHAIN_TWO
        knowledge_query = self._knowledge_query(state)
        knowledge = self.knowledge_base.search(knowledge_query, limit=1)
        state.knowledge_status = knowledge.status
        state.record("knowledge_searched", status=knowledge.status, match_count=len(knowledge.matches))

        reference_images = self._uploaded_images(state)[:MAX_QWEN_REFERENCE_IMAGES]
        context_parts = [
            "本轮上传图片是视觉参考，需保留主体、商品结构和关键识别特征。",
            self._uploaded_text_context(state),
            self._poster_context(state),
            knowledge.context_text,
        ]
        plan = self.gateway.plan_image_task(
            user_input=state.user_input,
            reference_images=reference_images,
            context="\n".join(part for part in context_parts if part),
            intent=state.intent,
        )
        state.record("image_prompt_planned", reference_source="upload")
        generated_url = self.gateway.generate_image(
            prompt=plan.image_prompt,
            reference_images=reference_images,
        )
        image_url = self._cache_generated_image(generated_url)
        state.record("image_generated", model=self.gateway.image_model, storage="redis")
        return self._result(
            state,
            reply_text=plan.reply_text,
            status="completed",
            image_url=image_url,
            image_prompt=plan.image_prompt,
            copywriting=plan.copywriting,
        )

    def _run_chain_three(self, state):
        state.route = ROUTE_CHAIN_THREE
        knowledge = self.knowledge_base.search(self._knowledge_query(state), limit=1)
        state.knowledge_status = knowledge.status
        state.record("knowledge_searched", status=knowledge.status, match_count=len(knowledge.matches))
        placeholder = self.chain_three.run(state.user_input, knowledge)
        state.record("chain_three_placeholder_returned")
        return self._result(
            state,
            reply_text=placeholder.reply_text,
            status=placeholder.status,
        )

    def _result(
        self,
        state,
        reply_text,
        status,
        image_url="",
        image_prompt="",
        copywriting=None,
    ):
        from runtime.models import Copywriting

        return OrchestrationResult(
            reply_text=reply_text,
            route=state.route,
            status=status,
            image_url=image_url,
            image_prompt=image_prompt,
            knowledge_status=state.knowledge_status,
            copywriting=copywriting or Copywriting(),
            task_id=state.task_id,
            actions=tuple(state.actions),
        )

    def _cache_generated_image(self, image_url):
        reference = load_reference_image(self.values, image_url)
        if not reference:
            raise RuntimeError("千问图片已生成，但无法下载到 Redis 图片缓存。")
        cached = self.image_cache.put_image(reference.content, reference.content_type)
        return cached.public_url

    @staticmethod
    def _has_history_image(state):
        return any(str(item.get("image_url") or "").strip() for item in state.visual_history)

    @staticmethod
    def _select_history_image(visual_history):
        candidates = [item for item in visual_history if str(item.get("image_url") or "").strip()]
        if not candidates:
            return {}
        return candidates[-1]

    @staticmethod
    def _uploaded_images(state):
        if not state.uploaded_content:
            return []
        return [str(item) for item in state.uploaded_content.data_urls if str(item).strip()]

    @staticmethod
    def _reference_to_data_url(reference):
        encoded = base64.b64encode(reference.content).decode("ascii")
        return f"data:{reference.content_type};base64,{encoded}"

    @staticmethod
    def _build_history_context(selected):
        return (
            f"历史用户需求：{selected.get('user_input', '')}\n"
            f"历史助手说明：{selected.get('assistant_text', '')}\n"
            f"历史生图提示词：{selected.get('image_prompt', '')[:2000]}"
        )

    @staticmethod
    def _uploaded_text_context(state):
        text = str(getattr(state.uploaded_content, "text", "") or "").strip()
        if not text:
            return ""
        return f"上传资料提取文字（不作为图片像素内容处理）：\n{text[:12000]}"

    @staticmethod
    def _poster_context(state):
        if not state.poster_context:
            return ""
        return "海报业务参数：" + json.dumps(state.poster_context, ensure_ascii=False)

    @staticmethod
    def _knowledge_query(state):
        parts = [state.user_input, str(getattr(state.uploaded_content, "text", "") or "")]
        parts.extend(str(value or "") for value in state.poster_context.values())
        return "\n".join(part.strip() for part in parts if part and part.strip())[:12000]
