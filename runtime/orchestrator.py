# -*- coding: utf-8 -*-

import base64

from infra.logger import generation_log_context, generation_stage, write_generation_event
from llm.prompt_builder import build_message
from llm.qwen_client import QwenGateway, normalize_image_model
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
from services.visual_reference_loader import (
    load_reference_image,
    prepare_visual_planner_image,
)


MAX_QWEN_REFERENCE_IMAGES = 3
MAX_EDIT_REVISION = 2
REUPLOAD_REQUIRED_MESSAGE = "请重新上传参考图片并仔细规划提示词。"


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
        if state.image_model:
            self.gateway.image_model = normalize_image_model(
                state.image_model,
                self.gateway.image_model,
                "图片生成",
            )
        with generation_log_context(task_id=state.task_id, conversation_id=state.conversation_id):
            with generation_stage(
                "intent.classify",
                has_uploaded_image=bool(self._uploaded_images(state)),
                has_history_image=self._has_history_image(state),
                force_image=state.force_image,
            ):
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
                state.route = ROUTE_TEXT
                selected_runner = self._run_text
            elif self._uploaded_images(state):
                state.route = ROUTE_CHAIN_TWO
                selected_runner = self._run_chain_two
            elif decision.use_previous_image and self._has_history_image(state):
                state.route = ROUTE_CHAIN_ONE
                selected_runner = self._run_chain_one
            else:
                state.route = ROUTE_CHAIN_THREE
                selected_runner = self._run_chain_three

            write_generation_event(
                "route.select",
                "completed",
                route=state.route,
                intent=decision.intent,
                use_previous_image=decision.use_previous_image,
                reason=decision.reason,
            )
            with generation_log_context(route=state.route):
                return selected_runner(state)

    def _run_text(self, state):
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
        selected = self._select_history_image(state.visual_history)
        edit_session = dict(state.edit_session or {})
        revision = self._edit_revision(edit_session)
        root_image_url = str(edit_session.get("root_image_url") or "").strip()
        latest_result_url = str(
            edit_session.get("latest_result_url") or selected.get("image_url") or ""
        ).strip()
        if revision >= MAX_EDIT_REVISION:
            return self._requires_reupload(state, revision, "edit_revision_limit")
        if revision != 1 or not root_image_url:
            return self._requires_reupload(state, revision, "missing_root_reference")

        with generation_stage(
            "reference_image.load",
            reference_source="root_and_previous_result",
            edit_revision=revision,
        ):
            root_reference = load_reference_image(self.values, root_image_url)
            previous_result = load_reference_image(self.values, latest_result_url)
        if not root_reference:
            return self._requires_reupload(state, revision, "root_reference_unavailable")

        root_data_url = self._reference_to_data_url(root_reference)
        previous_data_url = ""
        if previous_result:
            previous_data_url = self._reference_to_data_url(previous_result)
        # The previous result informs the planner; the root image remains the only
        # image sent to the final generator so unrelated edits are not compounded.
        planning_references = (
            [previous_data_url, root_data_url]
            if previous_data_url
            else [root_data_url]
        )
        planning_references = self._prepare_planner_references(planning_references)
        context = self._build_correction_context(selected, revision, bool(previous_result))
        with generation_stage("image_prompt.plan", reference_source="history"):
            plan = self.gateway.plan_image_task(
                user_input=state.user_input,
                reference_images=planning_references,
                context=context,
                intent=state.intent,
            )
        state.record(
            "image_prompt_planned",
            reference_source="previous_result_as_optimization_parameter",
            generation_base="root_reference_only",
            edit_revision=revision + 1,
        )
        with generation_stage(
            "image.generate",
            reference_source="root_only",
            edit_revision=revision + 1,
        ):
            generated_url = self.gateway.generate_image(
                prompt=plan.image_prompt,
                reference_images=[root_data_url],
            )
        with generation_stage("image.cache"):
            image_url = self._cache_generated_image(generated_url)
        state.record("image_generated", model=self.gateway.image_model, storage="redis")
        return self._result(
            state,
            reply_text=plan.reply_text,
            status="completed",
            image_url=image_url,
            image_prompt=plan.image_prompt,
            copywriting=plan.copywriting,
            root_reference_url=root_image_url,
            edit_revision=revision + 1,
        )

    def _run_chain_two(self, state):
        knowledge_query = self._knowledge_query(state)
        with generation_stage("knowledge.retrieve", query_chars=len(knowledge_query)):
            knowledge = self.knowledge_base.search(knowledge_query, limit=1)
        state.knowledge_status = knowledge.status
        state.record(
            "knowledge_searched",
            status=knowledge.status,
            match_count=len(knowledge.matches),
            **self._knowledge_score_trace(knowledge),
        )

        reference_images = self._uploaded_images(state)[:MAX_QWEN_REFERENCE_IMAGES]
        write_generation_event(
            "knowledge.result",
            "completed",
            knowledge_status=knowledge.status,
            match_count=len(knowledge.matches),
            **self._knowledge_score_trace(knowledge),
        )
        direct_prompt = str(state.user_input or "").strip()
        uploaded_text = str(getattr(state.uploaded_content, "text", "") or "").strip()
        if uploaded_text:
            direct_prompt = (
                f"{direct_prompt}\n\n"
                "上传资料文字补充（仅作辅助，不得覆盖用户请求和原图）：\n"
                f"{uploaded_text[:12000]}"
            )
        knowledge_context = str(getattr(knowledge, "context_text", "") or "").strip()
        if knowledge_context:
            direct_prompt = (
                f"{direct_prompt}\n\n"
                "可参考的知识库补充（仅在不与用户请求和原图冲突时使用）：\n"
                f"{knowledge_context[:6000]}"
            )
        with generation_stage("image.generate", reference_source="upload"):
            generated_url = self.gateway.generate_image(
                prompt=direct_prompt,
                reference_images=reference_images,
            )
        with generation_stage("image.cache"):
            image_url = self._cache_generated_image(generated_url)
        state.record("image_generated", model=self.gateway.image_model, storage="redis")
        return self._result(
            state,
            reply_text="已根据本轮要求生成图片。",
            status="completed",
            image_url=image_url,
            image_prompt=direct_prompt,
        )

    def _run_chain_three(self, state):
        knowledge_query = self._knowledge_query(state)
        with generation_stage("knowledge.retrieve", query_chars=len(knowledge_query)):
            knowledge = self.knowledge_base.search(knowledge_query, limit=1)
        state.knowledge_status = knowledge.status
        state.record(
            "knowledge_searched",
            status=knowledge.status,
            match_count=len(knowledge.matches),
            **self._knowledge_score_trace(knowledge),
        )
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
        root_reference_url="",
        edit_revision=0,
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
            root_reference_url=root_reference_url,
            edit_revision=edit_revision,
        )

    def _requires_reupload(self, state, revision, reason):
        state.record(
            "generation_refused",
            reason=reason,
            edit_revision=revision,
            max_edit_revision=MAX_EDIT_REVISION,
        )
        write_generation_event(
            "image.generate",
            "refused",
            reason=reason,
            edit_revision=revision,
            max_edit_revision=MAX_EDIT_REVISION,
        )
        return self._result(
            state,
            reply_text=REUPLOAD_REQUIRED_MESSAGE,
            status="requires_reupload",
            edit_revision=revision,
        )

    def _cache_generated_image(self, image_url):
        reference = load_reference_image(self.values, image_url)
        if not reference:
            raise RuntimeError("图片模型已生成图片，但无法下载到 Redis 图片缓存。")
        cached = self.image_cache.put_image(reference.content, reference.content_type)
        return cached.public_url

    def _prepare_planner_references(self, reference_images):
        prepared = []
        for index, image_url in enumerate(reference_images or [], start=1):
            planner_url, details = prepare_visual_planner_image(self.values, image_url)
            write_generation_event(
                "reference_image.prepare",
                "completed",
                purpose="visual_planner",
                reference_index=index,
                **details,
            )
            prepared.append(planner_url)
        return prepared

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

    @classmethod
    def _build_correction_context(cls, selected, revision, has_previous_result):
        previous_rule = (
            "参考图1是第一次生成结果，仅用于识别本轮需要优化的方向和范围；"
            "它不是最终生图底图，也不能把其中未经用户确认的变化继续带入结果。\n"
            "参考图2是用户最初上传的根参考图；最终生图模型只能以根参考图为唯一基准。"
        if has_previous_result
        else "第一次生成结果当前不可读取，只能依据用户文字说明和根参考图执行本轮修改。"
        )
        return (
            "【连续编辑规则｜最高优先级】\n"
            "这是当前编辑链允许的第2次且最后一次图片生成。\n"
            f"{previous_rule}\n"
            "必须把用户本轮明确指出的不满意之处转化为具体、可执行的优化方向，"
            "并限制修改范围；用户未明确要求修改的区域必须按根参考图保持，不得顺带重构。\n\n"
            f"【第一次生成结果信息｜仅用于定位优化方向和范围】\n{cls._build_history_context(selected)}\n\n"
            f"【当前修订号】\n输入版本：{revision}；本轮成功后版本：{revision + 1}。"
        )

    @staticmethod
    def _edit_revision(edit_session):
        try:
            return max(0, int(edit_session.get("revision") or 0))
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _uploaded_text_context(state):
        text = str(getattr(state.uploaded_content, "text", "") or "").strip()
        if not text:
            return ""
        return f"上传资料提取文字（不作为图片像素内容处理）：\n{text[:12000]}"

    @classmethod
    def _build_chain_two_context(cls, state, knowledge):
        user_input = str(state.user_input or "").strip()
        poster_context = dict(state.poster_context or {})
        campaign = str(poster_context.get("campaign") or "").strip()
        if campaign == user_input:
            campaign = ""

        business_context = (
            f"海报类型：{str(poster_context.get('poster_type') or '未指定').strip()}\n"
            f"活动主题：{campaign or '未单独指定'}\n"
            f"目标受众：{str(poster_context.get('target_audience') or '未指定').strip()}"
        )
        knowledge_context = str(getattr(knowledge, "context_text", "") or "").strip()
        sections = [
            (
                "【执行目标｜最高优先级】\n"
                f"{user_input}"
            ),
            (
                "【编辑边界】\n"
                "只修改用户在执行目标中明确要求修改的内容。\n"
                "用户未要求修改的主体、商品结构、Logo、文字、材质、构图、比例和其他细节保持不变。"
            ),
            (
                "【原图使用规则】\n"
                "本轮上传图片是视觉事实来源。\n"
                "必须直接观察图片确定主体、背景、构图和细节，不得凭文字猜测图片内容。"
            ),
            (
                "【业务参数｜低于执行目标和原图事实】\n"
                f"{business_context}\n"
                "业务参数只影响用户未明确指定的表现方式，不得扩大编辑范围。"
            ),
        ]
        uploaded_text = cls._uploaded_text_context(state)
        if uploaded_text:
            sections.append(
                "【上传资料文字｜辅助信息】\n"
                f"{uploaded_text}\n"
                "该信息用于补充背景资料，不得覆盖执行目标或原图事实。"
            )
        sections.append(
            "【知识库补充｜低于用户请求和原图事实】\n"
            f"{knowledge_context or '本次没有可用的知识库匹配。'}\n"
            "知识库只补充用户没有明确指定的细节，不得覆盖执行目标、编辑边界或原图事实。"
        )
        sections.append(
            "【输出要求】\n"
            "生成一个完整、可直接交给图片编辑模型执行的 image_prompt。\n"
            "明确写出保留项、修改项、主体、背景、色彩、构图、光线、画质和负面约束。"
        )
        return "\n\n".join(sections)

    @staticmethod
    def _knowledge_query(state):
        candidates = [
            state.user_input,
            str(getattr(state.uploaded_content, "text", "") or ""),
            state.poster_context.get("campaign", ""),
            state.poster_context.get("target_audience", ""),
        ]
        parts = []
        seen = set()
        for candidate in candidates:
            part = str(candidate or "").strip()
            identity = " ".join(part.split()).casefold()
            if not part or identity in seen:
                continue
            seen.add(identity)
            parts.append(part)
        return "\n".join(parts)[:12000]

    def _knowledge_score_trace(self, knowledge):
        if not knowledge.matches:
            return {
                "rerank_threshold": float(
                    getattr(self.knowledge_base, "rerank_min_score", 0.0)
                )
            }
        top_match = knowledge.matches[0]
        return {
            "top_vector_score": round(float(top_match.get("vector_score") or 0.0), 6),
            "top_rerank_score": round(float(top_match.get("rerank_score") or 0.0), 6),
            "rerank_threshold": float(
                getattr(self.knowledge_base, "rerank_min_score", 0.0)
            ),
        }
