# -*- coding: utf-8 -*-

"""Search-grounded image generation for the no-reference image route.

This is a small host-side adaptation of Gen-Searcher's runtime pattern.  The
agent model is deliberately supplied by the existing Qwen gateway; no
Gen-Searcher checkpoint is loaded here.  Search tools only collect evidence
and references.  The existing image gateway remains responsible for the
actual image generation.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import mimetypes
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib import error, request
from urllib.parse import unquote, urlparse

from infra.logger import generation_stage, write_generation_event
from knowledge.models import KnowledgeContext
from services.visual_reference_loader import prepare_visual_planner_image

try:
    from PIL import Image
except ImportError:  # pragma: no cover - Pillow is an application dependency
    Image = None


DEFAULT_MAX_TOOL_CALLS = 8
DEFAULT_IMAGE_RESULTS = 6
DEFAULT_IMAGE_REFERENCE_LIMIT = 5
DEFAULT_SEARCH_TIMEOUT = 20
DEFAULT_REFERENCE_REVIEW_TIMEOUT = 45
DEFAULT_IMAGE_DIR = "data/gen_searcher_images"
MAX_PLANNER_REFERENCE_IMAGES = 5
MAX_REVIEW_REFERENCE_IMAGES = 6
MAX_VISUAL_REQUIREMENTS = 4
MIN_REFERENCE_IMAGE_SIDE = 200


@dataclass(frozen=True)
class ChainThreeResult:
    reply_text: str
    status: str = "completed"
    generated_url: str = ""
    image_prompt: str = ""
    copywriting: dict[str, str] = field(default_factory=dict)
    references: tuple[dict[str, str], ...] = ()
    tool_calls: int = 0
    error: str = ""


class GenSearcherImageChain:
    """Agentic search -> grounded prompt -> existing image generator."""

    def __init__(self, values=None, gateway=None):
        self.values = dict(values or {})
        self.gateway = gateway
        self.max_tool_calls = _bounded_int(
            self.values.get("GEN_SEARCHER_MAX_TOOL_CALLS"),
            DEFAULT_MAX_TOOL_CALLS,
            1,
            8,
        )
        self.image_result_limit = _bounded_int(
            self.values.get("GEN_SEARCHER_IMAGE_RESULTS"),
            DEFAULT_IMAGE_RESULTS,
            1,
            10,
        )
        self.reference_limit = _bounded_int(
            self.values.get("GEN_SEARCHER_REFERENCE_LIMIT"),
            DEFAULT_IMAGE_REFERENCE_LIMIT,
            1,
            5,
        )
        self.timeout = _bounded_int(
            self.values.get("GEN_SEARCHER_SEARCH_TIMEOUT"),
            DEFAULT_SEARCH_TIMEOUT,
            3,
            60,
        )
        self.review_timeout = _bounded_int(
            self.values.get("GEN_SEARCHER_REFERENCE_REVIEW_TIMEOUT"),
            DEFAULT_REFERENCE_REVIEW_TIMEOUT,
            10,
            120,
        )

    def run(
        self,
        user_input: str,
        knowledge: KnowledgeContext,
        gateway=None,
        task_id: str = "",
    ) -> ChainThreeResult:
        gateway = gateway or self.gateway
        try:
            source_stat = Path(__file__).stat()
            write_generation_event(
                "runtime.code",
                "loaded",
                module="runtime.gen_searcher_chain",
                source_file=str(Path(__file__).resolve()),
                source_mtime_ns=source_stat.st_mtime_ns,
                source_size=source_stat.st_size,
            )
        except OSError:
            pass
        # Generic subjects need no real-world grounding; use the image model
        # directly and avoid unnecessary search/download/reference review.
        if _is_generic_image_task(user_input):
            prompt = _clean_prompt(str(user_input or "").strip()) or "生成一张高质量图片"
            write_generation_event("chain_three.direct_generate", "started", reference_count=0)
            try:
                generated_url = gateway.generate_image(prompt=prompt, reference_images=[])
                if not str(generated_url or "").strip():
                    raise RuntimeError("图片模型未返回图片 URL。")
            except Exception as exc:
                return ChainThreeResult(reply_text="图片生成失败，请稍后重试。", status="failed", image_prompt=prompt, error=str(exc).strip()[:1000])
            write_generation_event("chain_three.direct_generate", "completed", reference_count=0)
            return ChainThreeResult(reply_text="已根据您的描述直接生成图片，请参考附件。", status="completed", generated_url=str(generated_url).strip(), image_prompt=prompt, references=(), tool_calls=0)

        if gateway is None or not callable(getattr(gateway, "chat_completion", None)):
            return self._fallback(
                user_input,
                knowledge,
                "当前推理模型不可用，已保留链路三的任务信息。",
            )

        trajectory_id = str(task_id or uuid.uuid4().hex)
        entity_hint = _extract_entity_hint(user_input)
        image_map: dict[str, dict[str, str]] = {}
        evidence: list[dict[str, str]] = []
        transcript: list[dict[str, str]] = []
        tool_calls = 0
        image_search_done = False
        final = None
        last_error = ""

        while tool_calls < self.max_tool_calls and (
            image_search_done or tool_calls < self.max_tool_calls - 1
        ):
            try:
                with generation_stage(
                    "chain_three.planner_step",
                    tool_call_index=tool_calls + 1,
                    candidate_count=len(image_map),
                ):
                    action = self._next_action(
                        gateway,
                        user_input,
                        knowledge,
                        transcript,
                        image_map,
                        evidence,
                        image_search_done,
                    )
            except Exception as exc:
                last_error = str(exc).strip()[:1000]
                break
            if action.get("type") in {"final", "final_answer"}:
                final = _normalize_final_action(action)
                write_generation_event(
                    "chain_three.selection_decision",
                    "completed",
                    requested_reference_ids=_requested_reference_ids(final),
                    selection_reason=str(final.get("selection_reason") or "")[:1000],
                    visual_requirements=_normalize_visual_requirements(
                        final.get("visual_requirements")
                    ),
                    candidate_images=_reference_log_items(image_map.values()),
                )
                break
            if action.get("type") != "tool_call":
                last_error = "模型返回了无法识别的链路三动作。"
                transcript.append({"role": "tool", "content": last_error})
                continue

            tool = str(action.get("tool") or "").strip().lower()
            arguments = action.get("arguments")
            if not isinstance(arguments, dict):
                arguments = {}
            with generation_stage(
                f"chain_three.tool.{tool or 'unknown'}",
                tool=tool,
                tool_call_index=tool_calls + 1,
            ):
                result_text, new_refs, new_evidence = self._execute_tool(
                    tool,
                    arguments,
                    image_map,
                    trajectory_id,
                    entity_hint=entity_hint,
                )
            tool_calls += 1
            image_search_done = image_search_done or tool == "image_search"
            if new_refs:
                image_map.update(new_refs)
            evidence.extend(new_evidence)
            write_generation_event(
                "chain_three.tool_call",
                "completed",
                tool=tool,
                arguments=_safe_tool_arguments(arguments),
                result_chars=len(result_text),
                new_reference_count=len(new_refs),
            )
            transcript.append(
                {
                    "role": "tool",
                    "content": self._limit_text(
                        f"工具 {tool} 返回：\n{result_text}", 12000
                    ),
                }
            )

        # Gen-Searcher explicitly requires visual retrieval.  If the model
        # stopped before image_search, make one bounded corrective call using
        # the user's subject rather than silently producing an ungrounded image.
        if not image_search_done and tool_calls < self.max_tool_calls:
            with generation_stage(
                "chain_three.tool.image_search",
                tool="image_search",
                tool_call_index=tool_calls + 1,
                forced=True,
            ):
                result_text, new_refs, new_evidence = self._execute_tool(
                    "image_search",
                    {"query": _search_query(user_input)},
                    image_map,
                    trajectory_id,
                    entity_hint=entity_hint,
                )
            tool_calls += 1
            image_search_done = True
            image_map.update(new_refs)
            evidence.extend(new_evidence)
            write_generation_event(
                "chain_three.tool_call",
                "completed",
                tool="image_search",
                arguments={"query": _search_query(user_input)},
                result_chars=len(result_text),
                new_reference_count=len(new_refs),
                forced=True,
            )
            transcript.append({"role": "tool", "content": result_text})

        if final is None:
            final = self._fallback_action(user_input, knowledge, image_map)

        # The planner may declare visual layers that are still missing.  Run
        # only those model-requested searches before selecting references; a
        # logo is therefore retrieved for a poster when needed, but is never
        # imposed on unrelated image tasks.
        requirements = _normalize_visual_requirements(
            final.get("visual_requirements")
            or final.get("visual_plan")
            or final.get("reference_plan")
        )
        for requirement in requirements:
            role = requirement["visual_role"]
            if not requirement["required"] or _has_visual_role(image_map, role):
                continue
            if tool_calls >= self.max_tool_calls:
                break
            query = requirement["query"] or _search_query(user_input)
            with generation_stage(
                "chain_three.layer_search",
                tool="image_search",
                visual_role=role,
                search_layer=requirement["search_layer"],
                tool_call_index=tool_calls + 1,
            ):
                result_text, new_refs, new_evidence = self._execute_tool(
                    "image_search",
                    {
                        "query": query,
                        "visual_role": role,
                        "search_layer": requirement["search_layer"],
                        "selection_goal": requirement["reason"],
                    },
                    image_map,
                    trajectory_id,
                    entity_hint=entity_hint,
                )
            tool_calls += 1
            image_search_done = True
            image_map.update(new_refs)
            evidence.extend(new_evidence)
            write_generation_event(
                "chain_three.layer_search",
                "completed",
                visual_role=role,
                search_layer=requirement["search_layer"],
                query=query,
                reason=requirement["reason"],
                result_chars=len(result_text),
                new_reference_count=len(new_refs),
                required=True,
            )

        prompt = _clean_prompt(final.get("gen_prompt") or final.get("image_prompt"))
        if not prompt:
            prompt = _fallback_prompt(user_input, knowledge, image_map)
        selected = self._select_references(final, image_map, requirements)
        with generation_stage(
            "chain_three.reference_review.total",
            candidate_count=len(image_map),
            planned_count=len(selected),
        ):
            reviewed = self._review_references(gateway, user_input, image_map, selected)
        if reviewed:
            reviewed_selected = self._select_references(
                {"reference_images": reviewed.get("reference_images", [])},
                image_map,
                requirements,
            )
            if reviewed_selected:
                selected = reviewed_selected
        selected = selected[: self._generation_reference_limit(gateway)]
        prompt = _ensure_reference_ordinals(prompt, len(selected))

        write_generation_event(
            "chain_three.references",
            "completed",
            references=_reference_log_items(selected),
            reference_count=len(selected),
            candidate_count=len(image_map),
            visual_requirements=requirements,
            selection_audit=_selection_audit(image_map, selected),
        )
        write_generation_event(
            "chain_three.prompt",
            "completed",
            prompt=prompt,
            prompt_chars=len(prompt),
            reference_count=len(selected),
        )

        try:
            reference_data_urls = [
                _path_to_data_url(item.get("local_path", "")) or str(item.get("url") or "").strip()
                for item in selected
                if item.get("local_path") or item.get("url")
            ]
            reference_data_urls = [item for item in reference_data_urls if item]
            with generation_stage(
                "chain_three.image_generate",
                reference_count=len(reference_data_urls),
            ):
                generated_url = gateway.generate_image(
                    prompt=prompt,
                    reference_images=reference_data_urls,
                )
            if not str(generated_url or "").strip():
                raise RuntimeError("图片模型未返回图片 URL。")
        except Exception as exc:
            last_error = str(exc).strip()[:1000]
            return ChainThreeResult(
                reply_text="检索增强图片生成失败，请稍后重试。",
                status="failed",
                image_prompt=prompt,
                references=tuple(selected),
                tool_calls=tool_calls,
                error=last_error,
            )

        copywriting = final.get("copywriting")
        if not isinstance(copywriting, dict):
            copywriting = {}
        copywriting = {
            key: str(copywriting.get(key) or "").strip()[:400]
            for key in ("headline", "subheadline", "cta")
        }
        return ChainThreeResult(
            reply_text=str(final.get("reply_text") or "已完成图片检索增强并生成图片。").strip(),
            status="completed",
            generated_url=str(generated_url or "").strip(),
            image_prompt=prompt,
            copywriting=copywriting,
            references=tuple(selected),
            tool_calls=tool_calls,
            error=last_error,
        )

    @staticmethod
    def _generation_reference_limit(gateway):
        model = str(getattr(gateway, "image_model", "") or "").lower()
        return 3 if model.startswith("qwen") else 16

    def _next_action(
        self,
        gateway,
        user_input,
        knowledge,
        transcript,
        image_map,
        evidence,
        image_search_done,
    ):
        system = (
            "你是图片检索增强规划器，工作方式参考 Gen-Searcher。\n"
            "你必须根据任务自主决定是否调用 search、image_search、browse，"
            "多轮收集真实证据和视觉参考，最后返回 grounded prompt。\n"
            "每轮只返回一个严格 JSON 对象，不要 Markdown，不要 XML。\n"
            "工具动作格式：{\"type\":\"tool_call\",\"tool\":\"search|image_search|browse\"," 
            "\"arguments\":{...},\"reason\":\"...\"}\n"
            "最终动作格式：{\"type\":\"final\",\"gen_prompt\":\"...\"," 
            "\"reference_images\":[{\"img_id\":\"IMG_001\",\"note\":\"...\"}],"
            "\"visual_requirements\":[{\"visual_role\":\"identity|representative_scene|activity|detail|style\","
            "\"required\":true,\"query\":\"...\",\"search_layer\":1,\"reason\":\"...\"}],"
            "\"copywriting\":{\"headline\":\"\",\"subheadline\":\"\",\"cta\":\"\"},"
            "\"reply_text\":\"...\",\"selection_reason\":\"...\"}\n"
            "仅当任务确实需要真实世界视觉依据时调用 image_search；通用主体可直接返回 final 且不调用工具。避免重复查询。gen_prompt 不得包含 URL 或 IMG 编号，"
            "需要引用图片时使用“第一张参考图/第二张参考图”等序数。\n"
            "先判断生成图片真正需要哪些视觉事实，再按视觉层级检索；不要套用固定模板，也不要默认必须检索 Logo。"
            "对真实学校、品牌、地点等主体，通常可考虑 identity（Logo/校徽/标识）、"
            "representative_scene（最能代表主体的场景）、activity（活动/人群）、detail（局部细节）或 style（风格）层，"
            "但只有确实有助于本次生成的层才标记 required=true。每次 image_search 都必须在 arguments 中给出"
            "visual_role、search_layer 和 selection_goal，并为不同层使用不同查询；同一层不要重复搜索相同主体场景。\n"
            "关键词规划规则：查询要短、具体、面向当前层；优先官方名称、官方域名和最新结果；"
            "中文结果为空或不相关时改用英文主体名、官方名称或更短查询；不要重复完全相同的查询。\n"
            "参考图选择规则：相同 visual_role 只能保留一张，优先当前检索结果中来源权威、时间更新、画质高、"
            "分辨率高、细节和信息量丰富、最能代表主体的图片；不同层只有在视觉事实确实不同的时候才同时保留。"
            "最终 reference_images 选择 1 到 5 张，并在 selection_reason 中说明每一层的入选、替换和排除依据；"
            "visual_requirements 必须列出你判断过的关键层，便于后端补齐缺失层。"
        )
        candidate_images = self._planner_candidates(image_map)
        tool_state = {
            "request": str(user_input or ""),
            "knowledge": self._limit_text(getattr(knowledge, "context_text", ""), 5000),
            "tool_results": transcript[-6:],
            "available_image_ids": sorted(image_map),
            "image_candidates": [
                {
                    "img_id": item.get("img_id", ""),
                    "title": item.get("title", ""),
                    "page_url": item.get("page_url", ""),
                    "visual_role": item.get("visual_role", "general"),
                    "search_layer": item.get("search_layer", 0),
                    "search_query": item.get("search_query", ""),
                    "retrieved_at": item.get("retrieved_at", ""),
                    "width": item.get("width", 0),
                    "height": item.get("height", 0),
                    "quality_score": item.get("quality_score", 0),
                    "information_score": item.get("information_score", 0),
                }
                for item in image_map.values()
            ],
            "evidence_count": len(evidence),
            "image_search_done": image_search_done,
        }
        content = [{"type": "text", "text": json.dumps(tool_state, ensure_ascii=False)}]
        content.extend(candidate_images)
        write_generation_event(
            "chain_three.planner_visual_context",
            "completed",
            candidate_count=len(image_map),
            image_count=len(candidate_images),
            image_ids=[item.get("img_id", "") for item in self._diverse_candidate_pool(image_map)[:MAX_PLANNER_REFERENCE_IMAGES]],
        )
        response = gateway.chat_completion(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            model=getattr(gateway, "vl_model", None) or gateway.text_model,
            temperature=0.2,
            max_tokens=1800,
            response_format={"type": "json_object"},
            timeout=min(60, self.timeout),
            error_label="链路三检索推理模型",
        )
        content = _extract_response_content(response)
        return _parse_action(content)

    def _planner_candidates(self, image_map):
        inputs = []
        for item in self._diverse_candidate_pool(image_map)[:MAX_PLANNER_REFERENCE_IMAGES]:
            data_url = _path_to_data_url(item.get("local_path", ""))
            if not data_url:
                continue
            planner_url, details = prepare_visual_planner_image(self.values, data_url)
            write_generation_event(
                "chain_three.planner_image_prepare",
                "completed",
                img_id=item.get("img_id", ""),
                **details,
            )
            inputs.append(
                {"type": "text", "text": f"候选参考图 {item.get('img_id', '')}：{item.get('title', '')}"}
            )
            inputs.append({"type": "image_url", "image_url": {"url": planner_url}})
        return inputs

    def _review_references(self, gateway, user_input, image_map, planned):
        candidates = self._diverse_candidate_pool(image_map)[:MAX_REVIEW_REFERENCE_IMAGES]
        if not candidates:
            return {}
        content = [
            {
                "type": "text",
                "text": (
                    "请对候选参考图做生成前质量复核，只返回 JSON："
                    '{"reference_images":[{"img_id":"IMG_001","score":0.0,"note":""}],'
                    '"selection_reason":""}。\n'
                    f"任务：{user_input}\n"
                    "规则：优先主体相关、清晰、分辨率高、构图信息完整的图片；"
                    "相同 visual_role 或同一主体场景只保留一张；优先最新、权威、画质高且信息量大的图片；"
                    "最终选择 1 到 5 张，并返回每张图片的 visual_role。"
                ),
            }
        ]
        for item in candidates:
            data_url = _path_to_data_url(item.get("local_path", ""))
            if not data_url:
                continue
            planner_url, _details = prepare_visual_planner_image(self.values, data_url)
            content.append(
                {"type": "text", "text": f"候选图 {item.get('img_id', '')}：{item.get('title', '')}"}
            )
            content.append({"type": "image_url", "image_url": {"url": planner_url}})
        try:
            response = gateway.chat_completion(
                [
                    {
                        "role": "system",
                        "content": "你是图片参考质量审核器，必须直接观察图片并严格输出 JSON，不要 Markdown。",
                    },
                    {"role": "user", "content": content},
                ],
                model=getattr(gateway, "vl_model", None) or gateway.text_model,
                temperature=0.0,
                max_tokens=1200,
                response_format={"type": "json_object"},
                timeout=self.review_timeout,
                error_label="链路三参考图质量审核模型",
            )
            review = _parse_action(_extract_response_content(response))
        except Exception as exc:
            write_generation_event(
                "chain_three.reference_review",
                "failed",
                candidate_count=len(candidates),
                planned_ids=[item.get("img_id", "") for item in planned],
                error_message=str(exc)[:500],
            )
            return {}
        review_items = review.get("reference_images")
        if not isinstance(review_items, list):
            return {}
        valid_ids = set(image_map)
        accepted = []
        rejected = []
        minimum_score = _bounded_float(
            self.values.get("GEN_SEARCHER_MIN_REFERENCE_SCORE"),
            0.45,
            0.0,
            1.0,
        )
        for item in review_items[: self.reference_limit]:
            if not isinstance(item, dict):
                continue
            img_id = str(item.get("img_id") or "").strip()
            if img_id not in valid_ids:
                continue
            score = _bounded_float(item.get("score"), 0.0, 0.0, 1.0)
            entry = {
                "img_id": img_id,
                "note": str(item.get("note") or "").strip()[:500],
                "score": score,
                "visual_role": _normalize_visual_role(item.get("visual_role")),
            }
            if score >= minimum_score:
                accepted.append(entry)
            else:
                rejected.append(entry)
        if not accepted:
            return {}
        write_generation_event(
            "chain_three.reference_review",
            "completed",
            candidate_count=len(candidates),
            planned_ids=[item.get("img_id", "") for item in planned],
            accepted=accepted,
            rejected=rejected,
            minimum_score=minimum_score,
            selection_reason=str(review.get("selection_reason") or "")[:1000],
        )
        return {"reference_images": accepted}

    @staticmethod
    def _diverse_candidate_pool(image_map):
        """Return the strongest current candidate from each visual layer first."""
        candidates = [
            dict(item)
            for item in image_map.values()
            if item.get("local_path") or item.get("url")
        ]
        by_role = {}
        for item in candidates:
            role = _normalize_visual_role(item.get("visual_role"))
            item["visual_role"] = role
            current = by_role.get(role)
            if current is None or _reference_rank(item) > _reference_rank(current):
                by_role[role] = item
        selected = list(by_role.values())
        selected.sort(key=_reference_rank, reverse=True)
        return selected

    def _execute_tool(self, tool, arguments, image_map, trajectory_id, entity_hint=""):
        if tool == "image_search":
            query = str(arguments.get("query") or "").strip() or "电商产品图片"
            visual_role = _normalize_visual_role(arguments.get("visual_role"), query)
            search_layer = _bounded_int(arguments.get("search_layer"), 0, 0, 9)
            selection_goal = str(arguments.get("selection_goal") or "").strip()[:500]
            refs = _image_search(
                self.values,
                query,
                self.image_result_limit,
                trajectory_id,
                visual_role=visual_role,
                search_layer=search_layer,
                selection_goal=selection_goal,
                entity_hint=entity_hint,
            )
            registered = {}
            evidence = []
            for item in refs:
                identity = item.get("url") or item.get("local_path")
                existing_identities = {
                    value.get("url") or value.get("local_path")
                    for value in image_map.values()
                }
                existing_hashes = {
                    value.get("content_hash")
                    for value in image_map.values()
                    if value.get("content_hash")
                }
                if (
                    not identity
                    or identity in existing_identities
                    or item.get("content_hash") in existing_hashes
                ):
                    continue
                img_id = f"IMG_{len(image_map) + len(registered) + 1:03d}"
                item = dict(item)
                item["img_id"] = img_id
                item["visual_role"] = _normalize_visual_role(item.get("visual_role"), query)
                item["search_layer"] = search_layer
                item["search_query"] = query
                item["selection_goal"] = selection_goal
                registered[img_id] = item
                evidence.append(
                    {
                        "source_type": "image_search",
                        "title": item.get("title", ""),
                        "url": item.get("url", ""),
                    }
                )
            return _format_image_results(query, registered), registered, evidence
        if tool == "search":
            query = arguments.get("query") or arguments.get("queries") or ""
            text, items = _text_search(self.values, query, self.timeout)
            return text, {}, items
        if tool == "browse":
            url = str(arguments.get("url") or "").strip()
            query = str(arguments.get("query") or arguments.get("goal") or "").strip()
            text = _browse(self.values, url, query, self.timeout)
            return text, {}, [{"source_type": "browse", "url": url, "title": query}]
        return f"工具 {tool} 未注册。可用工具：search、image_search、browse。", {}, []

    def _select_references(self, final, image_map, requirements=None):
        requested = final.get("reference_images")
        ids = []
        if isinstance(requested, list):
            for item in requested:
                if isinstance(item, str):
                    ids.append(item)
                elif isinstance(item, dict):
                    ids.append(str(item.get("img_id") or "").strip())
        requested_items = []
        for img_id in dict.fromkeys(ids):
            if img_id in image_map and (
                image_map[img_id].get("local_path") or image_map[img_id].get("url")
            ):
                item = dict(image_map[img_id])
                requested_note = next(
                    (
                        entry.get("note", "")
                        for entry in (requested or [])
                        if isinstance(entry, dict) and str(entry.get("img_id") or "").strip() == img_id
                    ),
                    "",
                )
                requested_score = next(
                    (
                        entry.get("score")
                        for entry in (requested or [])
                        if isinstance(entry, dict)
                        and str(entry.get("img_id") or "").strip() == img_id
                    ),
                    None,
                )
                if requested_note:
                    item["note"] = str(requested_note).strip()[:500]
                if requested_score is not None:
                    item["reference_score"] = _bounded_float(requested_score, 0.0, 0.0, 1.0)
                requested_items.append(item)

        required_roles = [
            item["visual_role"]
            for item in (requirements or [])
            if item.get("required")
        ]
        selected = []
        used_roles = set()

        # A model-selected layer is represented by its strongest current image,
        # so a low-resolution duplicate cannot displace a newer, richer result.
        requested_roles = []
        for item in requested_items:
            role = _normalize_visual_role(item.get("visual_role"))
            if role not in requested_roles:
                requested_roles.append(role)
        for role in [*required_roles, *requested_roles]:
            if role in used_roles:
                continue
            role_candidates = [
                dict(item)
                for item in image_map.values()
                if _normalize_visual_role(item.get("visual_role")) == role
                and (item.get("local_path") or item.get("url"))
            ]
            if not role_candidates:
                continue
            best = max(role_candidates, key=_reference_rank)
            requested_match = next(
                (item for item in requested_items if _normalize_visual_role(item.get("visual_role")) == role),
                None,
            )
            if requested_match and requested_match.get("note"):
                best["note"] = requested_match["note"]
            if requested_match and requested_match.get("reference_score") is not None:
                best["reference_score"] = requested_match["reference_score"]
            best["visual_role"] = role
            selected.append(best)
            used_roles.add(role)
            if len(selected) >= self.reference_limit:
                break

        if len(selected) < self.reference_limit:
            for candidate in self._diverse_candidate_pool(image_map):
                role = _normalize_visual_role(candidate.get("visual_role"))
                if role in used_roles:
                    continue
                selected.append(candidate)
                used_roles.add(role)
                if len(selected) >= self.reference_limit:
                    break
        return selected

    @staticmethod
    def _fallback_action(user_input, knowledge, image_map):
        refs = [
            {
                "img_id": item.get("img_id", ""),
                "visual_role": _normalize_visual_role(item.get("visual_role")),
                "note": "检索到的视觉参考，用于确认对应视觉层。",
            }
            for item in sorted(
                (dict(value) for value in image_map.values()),
                key=_reference_rank,
                reverse=True,
            )[:DEFAULT_IMAGE_REFERENCE_LIMIT]
        ]
        return {
            "type": "final",
            "gen_prompt": _fallback_prompt(user_input, knowledge, image_map),
            "reference_images": refs,
            "reply_text": "已根据检索到的文本和视觉参考生成图片。",
        }

    @staticmethod
    def _fallback(user_input, knowledge, reason):
        return ChainThreeResult(
            reply_text=reason,
            status="placeholder",
            image_prompt=_fallback_prompt(user_input, knowledge, {}),
            error=reason,
        )

    @staticmethod
    def _limit_text(value, limit):
        text = str(value or "").strip()
        return text if len(text) <= limit else text[:limit] + "..."


def _bounded_int(value, fallback, minimum, maximum):
    try:
        return min(max(int(value), minimum), maximum)
    except (TypeError, ValueError):
        return fallback


def _bounded_float(value, fallback, minimum, maximum):
    try:
        return min(max(float(value), minimum), maximum)
    except (TypeError, ValueError):
        return fallback


def _extract_response_content(response):
    choices = response.get("choices", []) if isinstance(response, dict) else []
    if not choices:
        return ""
    content = choices[0].get("message", {}).get("content", "")
    if isinstance(content, list):
        content = "".join(
            str(item.get("text") or "") for item in content if isinstance(item, dict)
        )
    return str(content or "").strip()


def _parse_action(content):
    raw = str(content or "").strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE).strip()
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if match:
        raw = match.group(0)
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {"type": "invalid", "error": "invalid_json"}
    return value if isinstance(value, dict) else {"type": "invalid", "error": "invalid_action"}


def _normalize_final_action(action):
    """Accept both the local flat result and the shared image_task envelope."""
    normalized = dict(action or {})
    answer = normalized.get("answer")
    if isinstance(answer, str) and answer.strip():
        parsed = _parse_action(answer)
        if isinstance(parsed, dict):
            normalized.update({key: value for key, value in parsed.items() if key != "type"})
    image_task = normalized.get("image_task")
    if isinstance(image_task, dict):
        normalized.setdefault("gen_prompt", image_task.get("gen_prompt"))
        normalized.setdefault("reference_images", image_task.get("reference_images"))
        normalized.setdefault("visual_requirements", image_task.get("visual_requirements"))
    return normalized


def _search_query(user_input):
    words = re.findall(r"[\w\u4e00-\u9fff-]+", str(user_input or ""))
    return " ".join(words[:8]) or "电商产品视觉参考"


def _extract_entity_hint(value):
    """Extract the longest named institution/brand-like subject for search filtering."""
    text = str(value or "").strip()
    candidates = re.findall(
        r"[\u4e00-\u9fffA-Za-z0-9·()（）]{2,40}(?:大学科技学院|科技学院|学院|大学|学校)",
        text,
    )
    if not candidates:
        return ""
    candidate = max(candidates, key=len)
    candidate = re.sub(r"^(?:请|帮我|生成|设计|制作|做|为|给我|一张|一幅)+", "", candidate)
    return candidate.strip()


def _is_generic_image_task(value):
    """Return True when the request is a self-contained, non-factual subject.

    Such requests (e.g. a skateboard, chair, fruit or icon) benefit from the
    image model's native synthesis and should not trigger web grounding.
    Requests naming institutions, brands, places, people, products with a
    specific model, or an explicit real scene remain search-grounded.
    """
    text = str(value or "").strip()
    if not text or _extract_entity_hint(text):
        return False
    factual_markers = (
        "校徽", "学校", "大学", "学院", "品牌", "logo", "商标", "官网",
        "校园", "建筑", "景点", "地点", "街景", "真实", "实拍", "照片",
        "人物", "名人", "产品型号", "型号", "店铺", "门店",
    )
    if any(marker.casefold() in text.casefold() for marker in factual_markers):
        return False
    # Explicitly requested visual generation/editing of a simple subject.
    return any(token in text for token in ("生成", "画", "绘制", "制作", "设计", "图片", "图像"))


def _normalized_entity_text(value):
    return re.sub(r"[\s_·\\/\\-]+", "", unquote(str(value or "")).casefold())


def _matches_entity(item, entity_hint):
    if not entity_hint:
        return True
    target = _normalized_entity_text(entity_hint)
    haystack = " ".join(
        str(item.get(key) or "")
        for key in ("title", "source", "link", "pageUrl", "page_url", "snippet", "description")
    )
    return bool(target and target in _normalized_entity_text(haystack))


def _safe_tool_arguments(arguments):
    output = {}
    for key, value in dict(arguments or {}).items():
        if key in {"query", "goal", "url", "visual_role", "search_layer", "selection_goal"}:
            if isinstance(value, (list, tuple)):
                output[key] = [str(item)[:500] for item in value[:5]]
            else:
                output[key] = str(value or "")[:1000]
    return output


def _reference_log_items(references):
    return [
        {
            "img_id": str(item.get("img_id") or ""),
            "title": str(item.get("title") or "")[:300],
            "note": str(item.get("note") or "")[:500],
            "source_url": str(item.get("url") or "")[:2000],
            "page_url": str(item.get("page_url") or "")[:2000],
            "local_path": str(item.get("local_path") or "")[:1000],
            "visual_role": _normalize_visual_role(item.get("visual_role")),
            "search_layer": item.get("search_layer", 0),
            "search_query": str(item.get("search_query") or "")[:500],
            "selection_goal": str(item.get("selection_goal") or "")[:500],
            "retrieved_at": str(item.get("retrieved_at") or "")[:40],
            "published_at": str(item.get("published_at") or "")[:40],
            "width": item.get("width", 0),
            "height": item.get("height", 0),
            "bytes": item.get("bytes", 0),
            "format": str(item.get("format") or "")[:20],
            "quality_score": item.get("quality_score", 0),
            "information_score": item.get("information_score", 0),
            "authority_score": item.get("authority_score", 0),
            "reference_score": item.get("reference_score", 0),
            "content_hash": str(item.get("content_hash") or "")[:32],
        }
        for item in references or []
    ]


VISUAL_ROLE_ALIASES = {
    "logo": "identity",
    "校徽": "identity",
    "徽标": "identity",
    "标识": "identity",
    "identity": "identity",
    "brand": "identity",
    "identity_logo": "identity",
    "campus": "representative_scene",
    "scene": "representative_scene",
    "校园": "representative_scene",
    "场景": "representative_scene",
    "建筑": "representative_scene",
    "校园场景": "representative_scene",
    "校园建筑": "representative_scene",
    "代表性场景": "representative_scene",
    "representative": "representative_scene",
    "representative_scene": "representative_scene",
    "activity": "activity",
    "活动": "activity",
    "人物": "activity",
    "detail": "detail",
    "细节": "detail",
    "局部": "detail",
    "style": "style",
    "风格": "style",
    "general": "general",
}


def _normalize_visual_role(value, query=""):
    raw = str(value or "").strip().lower()
    if raw in VISUAL_ROLE_ALIASES:
        return VISUAL_ROLE_ALIASES[raw]
    for alias, role in VISUAL_ROLE_ALIASES.items():
        if alias and alias in raw:
            return role
    query_text = str(query or "").strip().lower()
    for alias, role in VISUAL_ROLE_ALIASES.items():
        if alias and alias in query_text:
            return role
    return "general"


def _normalize_visual_requirements(raw):
    output = []
    if not isinstance(raw, list):
        return output
    for item in raw[:MAX_VISUAL_REQUIREMENTS]:
        if not isinstance(item, dict):
            continue
        role = _normalize_visual_role(item.get("visual_role"), item.get("query"))
        query = str(item.get("query") or "").strip()[:500]
        reason = str(item.get("reason") or item.get("selection_goal") or "").strip()[:500]
        layer = _bounded_int(item.get("search_layer"), len(output) + 1, 1, 9)
        required_value = item.get("required")
        required = (
            required_value
            if isinstance(required_value, bool)
            else str(required_value or "").strip().lower() in {"1", "true", "yes", "required", "必须"}
        )
        normalized = {
            "visual_role": role,
            "required": required,
            "query": query,
            "search_layer": layer,
            "reason": reason,
        }
        if not any(existing["visual_role"] == role for existing in output):
            output.append(normalized)
    return output


def _has_visual_role(image_map, role):
    return any(
        _normalize_visual_role(item.get("visual_role")) == role
        and (item.get("local_path") or item.get("url"))
        for item in image_map.values()
    )


def _source_authority_score(page_url):
    hostname = urlparse(str(page_url or "")).hostname or ""
    hostname = hostname.lower().rstrip(".")
    if hostname.endswith(".edu.cn") or hostname.endswith(".ac.cn"):
        return 1.0
    if "official" in hostname or hostname.startswith("www."):
        return 0.8
    if hostname.endswith("baidu.com") or hostname.endswith("baike.com"):
        return 0.6
    return 0.4 if hostname else 0.0


def _reference_rank(item):
    """Rank current references by freshness, source authority and visual detail."""
    quality = _bounded_float(item.get("quality_score"), 0.0, 0.0, 1.0)
    information = _bounded_float(item.get("information_score"), 0.0, 0.0, 1.0)
    authority = _bounded_float(item.get("authority_score"), 0.0, 0.0, 1.0)
    published = str(item.get("published_at") or "").strip()
    published_epoch = 0.0
    if published:
        try:
            published_epoch = datetime.fromisoformat(published.replace("Z", "+00:00")).timestamp()
        except ValueError:
            published_epoch = 0.0
    retrieved = str(item.get("retrieved_at") or "").strip()
    retrieved_epoch = 0.0
    if retrieved:
        try:
            retrieved_epoch = datetime.fromisoformat(retrieved.replace("Z", "+00:00")).timestamp()
        except ValueError:
            retrieved_epoch = 0.0
    return (
        published_epoch or retrieved_epoch,
        round(0.55 * quality + 0.30 * information + 0.15 * authority, 6),
        int(item.get("width") or 0) * int(item.get("height") or 0),
        int(item.get("bytes") or 0),
    )


def _selection_audit(image_map, selected):
    selected_ids = {str(item.get("img_id") or "") for item in selected}
    groups = {}
    for item in image_map.values():
        role = _normalize_visual_role(item.get("visual_role"))
        groups.setdefault(role, []).append(item)
    audit = []
    for role, items in sorted(groups.items()):
        ranked = sorted(items, key=_reference_rank, reverse=True)
        audit.append(
            {
                "visual_role": role,
                "winner_img_id": next(
                    (item.get("img_id") for item in ranked if item.get("img_id") in selected_ids),
                    "",
                ),
                "candidate_order": [item.get("img_id", "") for item in ranked],
                "discarded_img_ids": [
                    item.get("img_id", "")
                    for item in ranked
                    if item.get("img_id", "") not in selected_ids
                ],
                "rank_factors": [
                    {
                        "img_id": item.get("img_id", ""),
                        "quality_score": item.get("quality_score", 0),
                        "information_score": item.get("information_score", 0),
                        "authority_score": item.get("authority_score", 0),
                        "retrieved_at": item.get("retrieved_at", ""),
                        "published_at": item.get("published_at", ""),
                    }
                    for item in ranked[:8]
                ],
            }
        )
    return audit


def _requested_reference_ids(final):
    requested = final.get("reference_images") if isinstance(final, dict) else []
    ids = []
    for item in requested if isinstance(requested, list) else []:
        if isinstance(item, str):
            value = item.strip()
        elif isinstance(item, dict):
            value = str(item.get("img_id") or "").strip()
        else:
            value = ""
        if value:
            ids.append(value)
    return ids


def _text_search(values, query, timeout):
    endpoint = str(values.get("TEXT_SEARCH_API_BASE_URL") or values.get("SEARCH_API_URL") or "").strip()
    api_key = str(values.get("SERPER_KEY_ID") or values.get("SEARCH_API_KEY") or "").strip()
    if not endpoint or not api_key:
        return "[search] 未配置 TEXT_SEARCH_API_BASE_URL/SEARCH_API_KEY。", []
    if isinstance(query, (list, tuple)):
        query = " ".join(str(item).strip() for item in query if str(item).strip())
    payload = {"q": str(query or "")[:500], "num": 5}
    try:
        with generation_stage(
            "chain_three.search_http",
            endpoint=_safe_url(endpoint),
            query=str(query or "")[:500],
        ):
            data = _post_json(endpoint, payload, api_key, timeout)
        items = data.get("organic") or data.get("results") or []
        lines = []
        evidence = []
        for item in items[:5]:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "").strip()
            url = str(item.get("link") or item.get("url") or "").strip()
            snippet = str(item.get("snippet") or item.get("description") or "").strip()
            lines.append(f"[{title}] {snippet} {url}".strip())
            evidence.append({"source_type": "text_search", "title": title, "url": url, "excerpt": snippet})
        return "\n".join(lines) or "[search] 未找到结果。", evidence
    except Exception as exc:
        return f"[search] 请求失败：{str(exc)[:500]}", []


def _image_search(
    values,
    query,
    limit,
    task_id,
    visual_role="general",
    search_layer=0,
    selection_goal="",
    entity_hint="",
):
    endpoint = str(values.get("IMAGE_SEARCH_API_BASE_URL") or "").strip()
    api_key = str(values.get("SERPER_KEY_ID") or values.get("SEARCH_API_KEY") or "").strip()
    if not endpoint or not api_key:
        return []
    def fetch_results(search_query, refined=False):
        try:
            with generation_stage(
                "chain_three.image_search_http",
                endpoint=_safe_url(endpoint),
                query=str(search_query or "")[:500],
                result_limit=limit,
                refined=refined,
            ):
                data = _post_json(
                    endpoint,
                    {"q": str(search_query or "")[:500], "num": limit},
                    api_key,
                    DEFAULT_SEARCH_TIMEOUT,
                )
            return data.get("images") or data.get("results") or []
        except Exception:
            return []

    raw_items = fetch_results(query)
    filtered_items = [item for item in raw_items if isinstance(item, dict) and _matches_entity(item, entity_hint)]
    rejected_items = [item for item in raw_items if isinstance(item, dict) and not _matches_entity(item, entity_hint)]
    if entity_hint and rejected_items:
        refined_query = f'"{entity_hint}" {query}'.strip()
        refined_items = fetch_results(refined_query, refined=True)
        existing_urls = {
            str(item.get("imageUrl") or item.get("image_url") or item.get("url") or item.get("thumbnailUrl") or "").strip()
            for item in filtered_items
            if isinstance(item, dict)
        }
        for item in refined_items:
            if not isinstance(item, dict) or not _matches_entity(item, entity_hint):
                continue
            item_url = str(
                item.get("imageUrl")
                or item.get("image_url")
                or item.get("url")
                or item.get("thumbnailUrl")
                or ""
            ).strip()
            if not item_url or item_url in existing_urls:
                continue
            filtered_items.append(item)
            existing_urls.add(item_url)
    write_generation_event(
        "chain_three.image_search_filter",
        "completed",
        entity_hint=entity_hint,
        raw_count=len(raw_items),
        accepted_count=len(filtered_items),
        rejected_count=len(rejected_items),
        refined=bool(entity_hint and rejected_items),
    )
    save_dir = Path(
        str(
            values.get("GEN_SEARCHER_IMAGE_DIR")
            or values.get("IMAGE_SEARCH_SAVE_DIR")
            or DEFAULT_IMAGE_DIR
        )
    ) / str(task_id)
    save_dir.mkdir(parents=True, exist_ok=True)
    output = []
    seen = set()
    seen_content = set()
    retrieved_at = datetime.now(timezone.utc).isoformat()
    for item in filtered_items[: max(limit, 1)]:
        if not isinstance(item, dict):
            continue
        url = str(
            item.get("imageUrl") or item.get("image_url") or item.get("url") or item.get("thumbnailUrl") or ""
        ).strip()
        if not url or url in seen:
            continue
        seen.add(url)
        with generation_stage(
            "chain_three.image_download",
            image_index=len(output) + 1,
            image_url=_safe_url(url),
        ):
            local_path = _download_image(
                url,
                save_dir,
                item.get("link") or item.get("pageUrl") or "",
            )
        if not local_path:
            continue
        metadata = _inspect_image_file(local_path)
        if not metadata:
            continue
        content_hash = metadata.get("content_hash", "")
        if content_hash and content_hash in seen_content:
            continue
        if content_hash:
            seen_content.add(content_hash)
        output.append(
            {
                "title": str(item.get("title") or item.get("source") or "图片参考").strip()[:300],
                "url": url,
                "page_url": str(item.get("link") or item.get("pageUrl") or "").strip(),
                "local_path": local_path,
                "visual_role": _normalize_visual_role(visual_role, query),
                "search_layer": search_layer,
                "search_query": str(query or "")[:500],
                "selection_goal": str(selection_goal or "")[:500],
                "retrieved_at": retrieved_at,
                "published_at": str(
                    item.get("date") or item.get("publishedAt") or item.get("published_at") or ""
                ).strip()[:40],
                "authority_score": _source_authority_score(
                    str(item.get("link") or item.get("pageUrl") or "")
                ),
                **metadata,
            }
        )
        if len(output) >= limit:
            break
    return output


def _browse(values, url, query, timeout):
    if not url:
        return "[browse] 缺少 URL。"
    endpoint = str(values.get("BROWSE_API_URL") or "").strip()
    if not endpoint:
        endpoint = "https://r.jina.ai/http://" + url.removeprefix("http://").removeprefix("https://")
    elif "{url}" in endpoint:
        endpoint = endpoint.replace("{url}", url)
    elif endpoint.endswith("/http://") or endpoint.endswith("/https://"):
        endpoint += url.removeprefix("http://").removeprefix("https://")
    api_key = str(values.get("JINA_API_KEY") or "").strip()
    headers = {"Accept": "text/plain"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        with generation_stage(
            "chain_three.browse_http",
            endpoint=_safe_url(endpoint),
            query=str(query or "")[:500],
        ):
            req = request.Request(endpoint, headers=headers, method="GET")
            with request.urlopen(req, timeout=timeout) as response:
                body = response.read().decode("utf-8")
        return f"页面目标：{query or '总结页面'}\n{body[:12000]}"
    except Exception as exc:
        return f"[browse] 请求失败：{str(exc)[:500]}"


def _post_json(endpoint, payload, api_key, timeout):
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = request.Request(
        endpoint,
        data=data,
        headers={"X-API-KEY": api_key, "Authorization": f"Bearer {api_key}", "Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _safe_url(value):
    """Keep host/path useful in diagnostics without logging query secrets."""
    parsed = urlparse(str(value or ""))
    if not parsed.scheme or not parsed.netloc:
        return str(value or "")[:500]
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"[:500]


def _download_image(url, save_dir, page_url):
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:32]
    path = save_dir / f"{digest}.img"
    try:
        headers = {"User-Agent": "Mozilla/5.0", "Accept": "image/*,*/*;q=0.8"}
        if page_url:
            headers["Referer"] = str(page_url)
        req = request.Request(url, headers=headers, method="GET")
        with request.urlopen(req, timeout=DEFAULT_SEARCH_TIMEOUT) as response:
            content = response.read()
            content_type = str(response.headers.get_content_type() or "").lower()
        if len(content) < 512 or (content_type and not content_type.startswith("image/")):
            return ""
        extension = mimetypes.guess_extension(content_type) or ".jpg"
        final_path = path.with_suffix(extension)
        with open(final_path, "wb") as handle:
            handle.write(content)
        return str(final_path)
    except (OSError, ValueError, error.URLError, TimeoutError):
        return ""


def _inspect_image_file(path):
    if Image is None:
        return {"bytes": Path(path).stat().st_size, "quality_score": 0.5, "information_score": 0.5}
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
            image_format = str(image.format or "").upper()
        if min(width, height) < MIN_REFERENCE_IMAGE_SIDE:
            return {}
        size_bytes = Path(path).stat().st_size
        resolution_score = min(1.0, (width * height) / float(1600 * 1600))
        size_score = min(1.0, size_bytes / float(250 * 1024))
        quality_score = round(0.7 * resolution_score + 0.3 * size_score, 3)
        with Image.open(path) as image:
            gray = image.convert("L")
            gray.thumbnail((128, 128), Image.Resampling.BILINEAR)
            histogram = gray.histogram()
        total_pixels = float(sum(histogram) or 1)
        entropy = -sum(
            (count / total_pixels) * math.log2(count / total_pixels)
            for count in histogram
            if count
        )
        information_score = round(min(1.0, entropy / 8.0), 3)
        digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()[:32]
        return {
            "width": width,
            "height": height,
            "bytes": size_bytes,
            "format": image_format,
            "content_hash": digest,
            "quality_score": quality_score,
            "information_score": information_score,
        }
    except (OSError, ValueError):
        return {}


def _path_to_data_url(path):
    try:
        path = str(path or "").strip()
        if not path:
            return ""
        content_type = mimetypes.guess_type(path)[0] or "image/jpeg"
        with open(path, "rb") as handle:
            encoded = base64.b64encode(handle.read()).decode("ascii")
        return f"data:{content_type};base64,{encoded}"
    except (OSError, ValueError):
        return ""


def _format_image_results(query, refs):
    lines = [f"图片搜索：{query}"]
    for img_id in sorted(refs):
        item = refs[img_id]
        lines.append(
            f"{img_id}：{item.get('title', '')}；层级：{item.get('search_layer', 0)}；"
            f"用途：{item.get('visual_role', 'general')}；画质：{item.get('quality_score', 0)}；"
            f"信息量：{item.get('information_score', 0)}；来源页：{item.get('page_url', '')}；"
            f"本地路径：{item.get('local_path', '')}"
        )
    return "\n".join(lines) if len(lines) > 1 else "图片搜索未找到可下载结果。"


def _clean_prompt(value):
    prompt = str(value or "").strip()
    prompt = re.sub(r"https?://\S+", "", prompt)
    prompt = re.sub(r"IMG_\d+", "", prompt, flags=re.IGNORECASE)
    return " ".join(prompt.split())[:12000]


def _ensure_reference_ordinals(prompt, count):
    if count <= 0:
        return prompt
    if re.search(r"第[一二三四五]张参考图|第一张参考图|第二张参考图", prompt):
        return prompt
    return f"参考第1张参考图中的主体和关键视觉事实。{prompt}"


def _fallback_prompt(user_input, knowledge, image_map):
    context = str(getattr(knowledge, "context_text", "") or "").strip()
    refs = "参考检索到的视觉资料" if image_map else "基于用户描述"
    return _clean_prompt(
        f"{user_input}。{refs}，保持主体清晰、构图稳定、光线自然、细节真实，"
        f"输出适合电商展示的高质量图片。补充资料：{context[:2500]}"
    )


__all__ = ["ChainThreeResult", "GenSearcherImageChain"]
