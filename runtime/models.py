# -*- coding: utf-8 -*-

import uuid
from dataclasses import dataclass, field
from typing import Any


ROUTE_TEXT = "text"
ROUTE_CHAIN_ONE = "chain_1_history_image"
ROUTE_CHAIN_TWO = "chain_2_uploaded_image"
ROUTE_CHAIN_THREE = "chain_3_placeholder"


@dataclass(frozen=True)
class IntentDecision:
    intent: str
    use_previous_image: bool = False
    reason: str = ""


@dataclass(frozen=True)
class Copywriting:
    headline: str = ""
    subheadline: str = ""
    cta: str = ""


@dataclass(frozen=True)
class ImagePromptPlan:
    reply_text: str
    image_prompt: str
    copywriting: Copywriting = field(default_factory=Copywriting)


@dataclass
class ImageTaskState:
    user_input: str
    conversation_id: str
    history_messages: list[dict[str, str]] = field(default_factory=list)
    visual_history: list[dict[str, Any]] = field(default_factory=list)
    uploaded_content: Any = None
    force_image: bool = False
    poster_context: dict[str, str] = field(default_factory=dict)
    task_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    intent: str = ""
    route: str = ""
    knowledge_status: str = "not_used"
    actions: list[dict[str, Any]] = field(default_factory=list)

    def record(self, action: str, **details: Any) -> None:
        self.actions.append({"action": action, **details})


@dataclass(frozen=True)
class OrchestrationResult:
    reply_text: str
    route: str
    status: str
    image_url: str = ""
    image_prompt: str = ""
    knowledge_status: str = "not_used"
    copywriting: Copywriting = field(default_factory=Copywriting)
    task_id: str = ""
    actions: tuple[dict[str, Any], ...] = ()
