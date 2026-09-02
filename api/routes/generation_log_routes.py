# -*- coding: utf-8 -*-

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

from core.config import read_config


router = APIRouter(prefix="/api/generation-logs", tags=["generation-logs"])


@router.get("/{task_id}")
def generation_log_detail(task_id: str):
    task_id = str(task_id or "").strip()
    if not task_id or len(task_id) > 128:
        raise HTTPException(status_code=400, detail="task_id 无效。")

    values = read_config(".env")
    generation_path = Path(values.get("GENERATION_LOG_PATH") or "logs/generation.jsonl")
    llm_path = Path(values.get("LLM_LOG_PATH") or "logs/llm.jsonl")
    events = _read_task_records(generation_path, task_id)
    request_logs = _read_task_records(llm_path, task_id)
    if not events and not request_logs:
        raise HTTPException(status_code=404, detail="未找到该 task_id 的生成日志。")

    reference_items = []
    final_prompt = ""
    image_input = {}
    image_model_inputs = []
    candidate_count = 0
    visual_requirements = []
    selection_audit = []
    for event in events:
        stage = event.get("stage")
        if stage == "chain_three.references":
            reference_items = event.get("references") or []
            try:
                candidate_count = int(event.get("candidate_count") or 0)
            except (TypeError, ValueError):
                candidate_count = 0
            visual_requirements = event.get("visual_requirements") or []
            selection_audit = event.get("selection_audit") or []
        elif stage == "chain_three.selection_decision":
            if not visual_requirements:
                visual_requirements = event.get("visual_requirements") or []
        elif stage == "chain_three.prompt":
            final_prompt = str(event.get("prompt") or "")
        elif stage == "chain_three.image_input":
            image_input = event
        elif stage in {"image_model.input", "image_model.payload", "image_model.generate"}:
            image_model_inputs.append(event)

    if not final_prompt:
        for record in reversed(request_logs):
            final_prompt = str(record.get("image_prompt") or "")
            if final_prompt:
                break

    reference_count = len(reference_items)
    if not reference_count:
        reference_count = int(image_input.get("reference_image_count") or 0)
    if not reference_count:
        for event in reversed(image_model_inputs):
            try:
                reference_count = int(event.get("reference_image_count") or 0)
            except (TypeError, ValueError):
                reference_count = 0
            if reference_count:
                break
    return {
        "task_id": task_id,
        "candidate_image_count": candidate_count,
        "reference_image_count": reference_count,
        "reference_images": reference_items,
        "visual_requirements": visual_requirements,
        "selection_audit": selection_audit,
        "final_image_prompt": final_prompt,
        "image_input": image_input,
        "image_model_inputs": image_model_inputs,
        "events": events,
        "request_logs": request_logs,
    }


def _read_task_records(path, task_id):
    if not path.is_file():
        return []
    records = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict) and str(record.get("task_id") or "") == task_id:
                    records.append(record)
    except OSError:
        return []
    return records


__all__ = ["router"]
