# -*- coding: utf-8 -*-

SYSTEM_MESSAGES = {
    "text": (
        "你是电商视觉与文案协作助手，不只是文案生成器。"
        "你需要结合历史上下文判断用户当前是在要成品、追问解释、评价上一版，还是要求继续修改。"
        "如果用户明确要求写、生成、输出、给我文案、给我标题、给我卖点，就直接输出可使用成品。"
        "如果用户只是反馈上一版，例如“不够大气”“不够高级”“不好看”“不对”“还差点意思”，"
        "不要直接重写完整成品；先指出上一版具体不足，再给出可执行优化方向，最后询问是否需要按建议重写。"
        "如果用户明确说“按你说的重写”“重新生成”“直接改成最终版”，再输出新版本。"
        "如果用户询问当前项目能力或实现逻辑，要基于本项目事实回答：当前已实现基于 conversation_id 的 MySQL 会话持久化，"
        "同一会话从 MySQL 读取历史消息和图片元数据，生成图片二进制缓存在 Redis；这属于会话级记忆，"
        "不是摘要记忆、向量检索或跨会话长期知识库。"
        "回答要像协作聊天，不要每次都套销售文案格式。"
    ),
    "image": (
        "你是电商图片生成策划助手。请先理解用户想要的图片，再返回严格 JSON。"
        "reply_text 是给用户看的简短说明，不能包含完整生图提示词；"
        "image_prompt 是专门给图片生成模型使用的提示词，必须包含主体、场景、风格、构图、光线、色彩、画幅和需要避免的内容。"
    ),
    "mixed": (
        "你是电商视觉与文案助手。请先根据用户需求生成可直接给用户使用的配套文案，"
        "再单独整理给图片生成模型使用的提示词，并返回严格 JSON。"
        "reply_text 是给用户看的文案或说明，不能把生图提示词原样塞进去；"
        "image_prompt 是专门给图片生成模型使用的提示词，必须具体描述主体、场景、风格、构图、光线、色彩、画幅和需要避免的内容。"
    ),
}


OUTPUT_STYLE_RULES = (
    "输出格式要求："
    "面向前端聊天气泡展示，默认使用干净的纯文本。"
    "不要使用 Markdown 语法，包括 **加粗**、### 标题、--- 分割线、> 引用、```代码块、Markdown 表格。"
    "不要使用 emoji 或装饰性符号作为项目符号。"
    "列表请使用“1. 2. 3.”或短横线，标题请使用中文标题加冒号。"
    "除非用户明确要求表格，否则不要输出表格。"
)


def build_visual_history_text(visual_history):
    if not visual_history:
        return ""

    lines = ["当前会话最近生成图片记录："]
    for index, item in enumerate(visual_history, start=1):
        image_analysis = str(item.get("image_analysis") or "").strip()
        image_status = "已读取图片并生成视觉摘要" if image_analysis else "仅有图片地址和历史提示词"
        selected_text = "；本轮选中作为编辑原图" if item.get("selected_for_edit") else ""
        lines.append(
            f"{index}. 用户需求：{item.get('user_input', '')[:300]}{selected_text}\n"
            f"   助手说明：{item.get('assistant_text', '')[:300]}\n"
            f"   图片提示词：{item.get('image_prompt', '')[:600]}\n"
            f"   图片地址：{item.get('image_url', '')}\n"
            f"   图片内容状态：{image_status}"
        )
        if image_analysis:
            lines.append(f"   历史图片视觉摘要：{image_analysis[:1200]}")
    return "\n".join(lines)


def build_visual_history_image_parts(visual_history):
    return []


def build_message(user_input, intent, history_messages=None, visual_history=None):
    system_message = SYSTEM_MESSAGES.get(intent, SYSTEM_MESSAGES["text"])
    system_message += (
        "历史消息用于理解当前会话上下文。"
        "当用户当前输入是对上一轮的评价、追问或修正时，必须优先基于上一轮助手内容进行分析，"
        "不要默认当成新的独立生成任务。"
        "如果用户提到上一张图、刚才那张、上一版、第二版、继续优化等表达，"
        "优先参考当前会话最近生成图片记录中的用户需求、助手说明和图片提示词。"
        "如果当前消息包含历史图片视觉摘要，必须优先基于该视觉摘要和用户本轮要求生成新方案，"
        "不要只依赖图片地址；重新生成时要保留参考图中的主体、构图和关键视觉细节，再做局部优化。"
        f"{OUTPUT_STYLE_RULES}"
    )
    if intent in ["image", "mixed"]:
        system_message += (
            "只能返回 JSON 对象，不要使用 Markdown，不要输出代码块。"
            'JSON 格式为 {"reply_text": "...", "image_prompt": "..."}。'
        )
    messages = [
        {"role": "system", "content": system_message},
    ]
    if history_messages:
        messages.extend(history_messages)
    visual_history_text = build_visual_history_text(visual_history)
    current_content = f"用户意图：{intent}\n用户原始需求：{user_input}"
    if visual_history_text:
        current_content = f"{visual_history_text}\n\n{current_content}"

    image_parts = build_visual_history_image_parts(visual_history)
    if image_parts:
        current_content = [{"type": "text", "text": current_content}, *image_parts]

    messages.append({
        "role": "user",
        "content": current_content,
    })
    return messages
