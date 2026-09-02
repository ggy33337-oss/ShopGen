# -*- coding: utf-8 -*-

import json

from core.config import read_config
from knowledge.quality import is_meaningful_knowledge_text
from llm.qwen_reranker import QwenReranker


CALIBRATION_CASES = (
    (
        "背景改成品牌蓝色",
        (
            (1, "品牌视觉规范", "商品图背景标准蓝为 #0057B8。"),
            (0, "春季活动", "春季活动主视觉统一使用草绿色背景。"),
            (0, "物流说明", "商品默认使用顺丰快递发货。"),
            (0, "颜色常识", "蓝色是一种常见的冷色调。"),
            (0, "1.00", "1.00"),
        ),
    ),
    (
        "把鞋面改成哑光黑色皮革",
        (
            (1, "鞋履材质规范", "哑光黑色皮革应保留细腻皮纹，避免镜面反光。"),
            (0, "亮面材质规范", "亮面漆皮需要高光反射效果。"),
            (0, "售后说明", "运动鞋支持七天无理由退换。"),
            (0, "2.1", "2.1"),
        ),
    ),
    (
        "保留左上角品牌 Logo 并扩大安全距离",
        (
            (1, "Logo 规范", "标志位于左上角，四周安全距离不得小于标志高度的二分之一。"),
            (0, "包装规范", "包装箱右下角印制生产日期。"),
            (0, "品牌常识", "Logo 是企业品牌的重要组成部分。"),
            (0, "03", "--- 03 ---"),
        ),
    ),
    (
        "把促销标题改成英文并使用品牌字体",
        (
            (1, "英文字体规范", "英文标题统一使用 Montserrat SemiBold，字母保持正常间距。"),
            (0, "中文字体规范", "中文正文使用思源黑体 Regular。"),
            (0, "售后说明", "本产品提供一年质保服务。"),
            (0, "字体常识", "字体可以传递品牌情绪。"),
        ),
    ),
    (
        "生成儿童保温杯促销图，突出食品接触安全",
        (
            (1, "儿童杯材质", "采用 316L 食品接触级不锈钢内胆，不含 BPA。"),
            (0, "成人杯包装", "成人商务保温杯采用深灰色礼盒包装。"),
            (0, "饮水常识", "杯子是日常饮水用品。"),
            (0, "乱码", "锟斤拷锟斤拷 000"),
        ),
    ),
    (
        "把这张商品图整体提亮一点，其他内容不变",
        (
            (0, "物流说明", "偏远地区需要增加配送时间。"),
            (0, "品牌颜色", "品牌标准红为 #D71920。"),
            (0, "售后说明", "客服电话工作时间为九点至十八点。"),
            (0, "1.00", "1.00"),
        ),
    ),
    (
        "背景改成品牌红色",
        (
            (1, "品牌色彩规范", "标准红为 #D71920，适用于促销背景。"),
            (0, "品牌辅助色", "辅助蓝为 #0057B8，用于科技类页面。"),
            (0, "颜色常识", "红色通常代表热情。"),
            (0, "商品参数", "产品净重 500 克。"),
        ),
    ),
    (
        "生成咖啡豆促销图，突出深烘焙与巧克力风味",
        (
            (1, "深烘焙风味", "深烘焙咖啡豆具有黑巧克力、焦糖和坚果风味，适合浓缩咖啡。"),
            (0, "浅烘焙风味", "浅烘焙咖啡豆突出柑橘和花香风味。"),
            (0, "营业时间", "咖啡门店每日八点营业。"),
            (0, "咖啡常识", "咖啡是一种常见饮品。"),
        ),
    ),
    (
        "裁成 1:1 商品主图，商品保持居中",
        (
            (1, "平台主图规范", "画幅为 1:1，商品主体居中并占画面约 75%。"),
            (0, "详情页规范", "详情页长图建议采用 3:4 比例。"),
            (0, "主图常识", "商品主图用于展示商品。"),
            (0, "页码", "第 12 页"),
        ),
    ),
    (
        "去掉图片上的临时水印，其他内容不变",
        (
            (0, "会员说明", "会员积分每消费一元累计一分。"),
            (0, "包装说明", "商品包装使用可回收纸张。"),
            (0, "水印常识", "水印是一种版权标识。"),
            (0, "编号", "0000"),
        ),
    ),
)


def build_candidate(index, title, content):
    return {
        "source_id": "calibration",
        "chunk_id": f"candidate-{index}",
        "title": title,
        "content": content,
        "chunk_type": "paragraph",
        "category": "电商视觉",
        "score": 0.0,
    }


def collect_scores(reranker):
    rows = []
    filtered_count = 0
    for query, labeled_documents in CALIBRATION_CASES:
        candidates = []
        labels = {}
        for index, (label, title, content) in enumerate(labeled_documents):
            if not is_meaningful_knowledge_text(content, title):
                filtered_count += 1
                continue
            candidate = build_candidate(index, title, content)
            candidates.append(candidate)
            labels[candidate["chunk_id"]] = label
        for candidate in reranker.rerank(query, candidates, top_n=len(candidates)):
            rows.append(
                {
                    "query": query,
                    "chunk_id": candidate["chunk_id"],
                    "label": labels[candidate["chunk_id"]],
                    "score": candidate["rerank_score"],
                    "title": candidate["title"],
                }
            )
    return rows, filtered_count


def select_threshold(rows, minimum_precision=0.95):
    best = None
    positive_scores = [item["score"] for item in rows if item["label"] == 1]
    negative_scores = [item["score"] for item in rows if item["label"] == 0]
    for step in range(1, 100):
        threshold = step / 100
        true_positive = sum(item["label"] == 1 and item["score"] >= threshold for item in rows)
        false_positive = sum(item["label"] == 0 and item["score"] >= threshold for item in rows)
        false_negative = sum(item["label"] == 1 and item["score"] < threshold for item in rows)
        true_negative = sum(item["label"] == 0 and item["score"] < threshold for item in rows)
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 1.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        f_half = (
            1.25 * precision * recall / (0.25 * precision + recall)
            if precision + recall
            else 0.0
        )
        candidate = {
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f0_5": f_half,
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "true_negative": true_negative,
        }
        separation_margin = min(
            threshold - max(negative_scores),
            min(positive_scores) - threshold,
        )
        rank = (
            precision >= minimum_precision,
            f_half,
            precision,
            recall,
            separation_margin,
            -threshold,
        )
        if best is None or rank > best[0]:
            best = (rank, candidate)
    return best[1]


def main():
    reranker = QwenReranker(read_config(".env"))
    rows, filtered_count = collect_scores(reranker)
    selected = select_threshold(rows)
    positives = [item["score"] for item in rows if item["label"] == 1]
    negatives = [item["score"] for item in rows if item["label"] == 0]
    result = {
        "model": reranker.model,
        "case_count": len(CALIBRATION_CASES),
        "evaluated_candidate_count": len(rows),
        "quality_filtered_count": filtered_count,
        "positive_score_range": [min(positives), max(positives)],
        "negative_score_range": [min(negatives), max(negatives)],
        "selected": selected,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
