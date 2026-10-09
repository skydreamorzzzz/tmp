#!/usr/bin/env python3
"""Append the completed factorial ablation results to the reproduction docs."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "artifacts/dpmotion_reproduction/ablations/full_2x2"
REPORT = ROOT / "docs/DPMOTION_REPRO_REPORT.md"
NOTES = ROOT / "docs/REPRO_ENGINEERING_NOTES.md"
MARKER = "## 2026-10-08：完整 RAG × guidance 2×2 消融"

def fmt(x, digits=6):
    return f"{x:.{digits}f}"

def main():
    repo = json.loads((EXP / "repo_ablation_comparison.json").read_text())
    clip = json.loads((EXP / "interclip_alignment.json").read_text())
    config = json.loads((EXP / "config.json").read_text())
    if config.get("status") != "complete":
        raise RuntimeError("generation is not marked complete")
    if any(repo["num_scenes"][c] != 120 for c in repo["conditions"]):
        raise RuntimeError("repo evaluator did not score all 120 scenes per condition")
    scores = clip["condition_scores"]
    if any(scores[c]["n_scenes"] != 120 for c in scores):
        raise RuntimeError("InterCLIP evaluator did not score all 120 scenes per condition")

    conds = ["A_no_rag_no_guidance", "B_rag_no_guidance",
             "C_no_rag_guidance", "D_rag_guidance"]
    labels = ["A InterGen", "B +RAG", "C +guidance", "D +RAG +guidance"]
    lines = [MARKER, "",
             "固定 120 个 test scenes，seed=`42+file_id`，每场150帧，原生 InterGen `forward_single`、1000-step ancestral DDPM、距离3m、distance weight 5。no-RAG 时 Group2 使用 Group1 测试描述；RAG 时 Group2 使用固定检索候选。四臂每个 case 都重置同一 seed。没有训练或改模型/参数。",
             "",
             "| 条件 | PeneBone depth sum (m, mean ± std) | penetration rate (raw fraction) | CFrame (%) | Contact (pairs/frame) | intra-scene diversity | Group2 InterCLIP matching distance ↓ | paired cosine ↑ | R-Precision R1 / R2 / R3 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for label, cond in zip(labels, conds):
        m = repo["conditions"][cond]
        i = scores[cond]["group2_conditioned_text_alignment"]
        lines.append("| {} | {} ± {} | {} | {} | {} | {} ± {} | {} | {} | {:.3f} / {:.3f} / {:.3f} |".format(
            label, fmt(m["penebone_mean"]), fmt(m["penebone_std"]),
            fmt(m["pene_rate_mean"]), fmt(m["cframe_all_mean"]), fmt(m["contact_all_mean"]),
            fmt(m["intra_scene_diversity_mean"]), fmt(m["intra_scene_diversity_std"]),
            fmt(i["matching_distance_mean"]), fmt(i["paired_cosine_similarity_mean"]),
            i["r_precision_top1"], i["r_precision_top2"], i["r_precision_top3"]))

    lines += ["", "### 配对效应", "",
              "InterCLIP Group2 的差值按同一 scene 配对；距离差 `<0` 表示 RAG/处理组更接近其条件文本。几何指标的正负方向按指标含义解释，下面报告效应量和 bootstrap 95% 区间。",
              "",
              "| 指标 / contrast | mean Δ | 95% bootstrap CI | per-scene directional fraction |",
              "|---|---:|---:|---:|"]
    for name, label in [("RAG_effect_without_guidance", "InterCLIP distance: RAG effect, guidance off"),
                        ("RAG_effect_with_guidance", "InterCLIP distance: RAG effect, guidance on"),
                        ("guidance_effect_without_RAG", "InterCLIP distance: guidance effect, RAG off"),
                        ("guidance_effect_with_RAG", "InterCLIP distance: guidance effect, RAG on")]:
        row = clip["paired_contrasts"][name]
        ci = row.get("bootstrap_95pct_ci", [float("nan"), float("nan")])
        lines.append("| {} | {} | [{}, {}] | {:.1%} lower |".format(
            label, fmt(row["distance_treated_minus_control_mean"]),
            fmt(ci[0]), fmt(ci[1]),
            row["fraction_treated_lower_distance"]))
    lines += ["", "### 几何指标的 factorial effects", "",
              "每个 effect 均为同场景配对差；RAG main effect 是 guidance off/on 两个 RAG 差值的平均，guidance main effect 同理，interaction 是两种 guidance 下 RAG 差值之差。CI 为 scene bootstrap 95%。",
              "",
              "| 指标 | RAG main Δ (95% CI) | guidance main Δ (95% CI) | RAG×guidance Δ (95% CI) |",
              "|---|---:|---:|---:|"]
    for metric in ["penebone", "pene_rate", "cframe_all", "contact_all", "diversity"]:
        effects = repo["per_scene_factorial_contrasts"][metric]
        rag, guidance, interaction = (effects[key] for key in
            ("rag_main_effect", "guidance_main_effect", "rag_guidance_interaction"))
        lines.append("| {} | {} [{}; {}] | {} [{}; {}] | {} [{}; {}] |".format(
            metric, fmt(rag["mean"]), fmt(rag["bootstrap_95pct_ci"][0]), fmt(rag["bootstrap_95pct_ci"][1]),
            fmt(guidance["mean"]), fmt(guidance["bootstrap_95pct_ci"][0]), fmt(guidance["bootstrap_95pct_ci"][1]),
            fmt(interaction["mean"]), fmt(interaction["bootstrap_95pct_ci"][0]), fmt(interaction["bootstrap_95pct_ci"][1])))

    off = clip["paired_r_precision_contrasts"]["RAG_effect_without_guidance"]["R@1"]
    on = clip["paired_r_precision_contrasts"]["RAG_effect_with_guidance"]["R@1"]
    lines += ["", "### RAG 的正向信号与限制", "",
              "在本次固定 120 条 Group2 文本/动作集合的 InterCLIP 120-way 配对 rank 诊断中，R@1 的逐 scene 命中率变化为：guidance off `+{:.3f}`（bootstrap 95% CI [{:.3f}, {:.3f}]），guidance on `+{:.3f}`（[{:.3f}, {:.3f}]）。这是当前实验中支持 RAG 的正向信号，但它只表示各条件自己的 120-way 文本/动作候选池中正配对排名改善；R@2/R@3 区间跨 0。".format(
                  off["mean_delta_recall_at_k"], *off["bootstrap_95pct_ci"],
                  on["mean_delta_recall_at_k"], *on["bootstrap_95pct_ci"]),
              "直接的 Group2 matching distance 并未改善：RAG 使均值分别增加 `+0.0113` 和 `+0.0078`（两者 95% CI 均跨 0），且只有 48.3% scene 的距离下降。因此不能把结果概括成普遍的文本动作对齐提升。KB 与 test split 重叠；120 个 query 有 107 个 top-1 原文命中。该 R@1 信号不等于无泄漏泛化收益，也不是论文完整 multi-group R-Precision protocol。", ""]

    lines += ["", "InterCLIP Group2 是 RAG 敏感的文本动作对齐诊断；120-way R-Precision 使用各条件自己的文本/动作集合，不等同论文完整 multi-group benchmark。几何 evaluator 的 PeneBone、penetration rate 和 intra-scene diversity 按仓库定义逐 scene 配对；CFrame/Contact 是仓库附加诊断。Δ>0 表示数值增加，不统一代表变好。完整 per-case 数值、所有 simple/main/interaction factorial contrasts 和 evaluator 中间特征见 `artifacts/dpmotion_reproduction/ablations/full_2x2/`。",
              "",
              "**数据边界：** RAG 知识库由全部 InterHuman annotations 构建，包含 test split；这是已证实的检索库/test 重叠。当前 120 query 中有 107 条 top-1 与 query 原文相同。此消融能说明在该泄漏设置下 RAG 的成对效应，不能作为无泄漏泛化收益证据。no-RAG 的 Group2 复用 Group1 文本，因此这是“替换为检索文本”的因果对照；它不会隔离出独立于文本差异的抽象检索机制。",
              "",
              "四臂最终是否出现正向 RAG 作用，以表中的 paired InterCLIP 和 geometry 结果为准；不因单个指标的有利方向宣称整体质量提高。",
              "分层看几何指标，RAG 在 guidance off 时降低了 PeneBone `−8.343`（95% CI [−12.207, −4.430]）、penetration rate `−0.001249`（[−0.001803, −0.000689]）、CFrame `−8.894` 个百分点和 Contact `−3.435` pairs/frame；guidance on 时这些 RAG 简单效应的区间均跨 0。RAG×guidance interaction 区间在这些指标上不跨 0，表明 RAG 的几何变化依赖 guidance 是否开启。",
              "**最可信结论：** 在此有 test overlap 的设置中，RAG 提高了本地 120-way InterCLIP R@1，并在 guidance off 时改善仓库几何穿透诊断；但 guidance on 下几何效应不明确，直接 matching distance 没改善，组内多样性变化也不稳定。把它表述为“检索文本改变了组2，并在部分当前诊断上有正向信号”是可信的；称为整体生成质量提升或无泄漏泛化证据则超出数据支持。",
              ""]
    block = "\n".join(lines)
    report_text = REPORT.read_text()
    if MARKER in report_text:
        report_text = report_text.split(MARKER, 1)[0].rstrip()
    REPORT.write_text(report_text + "\n\n" + block)
    notes_text = NOTES.read_text()
    note_marker = "## 2026-10-08：完整 RAG × guidance 2×2 消融完成"
    if note_marker in notes_text:
        notes_text = notes_text.split(note_marker, 1)[0].rstrip()
    notes_block = [note_marker, "",
        "- 120 个 test scenes × A/B/C/D 四臂全部生成；完整性为 480/480 scene-condition，指标见 `ablations/full_2x2/`。",
        "- A=no-RAG/no-guidance；B=RAG/no-guidance；C=no-RAG/guidance；D=RAG+guidance。seed=`42+file_id`，每场150帧、DDPM1000、distance=3m、weight=5。",
        "- RAG KB 与 test split 重叠是已证实的数据泄漏风险；120 个 query 有 107 个 top-1 原文命中。InterCLIP Group2 matching distance 没有正向变化，但本次非论文 protocol 的 120-way R@1 有改善；两者边界见主报告。",
        "- no-guidance 通过 runner 诊断 hook 将 guidance gradient 置零；上游模型和采样代码未改。",
        "- guidance 条件生成较慢且出现数次进度长尾；进程可恢复且 480/480 完成。单次观察到 NFS I/O wait，不能据此归因所有延迟。", ""]
    notes_block.insert(-1, "- 首次报告汇总因 evaluator contrast key 大小写不一致抛出 `KeyError`；几何与 InterCLIP 指标已成功生成，修正汇总映射后完整写入报告。未重跑生成或改动指标。")
    NOTES.write_text(notes_text + "\n\n" + "\n".join(notes_block))
    print("FULL_ABLATION_DOCS_FINALIZED")

if __name__ == "__main__":
    main()
