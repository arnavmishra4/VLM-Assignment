import os
import json
import re
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import PeftModel
from qwen_vl_utils import process_vision_info

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
OPERATION_CLASSES = [
    "Box Setup", "Inner Packing", "Tape", "Put Items",
    "Pack", "Wrap", "Label", "Final Check", "Idle", "Unknown"
]

PROCEDURAL_NEXT = {
    "Box Setup":     "Inner Packing",
    "Inner Packing": "Tape",
    "Tape":          "Put Items",
    "Put Items":     "Pack",
    "Pack":          "Wrap",
    "Wrap":          "Label",
    "Label":         "Final Check",
    "Final Check":   "Idle",
    "Idle":          "Box Setup",
    "Unknown":       "Unknown",
}

MODEL_ID   = "Qwen/Qwen2-VL-2B-Instruct"
BATCH_SIZE = 2   # start at 2; increase to 4 if VRAM allows

SYSTEM_PROMPT = """You are an expert warehouse operations analyst.
Given video frames from a 5-second packaging clip, respond ONLY with valid JSON:
{
  "dominant_operation": "<one of: Box Setup, Inner Packing, Tape, Put Items, Pack, Wrap, Label, Final Check, Idle, Unknown>",
  "temporal_segment": {"start_frame": <int>, "end_frame": <int>},
  "anticipated_next_operation": "<one of the same classes>",
  "confidence": <float 0-1>
}
No explanation. JSON only."""

# 4-bit config — used for BOTH base and fine-tuned to avoid OOM on T4
BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)


# ──────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ──────────────────────────────────────────────────────────────────────────────
def load_base_model():
    print("Loading BASE model (4-bit) …")
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        quantization_config=BNB_CONFIG,
        device_map="auto",
    )
    model.eval()
    return model


def load_finetuned_model(checkpoint_path: str):
    print(f"Loading FINE-TUNED model from {checkpoint_path} …")
    base = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        quantization_config=BNB_CONFIG,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, checkpoint_path)
    model.eval()
    return model


# ──────────────────────────────────────────────────────────────────────────────
# INFERENCE  (batched)
# ──────────────────────────────────────────────────────────────────────────────
def build_messages(pair: dict) -> list:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                *[{"type": "image", "image": f"file://{Path(fp).resolve()}"}
                  for fp in pair["sampled_frame_paths"]],
                {"type": "text",
                 "text": f"Clip ID: {pair['clip_id']}\nAnalyze these frames."},
            ],
        },
    ]


def run_batch_inference(model, processor, pairs: list) -> list:
    """
    Run inference on a batch of pairs.
    Returns list of parsed dicts in same order as pairs.
    """
    all_messages = [build_messages(p) for p in pairs]

    texts = [
        processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in all_messages
    ]

    # Collect all images in order — process_vision_info returns per-message images
    all_images = []
    for m in all_messages:
        imgs, _ = process_vision_info(m)
        all_images.extend(imgs)

    inputs = processor(
        text=texts,
        images=all_images,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=80,      # JSON response ≈ 80 tokens max
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    input_len = inputs["input_ids"].shape[1]
    results = []
    for out in output_ids:
        raw = processor.decode(out[input_len:], skip_special_tokens=True)
        results.append(parse_output(raw))

    return results


def parse_output(raw: str) -> dict:
    """Extract JSON from model output, return safe defaults on failure."""
    try:
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception:
        pass
    return {
        "dominant_operation":         "Unknown",
        "temporal_segment":           {"start_frame": 0, "end_frame": 0},
        "anticipated_next_operation": "Unknown",
        "confidence":                 0.0,
    }


# ──────────────────────────────────────────────────────────────────────────────
# METRICS
# ──────────────────────────────────────────────────────────────────────────────
def compute_tiou(pred_seg: dict, gt_seg: dict) -> float:
    p_start = pred_seg.get("start_frame", 0)
    p_end   = pred_seg.get("end_frame",   0)
    g_start = gt_seg.get("start_frame",   0)
    g_end   = gt_seg.get("end_frame",     0)

    if p_end <= p_start or g_end <= g_start:
        return 0.0

    intersection = max(0, min(p_end, g_end) - max(p_start, g_start))
    union = (p_end - p_start) + (g_end - g_start) - intersection
    return intersection / union if union > 0 else 0.0


def evaluate_model(model, processor, pairs: list) -> dict:
    """Batched evaluation loop — returns per-clip results + aggregate metrics."""
    results   = []
    all_preds = []

    # Run batched inference
    for i in range(0, len(pairs), BATCH_SIZE):
        batch = pairs[i:i + BATCH_SIZE]
        print(f"  Inferring clips {i+1}–{min(i+BATCH_SIZE, len(pairs))} / {len(pairs)} …",
              flush=True)
        try:
            preds = run_batch_inference(model, processor, batch)
        except RuntimeError as e:
            # OOM fallback — run one by one
            print(f"  ⚠️  Batch OOM, falling back to single inference: {e}")
            preds = []
            for pair in batch:
                try:
                    p = run_batch_inference(model, processor, [pair])
                    preds.extend(p)
                except Exception:
                    preds.append(parse_output(""))
        all_preds.extend(preds)

    # Score each clip
    for pair, pred in zip(pairs, all_preds):
        target = json.loads(pair["target_json"]) if isinstance(pair["target_json"], str) \
                 else pair["target_json"]

        gt_op   = target.get("dominant_operation", "Unknown")
        gt_next = target.get("anticipated_next_operation",
                             PROCEDURAL_NEXT.get(gt_op, "Unknown"))
        gt_seg  = target.get("temporal_segment", {"start_frame": 0, "end_frame": 125})

        pred_op   = pred.get("dominant_operation", "Unknown")
        pred_next = pred.get("anticipated_next_operation", "Unknown")
        pred_seg  = pred.get("temporal_segment", {"start_frame": 0, "end_frame": 0})

        tiou   = compute_tiou(pred_seg, gt_seg)
        oca_ok = int(pred_op.strip().lower() == gt_op.strip().lower())
        aa_ok  = int(pred_next.strip().lower() == gt_next.strip().lower())

        status = "✅" if (oca_ok and tiou >= 0.5 and aa_ok) else \
                 "⚠️"  if (oca_ok or  tiou >= 0.5 or  aa_ok) else "❌"
        print(f"  {status}  [{pair['clip_id']}]  OCA={oca_ok}  tIoU={tiou:.2f}  AA={aa_ok}")

        results.append({
            "clip_id":    pair["clip_id"],
            "gt_op":      gt_op,   "pred_op":   pred_op,
            "gt_next":    gt_next, "pred_next": pred_next,
            "gt_seg":     gt_seg,  "pred_seg":  pred_seg,
            "tiou":       tiou,
            "oca_ok":     oca_ok,
            "aa_ok":      aa_ok,
            "confidence": pred.get("confidence", 0.0),
        })

    oca        = np.mean([r["oca_ok"]        for r in results])
    tiou_score = np.mean([r["tiou"] >= 0.5   for r in results])
    aa         = np.mean([r["aa_ok"]         for r in results])

    return {
        "aggregate": {
            "OCA":      round(float(oca),        4),
            "tIoU@0.5": round(float(tiou_score), 4),
            "AA@1":     round(float(aa),         4),
        },
        "per_clip": results,
    }


# ──────────────────────────────────────────────────────────────────────────────
# VISUALISATION
# ──────────────────────────────────────────────────────────────────────────────
COLORS = {
    "base":      "#6c757d",
    "finetuned": "#0d6efd",
    "good":      "#198754",
    "bad":       "#dc3545",
    "accent":    "#fd7e14",
}


def plot_comparison(base_res: dict, ft_res: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics      = ["OCA", "tIoU@0.5", "AA@1"]
    descriptions = ["Operation Classification\nAccuracy",
                    "Temporal IoU\n(≥0.5 threshold)",
                    "Anticipation Accuracy\n(next operation)"]
    b_vals = [base_res["aggregate"][m] for m in metrics]
    f_vals = [ft_res["aggregate"][m]   for m in metrics]

    # ── 1. Bar chart ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Base vs Fine-Tuned — Aggregate Metrics", fontsize=15, fontweight="bold")
    for ax, metric, desc, bv, fv in zip(axes, metrics, descriptions, b_vals, f_vals):
        bars = ax.bar(["Base", "Fine-tuned"], [bv, fv],
                      color=[COLORS["base"], COLORS["finetuned"]],
                      width=0.45, edgecolor="white", linewidth=1.5)
        ax.set_ylim(0, 1.0)
        ax.set_title(f"{metric}\n{desc}", fontsize=10)
        ax.set_ylabel("Score")
        ax.axhline(1/9, color=COLORS["bad"],  linestyle="--", linewidth=1.2, label="Random (1/9)")
        ax.axhline(0.5, color=COLORS["good"], linestyle=":",  linewidth=1.2, label="Target (0.5)")
        for bar, val in zip(bars, [bv, fv]):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                    f"{val:.2%}", ha="center", va="bottom", fontsize=11, fontweight="bold")
        delta = fv - bv
        ax.text(0.5, 0.92, f"Δ {delta:+.2%}", transform=ax.transAxes,
                ha="center", fontsize=11, fontweight="bold",
                color=COLORS["good"] if delta >= 0 else COLORS["bad"])
        ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(out_dir / "metric_comparison_bar.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✅ Saved metric_comparison_bar.png")

    # ── 2. Per-clip tIoU ─────────────────────────────────────────────────
    clips    = [r["clip_id"] for r in base_res["per_clip"]]
    b_tious  = [r["tiou"]    for r in base_res["per_clip"]]
    ft_tious = [r["tiou"]    for r in ft_res["per_clip"]]
    x        = np.arange(len(clips))

    fig, ax = plt.subplots(figsize=(max(10, len(clips)*0.6), 5))
    ax.plot(x, b_tious,  "o--", color=COLORS["base"],      label="Base",       linewidth=1.5, markersize=6)
    ax.plot(x, ft_tious, "s-",  color=COLORS["finetuned"], label="Fine-tuned", linewidth=2,   markersize=7)
    ax.axhline(0.5, color=COLORS["accent"], linestyle="--", linewidth=1.5, label="tIoU@0.5 threshold")
    ax.fill_between(x, b_tious, ft_tious,
                    where=[f >  b for b, f in zip(b_tious, ft_tious)],
                    alpha=0.15, color=COLORS["good"], label="FT improves")
    ax.fill_between(x, b_tious, ft_tious,
                    where=[f <= b for b, f in zip(b_tious, ft_tious)],
                    alpha=0.15, color=COLORS["bad"],  label="Base wins")
    ax.set_xticks(x)
    ax.set_xticklabels([c.split("_")[-1] for c in clips], rotation=45, ha="right", fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Clip ID (suffix)")
    ax.set_ylabel("tIoU")
    ax.set_title("Per-Clip Temporal IoU: Base vs Fine-Tuned", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "tiou_per_clip.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✅ Saved tiou_per_clip.png")

    # ── 3. Per-op accuracy ────────────────────────────────────────────────
    from collections import defaultdict
    ops = sorted(set(r["gt_op"] for r in base_res["per_clip"] + ft_res["per_clip"]))

    def op_accuracy(per_clip):
        correct, total = defaultdict(int), defaultdict(int)
        for r in per_clip:
            total[r["gt_op"]]   += 1
            correct[r["gt_op"]] += r["oca_ok"]
        return {op: correct[op]/total[op] if total[op] else 0 for op in ops}

    b_acc  = op_accuracy(base_res["per_clip"])
    ft_acc = op_accuracy(ft_res["per_clip"])
    xo     = np.arange(len(ops))

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(xo - 0.19, [b_acc[o]  for o in ops], 0.38,
           color=COLORS["base"],      label="Base",       edgecolor="white")
    ax.bar(xo + 0.19, [ft_acc[o] for o in ops], 0.38,
           color=COLORS["finetuned"], label="Fine-tuned", edgecolor="white")
    ax.set_xticks(xo)
    ax.set_xticklabels(ops, rotation=30, ha="right")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("OCA per class")
    ax.set_title("Per-Operation Classification Accuracy", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "oca_per_operation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✅ Saved oca_per_operation.png")

    # ── 4. Radar chart ────────────────────────────────────────────────────
    labels  = ["OCA", "tIoU@0.5", "AA@1"]
    angles  = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    b_vals_r  = [base_res["aggregate"][m] for m in labels] + [base_res["aggregate"][labels[0]]]
    ft_vals_r = [ft_res["aggregate"][m]   for m in labels] + [ft_res["aggregate"][labels[0]]]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, b_vals_r,  "o-", linewidth=2, color=COLORS["base"],      label="Base")
    ax.fill(angles, b_vals_r,  alpha=0.15, color=COLORS["base"])
    ax.plot(angles, ft_vals_r, "s-", linewidth=2, color=COLORS["finetuned"], label="Fine-tuned")
    ax.fill(angles, ft_vals_r, alpha=0.20, color=COLORS["finetuned"])
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_title("Model Capability Radar", fontsize=13, fontweight="bold", pad=18)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    fig.savefig(out_dir / "radar_profile.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✅ Saved radar_profile.png")

    # ── 5. Dashboard ──────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("OpenPack VLM Evaluation Dashboard\nBase vs QLoRA Fine-Tuned (Qwen2-VL-2B)",
                 fontsize=14, fontweight="bold")
    gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    ax1 = fig.add_subplot(gs[0, 0])
    xp  = np.arange(len(metrics))
    ax1.bar(xp - 0.2, b_vals, 0.38, color=COLORS["base"],      label="Base")
    ax1.bar(xp + 0.2, f_vals, 0.38, color=COLORS["finetuned"], label="Fine-tuned")
    ax1.set_xticks(xp); ax1.set_xticklabels(metrics)
    ax1.set_ylim(0, 1); ax1.legend(fontsize=8)
    ax1.set_title("Aggregate Metrics", fontweight="bold")
    for xi, (bv, fv) in enumerate(zip(b_vals, f_vals)):
        ax1.text(xi-0.2, bv+0.02, f"{bv:.0%}", ha="center", fontsize=7)
        ax1.text(xi+0.2, fv+0.02, f"{fv:.0%}", ha="center", fontsize=7,
                 color=COLORS["finetuned"], fontweight="bold")

    ax2     = fig.add_subplot(gs[0, 1])
    deltas  = [fv - bv for bv, fv in zip(b_vals, f_vals)]
    ax2.bar(metrics, deltas,
            color=[COLORS["good"] if d >= 0 else COLORS["bad"] for d in deltas],
            edgecolor="white")
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_title("Delta (FT − Base)", fontweight="bold")
    ax2.set_ylabel("Improvement")
    for i, d in enumerate(deltas):
        ax2.text(i, d + (0.005 if d >= 0 else -0.015),
                 f"{d:+.2%}", ha="center", fontsize=9, fontweight="bold")

    ax3 = fig.add_subplot(gs[0, 2], polar=True)
    ax3.plot(angles, b_vals_r,  "o-", color=COLORS["base"],      linewidth=1.5, label="Base")
    ax3.fill(angles, b_vals_r,  alpha=0.12, color=COLORS["base"])
    ax3.plot(angles, ft_vals_r, "s-", color=COLORS["finetuned"], linewidth=2,   label="Fine-tuned")
    ax3.fill(angles, ft_vals_r, alpha=0.18, color=COLORS["finetuned"])
    ax3.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=9)
    ax3.set_ylim(0, 1)
    ax3.legend(fontsize=7, loc="upper right", bbox_to_anchor=(1.3, 1.1))
    ax3.set_title("Radar", fontweight="bold", pad=14)

    ax4 = fig.add_subplot(gs[1, :])
    x2  = np.arange(len(clips))
    ax4.plot(x2, b_tious,  "o--", color=COLORS["base"],      label="Base",       linewidth=1.5, markersize=5)
    ax4.plot(x2, ft_tious, "s-",  color=COLORS["finetuned"], label="Fine-tuned", linewidth=2,   markersize=6)
    ax4.axhline(0.5, color=COLORS["accent"], linestyle="--", linewidth=1.2, label="0.5 threshold")
    ax4.fill_between(x2, b_tious, ft_tious,
                     where=[f > b for b, f in zip(b_tious, ft_tious)],
                     alpha=0.12, color=COLORS["good"])
    ax4.set_xticks(x2)
    ax4.set_xticklabels([c.split("_")[-1] for c in clips], rotation=45, ha="right", fontsize=7)
    ax4.set_ylim(0, 1.05); ax4.set_ylabel("tIoU"); ax4.legend(fontsize=9)
    ax4.set_title("Per-Clip tIoU", fontweight="bold")

    fig.savefig(out_dir / "dashboard.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✅ Saved dashboard.png")


# ──────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────────────────────
def load_pairs(dataset_path: Path, frames_path: Path, max_clips: int = 20) -> list:
    pairs = []
    for json_file in sorted(dataset_path.glob("sample_*.json"))[:max_clips]:
        with open(json_file) as f:
            pair = json.load(f)
        pair["sampled_frame_paths"] = [
            str(frames_path / Path(fp.replace("\\", "/")).name)
            for fp in pair["sampled_frame_paths"]
        ]
        pairs.append(pair)
    print(f"Loaded {len(pairs)} evaluation pairs")
    return pairs


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main():
    DATASET_PATH         = Path("/kaggle/input/datasets/arnavmishra6996/vlm-qlora-dataset/training_data_samples")
    FRAMES_PATH          = Path("/kaggle/input/datasets/arnavmishra6996/vlm-qlora-dataset/S0500")
    FINETUNED_CHECKPOINT = "/kaggle/working/checkpoints/final"
    OUTPUT_DIR           = Path("/kaggle/working/eval_results")
    MAX_CLIPS            = 20

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pairs     = load_pairs(DATASET_PATH, FRAMES_PATH, MAX_CLIPS)
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    # ── Base model ────────────────────────────────────────────────────────
    print("\n═══ Evaluating BASE model ═══")
    base_model   = load_base_model()
    base_results = evaluate_model(base_model, processor, pairs)
    del base_model
    torch.cuda.empty_cache()

    # ── Fine-tuned model ──────────────────────────────────────────────────
    print("\n═══ Evaluating FINE-TUNED model ═══")
    ft_model   = load_finetuned_model(FINETUNED_CHECKPOINT)
    ft_results = evaluate_model(ft_model, processor, pairs)
    del ft_model
    torch.cuda.empty_cache()

    # ── Save results.json ─────────────────────────────────────────────────
    results_json = {
        "base_model":      base_results["aggregate"],
        "finetuned_model": ft_results["aggregate"],
        "per_clip_base":   base_results["per_clip"],
        "per_clip_ft":     ft_results["per_clip"],
    }
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"\n✅ results.json saved to {OUTPUT_DIR / 'results.json'}")

    # ── Plots ─────────────────────────────────────────────────────────────
    print("\n═══ Generating visualisations ═══")
    plot_comparison(base_results, ft_results, OUTPUT_DIR)

    # ── Summary table ─────────────────────────────────────────────────────
    print("\n" + "═"*52)
    print(f"{'Metric':<15} {'Base':>10} {'Fine-Tuned':>12} {'Delta':>10}")
    print("─"*52)
    for m in ["OCA", "tIoU@0.5", "AA@1"]:
        bv  = base_results["aggregate"][m]
        fv  = ft_results["aggregate"][m]
        d   = fv - bv
        sym = "▲" if d > 0 else "▼"
        print(f"{m:<15} {bv:>10.2%} {fv:>12.2%} {sym}{abs(d):>8.2%}")
    print("═"*52)

    aa_delta = ft_results["aggregate"]["AA@1"] - base_results["aggregate"]["AA@1"]
    if aa_delta > 0.10:
        print("✅ Model learned procedural grammar (AA@1 improved > 10pp)")
    else:
        print("⚠️  Temporal learning weak — consider more epochs or boundary-focused clips")


if __name__ == "__main__":
    main()
