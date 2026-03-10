import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def _to_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _load_rows(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _metric_template():
    return {
        "support": 0,
        "correct": 0,
        "avg_pred_conf_sum": 0.0,
        "avg_prompt_prob_sum": 0.0,
        "prompt_prob_count": 0,
        "pred_counter": Counter(),
    }


def summarize_run(path):
    rows = _load_rows(path)
    prompted = [r for r in rows if str(r.get("prompt_label", "")).strip()]
    labels = sorted({str(r["prompt_label"]).strip() for r in prompted if str(r["prompt_label"]).strip()})

    per_label = defaultdict(_metric_template)
    total_correct = 0
    total_prompted = 0
    confusion = Counter()

    for r in prompted:
        p = str(r.get("prompt_label", "")).strip()
        y = str(r.get("pred_label", "")).strip()
        if not p:
            continue
        total_prompted += 1
        m = per_label[p]
        m["support"] += 1
        m["pred_counter"][y] += 1
        m["avg_pred_conf_sum"] += _to_float(r.get("pred_conf", 0.0))
        prompt_prob_col = f"fused_{p}"
        if prompt_prob_col in r and str(r.get(prompt_prob_col, "")).strip():
            m["avg_prompt_prob_sum"] += _to_float(r.get(prompt_prob_col, 0.0))
            m["prompt_prob_count"] += 1
        if y == p:
            m["correct"] += 1
            total_correct += 1
        confusion[(p, y)] += 1

    per_label_out = {}
    recalls = []
    for label in labels:
        m = per_label[label]
        support = int(m["support"])
        correct = int(m["correct"])
        recall = (correct / support) if support > 0 else 0.0
        recalls.append(recall)

        # Most common wrong prediction
        wrong = [(k, v) for k, v in m["pred_counter"].items() if k != label]
        wrong.sort(key=lambda x: x[1], reverse=True)
        top_confusion = wrong[0][0] if wrong else ""
        top_confusion_count = int(wrong[0][1]) if wrong else 0
        top_confusion_rate = (top_confusion_count / support) if support > 0 else 0.0

        avg_pred_conf = m["avg_pred_conf_sum"] / max(support, 1)
        if m["prompt_prob_count"] > 0:
            avg_prompt_prob = m["avg_prompt_prob_sum"] / m["prompt_prob_count"]
        else:
            avg_prompt_prob = 0.0

        per_label_out[label] = {
            "support": support,
            "recall": recall,
            "avg_pred_conf": avg_pred_conf,
            "avg_prompt_prob": avg_prompt_prob,
            "top_confusion": top_confusion,
            "top_confusion_rate": top_confusion_rate,
        }

    overall_accuracy = (total_correct / total_prompted) if total_prompted > 0 else 0.0
    balanced_accuracy = (sum(recalls) / len(recalls)) if recalls else 0.0

    return {
        "path": str(Path(path)),
        "rows_total": len(rows),
        "rows_prompted": total_prompted,
        "labels": labels,
        "overall_accuracy": overall_accuracy,
        "balanced_accuracy": balanced_accuracy,
        "per_label": per_label_out,
        "confusion_counts": {f"{k[0]}->{k[1]}": int(v) for k, v in confusion.items()},
    }


def compare_runs(base, cand):
    labels = sorted(set(base["labels"]) | set(cand["labels"]))
    per_label_delta = {}
    for label in labels:
        b = base["per_label"].get(label, {})
        c = cand["per_label"].get(label, {})
        per_label_delta[label] = {
            "support_base": int(b.get("support", 0)),
            "support_cand": int(c.get("support", 0)),
            "recall_base": float(b.get("recall", 0.0)),
            "recall_cand": float(c.get("recall", 0.0)),
            "recall_delta": float(c.get("recall", 0.0) - b.get("recall", 0.0)),
            "avg_pred_conf_base": float(b.get("avg_pred_conf", 0.0)),
            "avg_pred_conf_cand": float(c.get("avg_pred_conf", 0.0)),
            "avg_pred_conf_delta": float(c.get("avg_pred_conf", 0.0) - b.get("avg_pred_conf", 0.0)),
            "avg_prompt_prob_base": float(b.get("avg_prompt_prob", 0.0)),
            "avg_prompt_prob_cand": float(c.get("avg_prompt_prob", 0.0)),
            "avg_prompt_prob_delta": float(c.get("avg_prompt_prob", 0.0) - b.get("avg_prompt_prob", 0.0)),
            "top_confusion_base": str(b.get("top_confusion", "")),
            "top_confusion_cand": str(c.get("top_confusion", "")),
            "top_confusion_rate_base": float(b.get("top_confusion_rate", 0.0)),
            "top_confusion_rate_cand": float(c.get("top_confusion_rate", 0.0)),
            "top_confusion_rate_delta": float(
                c.get("top_confusion_rate", 0.0) - b.get("top_confusion_rate", 0.0)
            ),
        }

    return {
        "overall_accuracy_base": float(base["overall_accuracy"]),
        "overall_accuracy_cand": float(cand["overall_accuracy"]),
        "overall_accuracy_delta": float(cand["overall_accuracy"] - base["overall_accuracy"]),
        "balanced_accuracy_base": float(base["balanced_accuracy"]),
        "balanced_accuracy_cand": float(cand["balanced_accuracy"]),
        "balanced_accuracy_delta": float(cand["balanced_accuracy"] - base["balanced_accuracy"]),
        "rows_prompted_base": int(base["rows_prompted"]),
        "rows_prompted_cand": int(cand["rows_prompted"]),
        "per_label": per_label_delta,
    }


def _print_summary(base, cand, delta):
    print("\n=== Baseline ===")
    print(f"file: {base['path']}")
    print(
        f"prompted_rows={base['rows_prompted']} "
        f"acc={base['overall_accuracy']:.3f} "
        f"bal_acc={base['balanced_accuracy']:.3f}"
    )

    print("\n=== Candidate ===")
    print(f"file: {cand['path']}")
    print(
        f"prompted_rows={cand['rows_prompted']} "
        f"acc={cand['overall_accuracy']:.3f} "
        f"bal_acc={cand['balanced_accuracy']:.3f}"
    )

    print("\n=== Delta (candidate - baseline) ===")
    print(
        f"acc_delta={delta['overall_accuracy_delta']:+.3f} "
        f"bal_acc_delta={delta['balanced_accuracy_delta']:+.3f}"
    )

    print("\nPer-gesture deltas:")
    labels_sorted = sorted(
        delta["per_label"].items(),
        key=lambda kv: kv[1]["recall_delta"],
    )
    for label, d in labels_sorted:
        print(
            f"  {label:<12} "
            f"recall {d['recall_base']:.3f}->{d['recall_cand']:.3f} ({d['recall_delta']:+.3f}) | "
            f"prompt_prob {d['avg_prompt_prob_base']:.3f}->{d['avg_prompt_prob_cand']:.3f} "
            f"({d['avg_prompt_prob_delta']:+.3f}) | "
            f"top_conf {d['top_confusion_base'] or '-'}->{d['top_confusion_cand'] or '-'}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Compare two realtime confidence-analysis CSV runs."
    )
    parser.add_argument("--baseline", required=True, help="Baseline CSV path.")
    parser.add_argument("--candidate", required=True, help="Candidate CSV path.")
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional output JSON path to save full comparison artifacts.",
    )
    args = parser.parse_args()

    base = summarize_run(args.baseline)
    cand = summarize_run(args.candidate)
    delta = compare_runs(base, cand)
    _print_summary(base, cand, delta)

    if args.output_json:
        payload = {"baseline": base, "candidate": cand, "delta": delta}
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nSaved JSON report to {out}")


if __name__ == "__main__":
    main()
