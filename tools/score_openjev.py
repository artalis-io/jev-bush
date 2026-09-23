#!/usr/bin/env python3
"""Score Jev/OpenJev JSONL predictions against typed-decisions gold labels."""

import argparse
import json
import math
from pathlib import Path


def distributions(answer, kind):
    if kind == "noul":
        p = answer.get("probabilities")
        if p is None:
            yes = float(answer["noul"])
            p = {"true": yes, "false": 1.0 - yes}
        return {str(k): float(v) for k, v in p.items()}
    return {str(k): float(v) for k, v in answer["probabilities"].items()}


def add_stats(groups, key, hit, soft_ce, hard_ce, row_brier, row_tv):
    stats = groups.setdefault(
        key,
        {
            "decisions": 0,
            "correct": 0,
            "soft_ce": 0.0,
            "label_ce": 0.0,
            "brier": 0.0,
            "tv": 0.0,
        },
    )
    stats["decisions"] += 1
    stats["correct"] += hit
    stats["soft_ce"] += soft_ce
    stats["label_ce"] += hard_ce
    stats["brier"] += row_brier
    stats["tv"] += row_tv


def finish_groups(groups):
    return {
        key: {
            "decisions": stats["decisions"],
            "argmax_agreement": stats["correct"] / stats["decisions"],
            "soft_cross_entropy": stats["soft_ce"] / stats["decisions"],
            "label_log_loss": stats["label_ce"] / stats["decisions"],
            "brier": stats["brier"] / stats["decisions"],
            "total_variation": stats["tv"] / stats["decisions"],
        }
        for key, stats in sorted(groups.items())
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gold")
    parser.add_argument("predictions")
    parser.add_argument("--bins", type=int, default=10)
    ns = parser.parse_args()
    gold_rows = [json.loads(x) for x in Path(ns.gold).read_text(encoding="utf-8").splitlines() if x.strip()]
    predictions = [json.loads(x) for x in Path(ns.predictions).read_text(encoding="utf-8").splitlines() if x.strip()]
    gold_ids = [str(x["id"]) for x in gold_rows]
    prediction_ids = [str(x["id"]) for x in predictions]
    if len(set(gold_ids)) != len(gold_ids):
        raise ValueError("duplicate id in gold data")
    if len(set(prediction_ids)) != len(prediction_ids):
        raise ValueError("duplicate id in predictions")
    if set(prediction_ids) != set(gold_ids):
        missing = sorted(set(gold_ids) - set(prediction_ids))
        extra = sorted(set(prediction_ids) - set(gold_ids))
        raise ValueError(f"prediction id mismatch: missing={missing[:5]} extra={extra[:5]}")
    by_id = {str(x["id"]): x for x in predictions}
    n = correct = 0
    log_loss = label_log_loss = brier = tv = score_abs = 0.0
    score_n = 0
    calibration = [[] for _ in range(ns.bins)]
    kinds = {}
    workflows = {}
    for row in gold_rows:
        rid = str(row["id"])
        pred = by_id[rid]["answers"]
        gold = json.loads(row["gold"]) if isinstance(row["gold"], str) else row["gold"]
        if set(pred) != set(gold):
            raise ValueError(f"answer id mismatch for {rid}")
        workflow = str(row.get("workflow", row.get("category", "unknown")))
        for qid, target in gold.items():
            answer = pred[qid]
            kind = target["type"]
            if answer.get("type") != kind:
                raise ValueError(f"answer type mismatch for {rid}/{qid}")
            pp = distributions(answer, kind)
            gp = {str(k): float(v) for k, v in target["probabilities"].items()}
            # JSON object order is the declared candidate order. Preserve it so
            # exact probability ties resolve the same way as Jev Bush's argmax.
            if set(pp) != set(gp):
                raise ValueError(f"candidate mismatch for {rid}/{qid}")
            if any(not math.isfinite(value) or value < 0.0 for value in gp.values()):
                raise ValueError(f"invalid gold probabilities for {rid}/{qid}")
            gold_total = sum(gp.values())
            # Gold distributions are serialized to six decimal places.
            if not math.isfinite(gold_total) or abs(gold_total - 1.0) > 1e-5:
                raise ValueError(f"gold probabilities do not sum to one for {rid}/{qid}: {gold_total}")
            gp = {key: value / gold_total for key, value in gp.items()}
            # Prediction object order is the declared candidate order emitted
            # by Jev Bush; use it for deterministic argmax tie-breaking.
            labels = list(pp)
            if any(not math.isfinite(value) or value < 0.0 for value in pp.values()):
                raise ValueError(f"invalid probabilities for {rid}/{qid}")
            total = sum(pp.values())
            if not math.isfinite(total) or abs(total - 1.0) > 1e-6:
                raise ValueError(f"probabilities do not sum to one for {rid}/{qid}: {total}")
            predicted = max(labels, key=lambda k: pp[k])
            if kind == "noul" and abs(float(answer["noul"]) - pp["true"]) > 1e-9:
                raise ValueError(f"noul/probability mismatch for {rid}/{qid}")
            if kind == "choice" and str(answer["choice"]) != predicted:
                raise ValueError(f"choice/probability mismatch for {rid}/{qid}")
            if kind == "score":
                expected_score = sum(float(label) * pp[label] for label in labels)
                if abs(float(answer["score"]) - expected_score) > 1e-9:
                    raise ValueError(f"score/probability mismatch for {rid}/{qid}")
            actual = str(target.get("label", max(labels, key=lambda k: gp.get(k, 0.0))))
            hit = predicted == actual
            confidence = pp[predicted]
            bucket = min(ns.bins - 1, int(confidence * ns.bins))
            calibration[bucket].append((confidence, 1.0 if hit else 0.0))
            n += 1
            correct += hit
            soft_ce = -sum(gp.get(k, 0.0) * math.log(max(pp[k], 1e-300)) for k in labels)
            hard_ce = -math.log(max(pp.get(actual, 0.0), 1e-300))
            row_brier = sum((pp[k] - gp.get(k, 0.0)) ** 2 for k in labels)
            row_tv = 0.5 * sum(abs(pp[k] - gp.get(k, 0.0)) for k in labels)
            add_stats(kinds, kind, hit, soft_ce, hard_ce, row_brier, row_tv)
            add_stats(workflows, workflow, hit, soft_ce, hard_ce, row_brier, row_tv)
            log_loss += soft_ce
            label_log_loss += hard_ce
            brier += row_brier
            tv += row_tv
            if kind == "score":
                answer_score = float(answer["score"])
                target_score = float(target["score"])
                if not math.isfinite(answer_score) or not math.isfinite(target_score):
                    raise ValueError(f"invalid score for {rid}/{qid}")
                score_abs += abs(answer_score - target_score)
                score_n += 1
    if n == 0:
        raise ValueError("no labeled decisions")
    ece = sum(len(bucket) / n * abs(sum(x for x, _ in bucket) / len(bucket) - sum(y for _, y in bucket) / len(bucket)) for bucket in calibration if bucket)
    by_type = finish_groups(kinds)
    result = {
        "rows": len(gold_rows), "decisions": n, "by_type": by_type,
        "by_workflow": finish_groups(workflows),
        "argmax_agreement": correct / n, "soft_cross_entropy": log_loss / n,
        "label_log_loss": label_log_loss / n,
        "brier": brier / n, "total_variation": tv / n, "ece": ece,
        "score_mae": score_abs / score_n if score_n else None,
    }
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
