"""Compare two development reports scored with identical rules: metrics, per-label, per-answer and
per-claim regressions. Usage: compare_runs.py runs/v1_development_*.json runs/v2_development_*.json"""
from __future__ import annotations

import json
import sys
from pathlib import Path

LABELS = ("supported", "contradicted", "unknown")


def load(p: str) -> dict:
    return json.loads(Path(p).read_text())


def claim_state(report: dict) -> dict[str, str]:
    """gold claim_id -> 'correct' | 'wrong_label' | 'missed'"""
    state = {cid: "missed" for cid in report["missed_gold_ids"]}
    for m in report["matched"]:
        state[m["claim_id"]] = "correct" if m["label_correct"] else "wrong_label"
    return state


def main(a_path: str, b_path: str) -> str:
    a, b = load(a_path), load(b_path)
    for k in ("dataset_version", "evaluator_version", "fact_version", "provider", "model", "file_hashes", "settings"):
        if a["run"][k] != b["run"][k]:
            print(f"WARNING: runs differ in {k}", file=sys.stderr)
    av, bv = a["run"]["agent_version"], b["run"]["agent_version"]
    out = [f"# {av} vs {bv} on development ({a['counts']['answers']} answers, G={a['counts']['G']})", ""]
    out += ["| metric | " + av + " | " + bv + " | delta |", "|---|---|---|---|"]

    def row(name, x, y):
        out.append(f"| {name} | {x:.3f} | {y:.3f} | {y - x:+.3f} |")

    ma, mb = a["metrics"], b["metrics"]
    row("joint F1 (main)", ma["joint"]["f1"], mb["joint"]["f1"])
    row("joint precision", ma["joint"]["precision"], mb["joint"]["precision"])
    row("joint recall = coverage C/G", ma["joint"]["recall"], mb["joint"]["recall"])
    row("extraction F1", ma["extraction"]["f1"], mb["extraction"]["f1"])
    row("extraction precision", ma["extraction"]["precision"], mb["extraction"]["precision"])
    row("extraction recall", ma["extraction"]["recall"], mb["extraction"]["recall"])
    row("macro F1", ma["macro_f1"], mb["macro_f1"])
    for lb in LABELS:
        row(f"{lb} F1", ma["per_label"][lb]["f1"], mb["per_label"][lb]["f1"])
    ca, cb = a["counts"], b["counts"]
    out += ["", "| count | " + av + " | " + bv + " |", "|---|---|---|"]
    for k in ("P", "M", "C", "unmatched_predictions", "answer_errors"):
        out.append(f"| {k} | {ca[k]} | {cb[k]} |")

    pa = {r["answer_id"]: r for r in a["answers"]}
    pb = {r["answer_id"]: r for r in b["answers"]}
    out += ["", "## Per-answer joint F1 changes", "", "| answer | " + av + " | " + bv + " | delta | status " + av + "/" + bv + " |", "|---|---|---|---|---|"]
    worse = better = 0
    for aid in sorted(pa):
        x, y = pa[aid]["scores"]["joint_f1"] or 0.0, pb[aid]["scores"]["joint_f1"] or 0.0
        if abs(y - x) > 1e-9:
            worse += y < x
            better += y > x
            out.append(f"| {aid} | {x:.2f} | {y:.2f} | {y - x:+.2f} | {pa[aid]['status']}/{pb[aid]['status']} |")
    out += ["", f"Answers improved: {better}; regressed: {worse}; unchanged: {len(pa) - better - worse}.", ""]

    sa, sb = claim_state(a), claim_state(b)
    regress = sorted(c for c in sa if sa[c] == "correct" and sb.get(c) != "correct")
    fixed = sorted(c for c in sa if sa[c] != "correct" and sb.get(c) == "correct")
    out += [f"## Claim-level: fixed {len(fixed)}, regressed {len(regress)}", ""]
    out += ["Regressed claims (correct in " + av + ", not in " + bv + "):", ""]
    out += [f"- {c}: {sb.get(c)}" for c in regress] or ["- none"]
    trans: dict[tuple, int] = {}
    for c in sa:
        trans[(sa[c], sb.get(c))] = trans.get((sa[c], sb.get(c)), 0) + 1
    out += ["", "Transitions (" + av + " -> " + bv + "):", ""]
    out += [f"- {x} -> {y}: {n}" for (x, y), n in sorted(trans.items())]
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    text = main(sys.argv[1], sys.argv[2])
    print(text)
    if len(sys.argv) > 3:
        Path(sys.argv[3]).write_text(text)
