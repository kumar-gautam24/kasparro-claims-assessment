"""Export the Langfuse observations and scores of every trace in a run report (sanitized: no keys).
Usage: export_langfuse.py runs/<report>.json reports/langfuse/<name>.json"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
HOST = os.getenv("LANGFUSE_HOST", "http://localhost:3000")
AUTH = (os.environ["LANGFUSE_PUBLIC_KEY"], os.environ["LANGFUSE_SECRET_KEY"])


def get(client: httpx.Client, path: str, **params) -> list[dict]:
    r = client.get(f"{HOST}{path}", params=params, auth=AUTH, timeout=30)
    r.raise_for_status()
    return r.json().get("data", [])


def main(report_path: str, out_path: str) -> None:
    report = json.loads(Path(report_path).read_text())
    keep_obs = ("id", "traceId", "type", "name", "startTime", "endTime", "level", "statusMessage", "model",
                "usageDetails", "input", "output", "metadata", "parentObservationId", "latency")
    traces = []
    with httpx.Client() as client:
        for a in report["answers"]:
            tid = a["trace_id"]
            if not tid:
                continue
            obs = get(client, "/api/public/v2/observations", traceId=tid, limit=100,
                      fields="core,basic,io,usage,metadata,model")
            scores = get(client, "/api/public/v3/scores", traceId=tid, limit=100)
            traces.append({
                "answer_id": a["answer_id"], "trace_id": tid, "trace_url": a.get("trace_url"),
                "observations": [{k: o.get(k) for k in keep_obs if k in o} for o in obs],
                "scores": [{k: s.get(k) for k in ("name", "value", "comment", "metadata", "dataType")} for s in scores],
            })
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps({"run": report["run"]["run_id"], "agent_version": report["run"]["agent_version"],
                                          "split": report["run"]["split"], "traces": traces}, indent=1, ensure_ascii=False))
    print(f"exported {len(traces)} traces -> {out_path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
