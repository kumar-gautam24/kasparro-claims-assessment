"""CLI: uv run python scripts/run_eval.py --version v1 --split development"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import db  # noqa: E402
from app.runner import run_split  # noqa: E402


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", choices=["v1", "v2"], required=True)
    ap.add_argument("--split", choices=["development", "check"], required=True)
    ap.add_argument("--answers", nargs="*", help="optional subset of answer_ids (smoke tests only)")
    ap.add_argument("--no-db", action="store_true")
    args = ap.parse_args()
    conn = None
    if not args.no_db:
        conn = await db.connect()
        await db.init_schema(conn)
    try:
        report = await run_split(args.version, args.split, conn=conn, answer_ids=args.answers)
    finally:
        if conn is not None:
            await conn.close()
    print(json.dumps({"run_id": report["run"]["run_id"], "report": report.get("report_path"),
                      "counts": report["counts"], "metrics": report["metrics"]}, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
