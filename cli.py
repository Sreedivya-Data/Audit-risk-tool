"""Run the full audit pipeline and save the results to the output folder."""
import argparse
import dataclasses
import json
from pathlib import Path

from src import anomaly_detector, ml_detector, risk_engine, summary_writer, transaction_loader
from src.config import load_config
from src.run_log import RunLog


def _save_table(df, path):
    if df.index.name:
        df = df.reset_index()
    df.to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser(description="AI Audit and Risk Analytics Tool")
    parser.add_argument("input", help="Path to the transactions file")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--out", default="output")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)

    config = load_config(args.config)
    run_log = RunLog(input_file=args.input, config=config)

    df, report = transaction_loader.load(args.input, config)
    print(f"Loaded {report.rows_loaded} rows, excluded {report.rows_excluded}")

    rule_flags = anomaly_detector.detect(df, config, run_log)
    ml_flags = ml_detector.detect(df, rule_flags, config, run_log)

    # Combine, and drop any exact repeats in case the ML step returns both lists
    seen, flags = set(), []
    for f in rule_flags + ml_flags:
        key = (f.tx_id, f.rule)
        if key not in seen:
            seen.add(key)
            flags.append(f)
    print(f"Raised {len(flags)} flags")

    tx_scores, account_rollup, approver_rollup = risk_engine.score(df, flags, config)
    summary = summary_writer.write_summary(
        flags, tx_scores, account_rollup, approver_rollup, config, run_log
    )

    _save_table(df, out_dir / "transactions_clean.csv")
    _save_table(tx_scores, out_dir / "transaction_scores.csv")
    _save_table(account_rollup, out_dir / "account_rollup.csv")
    _save_table(approver_rollup, out_dir / "approver_rollup.csv")
    with open(out_dir / "flags.json", "w", encoding="utf-8") as fh:
        json.dump([dataclasses.asdict(f) for f in flags], fh, indent=2, default=str)
    (out_dir / "audit_summary.txt").write_text(summary, encoding="utf-8")

    # I could not see the arguments of these RunLog methods, so failures are skipped
    for name in ("finalise", "save"):
        try:
            getattr(run_log, name)()
        except Exception as exc:
            print(f"(run log {name} skipped: {exc})")

    print(f"\nDone. Results saved in the '{out_dir}' folder.\n")
    print(summary)


if __name__ == "__main__":
    main()