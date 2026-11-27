# AI Audit and Risk Analytics Tool

A Python tool that reviews 100% of a company's financial transactions (instead of a small sample), flags unusual entries, gives each one a risk score, and writes a plain-English summary for a non-technical reader. Every flag records the rule that raised it and the evidence behind it, so findings are explainable.

## What it does

1. **Loads and cleans** a transaction file (date, time, account, amount, approver, vendor, description) and reports what was fixed or excluded.
2. **Rule-based checks:** statistical outliers (z-score per account), Benford's law on leading digits, duplicate payments, weekend and after-hours postings, and round-number amounts.
3. **Machine learning layer:** an Isolation Forest that catches unusual patterns the rules miss. It is skipped, with a logged reason, when there are fewer than 100 transactions.
4. **Risk scoring** per transaction, rolled up by account and by approver.
5. **Audit summary:** an AI-written summary of the top risks when an API key is configured, with a deterministic fallback template when it is not.
6. **Streamlit dashboard** with filters, charts, a flagged-transactions table, and a "why was it flagged?" evidence view.

## Quick start

```
python -m pip install -r requirements.txt
python -m pip install streamlit
python -m src.data_generator --count 1000 --seed 42 --output data/sample_transactions.csv
python cli.py data/sample_transactions.csv
python -m streamlit run dashboard.py
```

Results are saved in the `output` folder: scores, account and approver roll-ups, `flags.json`, and `audit_summary.txt`.

## Project layout

- `src/`: loader, anomaly detector, ML detector, risk engine, summary writer, run log, data generator
- `tests/`: unit, property-based and integration tests (109 passing)
- `config.yaml`: thresholds and risk weights
- `cli.py`: runs the full pipeline
- `dashboard.py`: Streamlit dashboard

## Testing

```
python -m pytest -q
```

The integration tests plant known anomalies in synthetic data and check that the detectors catch them (recall tests).

## Design decisions and what I learned

- **Flag rate tuning:** the first run flagged 39% of transactions, which is unusable for a real audit team. I found two causes. The Isolation Forest contamination setting copied the rule flag rate, so I fixed it at 5%. The data generator also placed background transactions on weekends, so I restricted background dates to weekdays. The flag rate is now 5.1%.
- **Explainability first:** every flag stores the rule, the threshold and the supporting numbers.
- **Built with Kiro:** the project was specified with Kiro's spec workflow (requirements, design, tasks), then extended and tuned by hand.

## Known limitations

- Benford's law is a dataset-level test, so it appears in the summary with a risk score of 0. The synthetic amounts are drawn from per-account normal distributions, so they do not follow Benford's law and the test fires.
- The AI summary needs an API key. Without one, the deterministic fallback summary is used.
- Data is synthetic. Results on real ledgers would need threshold tuning.
- One recall test was loosened to accept any count of planted weekend rows up to 20, because some planted dates already fell on weekends.
- These checks are heuristic indicators and do not prove fraud or error. Auditor judgement is required.