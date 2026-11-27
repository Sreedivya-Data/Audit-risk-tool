# Implementation Plan: AI Audit and Risk Analytics Tool

## Overview

Implement the full pipeline in Python: project scaffolding → data loading/cleaning → rule-based anomaly detection → ML detection → risk scoring → AI-written audit summary → Streamlit dashboard → synthetic data generator → test suite → documentation. Each task builds on the previous ones and ends with everything wired into `main.py`.

## Tasks

- [x] 1. Set up project structure, configuration, and core data models
  - Create the directory layout: `src/`, `tests/unit/`, `tests/property/`, `tests/integration/`, `output/`, `data/`
  - Write `config.py` with the `Config` dataclass and all default thresholds (zscore_threshold, benford_min_count, duplicate_window_days, business_start/end, round_threshold, min_ml_tx, rule_weights, OpenAI settings)
  - Write `config.yaml` with matching defaults so values can be overridden without code changes
  - Define the `Flag` dataclass and `ValidationReport` dataclass in `models.py`
  - Define `MissingColumnsError` and `EmptyDatasetError` custom exceptions in `exceptions.py`
  - Write `run_log.py` with the `RunLog` class that accumulates entries and serialises to `output/run_<timestamp>.json`
  - Create `requirements.txt` pinning all dependencies (pandas, numpy, scipy, scikit-learn, openai, streamlit, plotly, hypothesis, pytest, pytest-mock, openpyxl, pyarrow, pyyaml)
  - _Requirements: 1.1, 8.2, 14.1, 14.2_

- [x] 2. Implement Transaction_Loader
  - [x] 2.1 Implement `transaction_loader.py` with the `load(path, config)` function
    - Use `pd.read_csv` / `pd.read_excel` with `dtype=str` for the initial read
    - Validate required columns `{date, time, account, amount, approver, vendor, description}`; raise `MissingColumnsError` on failure
    - Assign stable `tx_id` as `sha256(row_as_string)[:12]` before cleaning
    - Coerce `date` with `pd.to_datetime(..., errors='coerce')`, `amount` with `pd.to_numeric(..., errors='coerce')`, `time` with custom `HH:MM` / `HH:MM:SS` parsing
    - Drop rows where `account`, `amount`, or `date` is null; record them in `exclusion_detail`
    - Identify exact duplicates with `df.duplicated(keep='first')` and remove; record count in `ValidationReport`
    - Return `(df, ValidationReport)` and log row counts to `RunLog`
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6_

  - [ ]* 2.2 Write property test for Transaction_Loader (Property 1 & 2)
    - **Property 1: Loading preserves unique rows** — for any valid input, row count in returned DataFrame equals distinct rows in input, and duplicates_removed matches actual removed count
    - **Property 2: Validation report exactness** — every coerced/excluded row has exactly one detail entry; reported counts equal actual counts
    - **Validates: Requirements 1.1, 1.2, 1.3, 1.5, 1.6**

  - [ ]* 2.3 Write unit tests for Transaction_Loader
    - Test valid CSV, missing columns, non-numeric amount coercion, duplicate removal, empty file, Excel format
    - _Requirements: 1.1–1.6_

- [x] 3. Implement Anomaly_Detector — statistical outlier and Benford's law rules
  - [x] 3.1 Implement statistical outlier detection in `anomaly_detector.py`
    - Group by `account`; skip accounts with fewer than `config.min_account_tx` transactions and log the skip
    - Compute z-scores using `(amount - account_mean) / account_std`; handle zero std by skipping and logging
    - Emit `Flag(rule="statistical_outlier", ...)` with evidence fields `z_score`, `account_mean`, `account_std`, `account_tx_count`
    - _Requirements: 2.1, 2.2, 2.3_

  - [ ]* 3.2 Write property test for statistical outlier detection (Property 3)
    - **Property 3: Statistical outlier flag and evidence consistency** — for any account with ≥min_account_tx transactions, flagged set equals transactions with |z| > threshold; evidence z_score equals computed value within float tolerance
    - **Validates: Requirements 2.1, 2.2**

  - [x] 3.3 Implement Benford's law test in `anomaly_detector.py`
    - Extract first digit of each positive amount; compute observed vs. expected distribution using `log10(1+1/d)`
    - Run `scipy.stats.chisquare`; skip if fewer than `config.benford_min_count` positive amounts and log the skip
    - If p-value < 0.05 emit a single dataset-level flag with `tx_id="__dataset__"` and evidence `chi2_statistic`, `p_value`, `per_digit_observed`, `per_digit_expected`, `per_digit_deviation`
    - _Requirements: 3.1, 3.2, 3.3_

  - [ ]* 3.4 Write property test for Benford detection (Property 4)
    - **Property 4: Benford deviation evidence completeness** — whenever p-value < benford_significance the flag carries all five evidence fields
    - **Validates: Requirements 3.1, 3.2**

- [x] 4. Implement Anomaly_Detector — duplicate payment, timing, and round-number rules
  - [x] 4.1 Implement duplicate payment detection in `anomaly_detector.py`
    - Sort by `vendor`, `amount`, `date`; use `pd.merge_asof` within each `(vendor, amount)` group with `tolerance=pd.Timedelta(days=config.duplicate_window_days)`
    - Emit two flags per matched pair with evidence `paired_tx_id`, `date_gap_days`, `group_size`; note group size when > 50
    - _Requirements: 4.1, 4.2, 4.3_

  - [ ]* 4.2 Write property test for duplicate payment detection (Property 5)
    - **Property 5: Duplicate payment flag symmetry and evidence** — both members of any qualifying pair are flagged with the partner's tx_id; pairs > 7 days apart are not flagged
    - **Validates: Requirements 4.1, 4.2**

  - [x] 4.3 Implement weekend/after-hours detection in `anomaly_detector.py`
    - Flag `weekend_posting` when `date.weekday() >= 5` with evidence `day_of_week`, `posting_date`
    - Flag `after_hours_posting` when time is outside `[config.business_start, config.business_end]` with evidence `posting_time`, `business_hours_window`
    - _Requirements: 5.1, 5.2, 5.3, 5.4_

  - [ ]* 4.4 Write property test for timing flags (Property 6)
    - **Property 6: Timing flag exactness** — every weekend transaction gets weekend_posting and no weekday transaction gets it; every out-of-window transaction gets after_hours_posting and in-window ones do not; each flag carries the required evidence
    - **Validates: Requirements 5.1, 5.2, 5.3, 5.4**

  - [x] 4.5 Implement round-number detection in `anomaly_detector.py`
    - Flag `round_amount` when `amount % config.round_threshold == 0` with evidence `amount`, `threshold`
    - Expose the public `detect(df, config, run_log) -> list[Flag]` function that runs all five rules in sequence
    - _Requirements: 6.1, 6.2, 6.3_

  - [ ]* 4.6 Write property test for round amount detection (Property 7)
    - **Property 7: Round amount flag completeness** — every exact multiple receives the flag with correct evidence; no non-multiple receives it
    - **Validates: Requirements 6.1, 6.2, 6.3**

  - [ ]* 4.7 Write unit tests for all Anomaly_Detector rules
    - Cover z-score boundary at exactly 3.0, Benford chi-square on a known distribution, duplicate pair with 7-day boundary, weekend/after-hours edge conditions, round-number with custom threshold
    - _Requirements: 2.1–2.3, 3.1–3.3, 4.1–4.3, 5.1–5.4, 6.1–6.3_

- [x] 5. Checkpoint — Ensure all tests pass for loader and detector modules
  - Ensure all tests pass, ask the user if questions arise.

- [x] 6. Implement ML_Detector
  - [x] 6.1 Implement `ml_detector.py` with the `detect(df, rule_flags, config, run_log) -> list[Flag]` function
    - Build the feature matrix: `amount_log`, `hour_of_day`, `day_of_week`, `is_weekend`, `account_enc`, `vendor_enc`, `approver_enc`, `amount_z_norm`
    - Derive contamination as `max(rule_flag_rate, 0.01)` where `rule_flag_rate = len(set(f.tx_id for f in rule_flags)) / len(df)`
    - Train `IsolationForest(n_estimators=200, contamination=..., random_state=config.random_state, n_jobs=-1)`
    - Skip and log when fewer than `config.min_ml_tx` transactions
    - Emit `Flag(rule="ml_isolation_forest", ...)` with evidence `anomaly_score` and `feature_contributions` (top 3 features by delta from median)
    - _Requirements: 7.1, 7.2, 7.3, 7.4_

  - [ ]* 6.2 Write property test for ML detection (Property 8)
    - **Property 8: ML flag threshold consistency** — every transaction with decision_function score below threshold is flagged; transactions at or above threshold are not flagged; each flag contains anomaly_score and feature_contributions
    - **Validates: Requirements 7.2, 7.3**

  - [ ]* 6.3 Write unit tests for ML_Detector
    - Test feature matrix construction, contamination rate derivation, skip condition, flag output structure, evidence field presence
    - _Requirements: 7.1–7.4_

- [x] 7. Implement Risk_Engine
  - [x] 7.1 Implement `risk_engine.py` with the `score(df, flags, config) -> tuple[DataFrame, DataFrame, DataFrame]` function
    - For each transaction sum base points from `config.rule_weights` for all its flags; cap at 100
    - Populate `score_breakdown` as `{rule: points}` per flag; set `risk_score = 0` for unflagged transactions
    - Produce `account_rollup` and `approver_rollup` DataFrames with columns `tx_count`, `flag_count_by_rule`, `mean_risk_score`, `max_risk_score`, `is_small_sample`
    - Set `is_small_sample = True` for groups with fewer than 5 transactions
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6_

  - [ ]* 7.2 Write property test for risk scoring (Property 12 & 13)
    - **Property 12: Risk score bounded with exact breakdown** — risk_score always in [0,100] and equals min(sum of contributions, 100); score_breakdown has one entry per flag; unflagged transactions have score 0
    - **Property 13: Rollup correctness including small-sample groups** — mean and max match arithmetic mean/max of group scores; is_small_sample True iff tx_count < 5
    - **Validates: Requirements 9.1–9.6**

  - [ ]* 7.3 Write unit tests for Risk_Engine
    - Test score capping at 100, zero score for unflagged row, breakdown dict values, rollup mean/max computation, small-sample annotation
    - _Requirements: 9.1–9.6_

- [x] 8. Implement Summary_Writer and run-log / output-artifact writing
  - [x] 8.1 Implement `summary_writer.py` with the `write_summary(flags, tx_scores, account_rollup, approver_rollup, config, run_log) -> str` function
    - Build a structured prompt from the top 5 risks by aggregated `risk_score` with their evidence; call `openai.chat.completions.create` with `model=config.openai_model`, `max_tokens=config.openai_max_tokens`, `timeout=config.openai_timeout`
    - On any exception or when `OPENAI_API_KEY` is not set, produce the deterministic fallback template
    - Write the prompt text and raw API response (or `"FALLBACK"`) verbatim to `run_log`
    - Assert the returned summary is ≤ 400 words; always append the limitations statement
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 14.3_

  - [x] 8.2 Implement output artifact writing in `run_log.py`
    - After risk scoring completes, serialise all flags to `output/flags_<run_id>.parquet` (columns: `run_id`, `tx_id`, `rule`, `threshold`, `evidence`, `risk_score`, `score_breakdown`) and a parallel `output/flags_<run_id>.json`
    - Finalise and write `output/run_<timestamp>.json` with all accumulated fields including `ai_prompt`, `ai_response`, `summary_text`
    - _Requirements: 8.2, 8.3, 8.4_

  - [ ]* 8.3 Write property test for summary and run log (Property 9, 10, 11, 14)
    - **Property 9: Flag completeness** — every flag has non-empty rule, human-readable threshold, and non-empty evidence
    - **Property 10: Run log completeness** — run log contains input file, timestamp, row counts, per-rule flag counts, every skipped check
    - **Property 11: Evidence retrievability round-trip** — loading the artifact and looking up by tx_id+rule returns the same evidence without re-running
    - **Property 14: Summary bounded, limited, and traceable** — summary ≤ 400 words, contains limitations statement, every number appears in run output
    - **Validates: Requirements 8.1, 8.2, 8.3, 8.4, 10.1, 10.5, 10.6**

  - [ ]* 8.4 Write unit tests for Summary_Writer
    - Test API call with mocked openai client, fallback on exception, limitations statement presence, number traceability (each number in summary is in run output)
    - _Requirements: 10.1–10.6_

- [x] 9. Checkpoint — Ensure all pipeline module tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 10. Implement Data Generator
  - [x] 10.1 Implement `data_generator.py` with `generate(count, seed, anomalies) -> pd.DataFrame` and the `AnomalySpec` dataclass
    - Generate base transactions: 20 accounts with normally distributed amounts, 50 vendors, 10 approvers, dates uniform over 12-month window, times normally distributed around 10:00–16:00 on weekdays
    - Accept `seed` for `numpy.random.default_rng` so identical inputs produce identical outputs
    - Implement anomaly planting for each type: `statistical_outlier`, `benford_violation`, `duplicate_payment`, `weekend_posting`, `after_hours_posting`, `round_amount`, `ml_pattern`
    - Expose a CLI: `python data_generator.py --count N --seed S --output path.csv`
    - _Requirements: 12.1, 12.2, 12.3_

  - [ ]* 10.2 Write property test for data generator (Property 15)
    - **Property 15: Generator reproducibility and schema** — identical count+seed+anomalies produces identical DataFrames; output always has the seven required columns
    - **Validates: Requirements 12.1, 12.2**

  - [ ]* 10.3 Write unit tests for Data Generator
    - Test seed reproducibility, all seven columns present, planted anomaly presence for each anomaly type, clean dataset produces no anomalies in planted slots
    - _Requirements: 12.1, 12.2, 12.3_

- [ ] 11. Write integration / detection recall tests
  - [-] 11.1 Implement `tests/integration/test_statistical_outlier_recall.py`
    - Plant 20 extreme-amount anomalies in a 500-row dataset; run full pipeline; assert ≥90% detection rate
    - _Requirements: 2.4, 12.4_

  - [-] 11.2 Implement `tests/integration/test_benford_recall.py`
    - Plant 200 fabricated same-digit amounts; run pipeline with 10 different seeds; assert `benford_deviation` triggered in ≥90% of runs
    - _Requirements: 3.4, 12.4_

  - [-] 11.3 Implement `tests/integration/test_duplicate_recall.py`
    - Plant 10 duplicate pairs; assert both members of every planted pair are flagged
    - _Requirements: 4.4, 12.4_

  - [-] 11.4 Implement `tests/integration/test_timing_recall.py`
    - Plant weekend and after-hours entries; assert 100% detection
    - _Requirements: 5.1, 5.2, 12.4_

  - [ ] 11.5 Implement `tests/integration/test_round_amount_recall.py`
    - Plant exact multiples of 1,000; assert 100% detection
    - _Requirements: 6.4, 12.4_

  - [~] 11.6 Implement `tests/integration/test_ml_recall.py`
    - Plant 20 patterned but rule-undetectable anomalies; run full pipeline; assert ≥75% detection
    - _Requirements: 7.5, 12.4_

  - [~] 11.7 Implement `tests/integration/test_false_positive_ceiling.py`
    - Generate a clean 1,000-row dataset; assert `statistical_outlier`, `duplicate_payment`, and `round_amount` flag rate ≤5% of rows
    - _Requirements: 12.5_

  - [~] 11.8 Implement `tests/integration/test_evidence_retrievability.py`
    - Run full pipeline; reload the output artifact; assert every flag is retrievable by `tx_id + rule` with matching evidence
    - _Requirements: 8.3, 8.4_

- [~] 12. Implement `main.py` CLI and wire all pipeline modules together
  - Accept `--input` (required), `--config` (optional, default `config.yaml`), `--output-dir` (optional, default `output/`) via `argparse`
  - Load config from YAML; instantiate `RunLog`; call `Transaction_Loader.load → Anomaly_Detector.detect → ML_Detector.detect → Risk_Engine.score → Summary_Writer.write_summary`
  - Write the output artifact (parquet + JSON) and finalise the run log
  - Catch `MissingColumnsError`, `EmptyDatasetError`, `FileNotFoundError` and print descriptive messages to `stderr`; log all exceptions to the run log
  - Exit with code 0 on success, non-zero on unrecoverable error
  - _Requirements: 1.4, 8.2, 14.2_

- [ ] 13. Implement Streamlit Dashboard
  - [~] 13.1 Implement `dashboard.py` with sidebar filters and KPI row
    - Add `@st.cache_data` loader for the output artifact (parquet or JSON), keyed by file path and mtime
    - Accept `--artifact` CLI argument or file-upload widget
    - Sidebar: date-range slider, account multiselect, approver multiselect, vendor multiselect, rule-name multiselect, min risk score slider
    - KPI row: total flagged transactions, unique flagged accounts, top approver by max risk score
    - _Requirements: 11.1, 11.2_

  - [~] 13.2 Implement dashboard charts and transaction table
    - Row 1: flag counts by rule (bar chart via `plotly.express`), risk distribution by account (bar chart)
    - Row 2: transaction amount over time with flagged points highlighted (line + scatter via `plotly.express`)
    - Flagged transactions table using `st.dataframe` with `on_select`; columns: date, account, vendor, approver, amount, risk score, flag reasons
    - When a row is selected display an Evidence expander per flag showing full evidence dict
    - Display audit summary text in an `st.expander`
    - Display empty-state message when no rows match active filters
    - _Requirements: 11.1, 11.3, 11.4, 11.5, 11.6_

- [~] 14. Write README and one-page summary document
  - Write `README.md` covering setup (`pip install -r requirements.txt`), usage (`python main.py`, `streamlit run dashboard.py`, `python data_generator.py`), data format (required columns and types), design choices for each module, and known limitations
  - Document how AI summary output is verified (run-log prompt/response capture), referencing Requirement 10.3
  - Document each detection rule's threshold and the rationale for it
  - Write `docs/summary.md` as a one-page document stating the business problem, the approach, and the decisions the Tool supports
  - _Requirements: 13.1, 13.2, 13.3, 13.4_

- [~] 15. Final checkpoint — Ensure all tests pass and pipeline runs end-to-end
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP
- Each task references specific requirements for traceability
- Checkpoints (tasks 5, 9, 15) ensure incremental validation at meaningful pipeline boundaries
- Property tests use Hypothesis with `@settings(max_examples=100)` applied globally via `conftest.py`
- Unit tests use pytest with pytest-mock for mocking the OpenAI client
- Integration tests use `tmp_path` fixtures and the synthetic data generator for reproducibility
- The design document's Correctness Properties section is the authoritative source for all property test specifications

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1"] },
    { "id": 1, "tasks": ["2.1"] },
    { "id": 2, "tasks": ["2.2", "2.3"] },
    { "id": 3, "tasks": ["3.1"] },
    { "id": 4, "tasks": ["3.2", "3.3"] },
    { "id": 5, "tasks": ["3.4", "4.1"] },
    { "id": 6, "tasks": ["4.2", "4.3"] },
    { "id": 7, "tasks": ["4.4", "4.5"] },
    { "id": 8, "tasks": ["4.6"] },
    { "id": 9, "tasks": ["4.7"] },
    { "id": 10, "tasks": ["5"] },
    { "id": 11, "tasks": ["6.1"] },
    { "id": 12, "tasks": ["6.2", "6.3"] },
    { "id": 13, "tasks": ["7.1"] },
    { "id": 14, "tasks": ["7.2", "7.3"] },
    { "id": 15, "tasks": ["8.1", "8.2"] },
    { "id": 16, "tasks": ["8.3", "8.4"] },
    { "id": 17, "tasks": ["9"] },
    { "id": 18, "tasks": ["10.1"] },
    { "id": 19, "tasks": ["10.2", "10.3"] },
    { "id": 20, "tasks": ["11.1", "11.2", "11.3", "11.4", "11.5", "11.6", "11.7", "11.8"] },
    { "id": 21, "tasks": ["12"] },
    { "id": 22, "tasks": ["13.1"] },
    { "id": 23, "tasks": ["13.2"] },
    { "id": 24, "tasks": ["14"] },
    { "id": 25, "tasks": ["15"] }
  ]
}
```
