# Requirements Document

## Introduction

The AI Audit and Risk Analytics Tool is a Python application that ingests a file of financial transactions, flags unusual entries using explainable rule-based checks and a machine-learning model, scores risk at the transaction, account, and approver level, and produces a plain-English audit summary for non-technical clients. The tool supports the kind of analytical work performed at audit firms (e.g., Deloitte), with every flagged item accompanied by the reason it was flagged. The tool includes a Streamlit dashboard for interactive review of flagged transactions.

## Glossary

- **Tool**: The AI Audit and Risk Analytics Tool application as a whole.
- **Transaction**: A single journal entry in the input data, with fields for date, time, account, amount, approver, vendor, and description.
- **Transaction_Loader**: The module that reads, cleans, and validates the input transaction file.
- **Anomaly_Detector**: The module that applies rule-based anomaly checks to transactions.
- **ML_Detector**: The module that applies the Isolation Forest model to detect unusual patterns.
- **Risk_Engine**: The module that assigns risk scores to transactions and aggregates scores by account and approver.
- **Summary_Writer**: The module that generates the plain-English audit summary from flagged transactions and evidence.
- **Dashboard**: The Streamlit application providing filters, charts, and a table of flagged transactions.
- **Flag**: A record marking a transaction as unusual, containing the rule or model that triggered it and the supporting evidence.
- **Evidence**: The concrete data points (e.g., z-score value, matched duplicate partner) supporting a Flag.
- **Planted_Anomaly**: An intentionally inserted anomaly in the synthetic dataset, used to test detection accuracy.
- **Z_Score**: The standardized deviation of a transaction amount from the account-level mean amount.
- **Benford's_Law**: The expected distribution of leading digits in naturally occurring financial data.
- **Round_Number**: An amount that is an exact multiple of a configured threshold (e.g., 1,000).
- **After_Hours**: A posting time outside the configured business-hours window (default 08:00-18:00 local time).
- **Risk_Score**: A numeric score in the range 0-100 assigned to each transaction, with higher scores indicating greater risk.

## Requirements

### Requirement 1: Load and Clean Transactions

**User Story:** As an auditor, I want the Tool to load and clean the transaction file, so that downstream analysis runs on valid, consistently typed data.

#### Acceptance Criteria

1. WHEN a valid transaction file (CSV or Excel) is provided, THE Transaction_Loader SHALL parse it into a pandas DataFrame with columns date, time, account, amount, approver, vendor, and description.
2. WHEN a value cannot be converted to its expected type, THE Transaction_Loader SHALL coerce it where unambiguous and record the raw value in a validation report.
3. WHEN a required field is missing for a row, THE Transaction_Loader SHALL exclude the row from analysis and record the excluded row and reason in the validation report.
4. WHEN the input file does not contain the required columns, THE Transaction_Loader SHALL raise a descriptive error listing the missing columns.
5. THE Transaction_Loader SHALL report the count of rows loaded, rows coerced, and rows excluded after processing.
6. WHEN a duplicate row (identical across all columns) appears in the input, THE Transaction_Loader SHALL keep one instance and record the removal in the validation report.

### Requirement 2: Statistical Outlier Detection

**User Story:** As an auditor, I want amounts that are unusual relative to their account history flagged, so that potentially misstated entries are reviewed.

#### Acceptance Criteria

1. WHEN a transaction's absolute Z_Score within its account exceeds 3.0, THE Anomaly_Detector SHALL flag the transaction with rule "statistical_outlier".
2. THE Anomaly_Detector SHALL store the computed Z_Score and the account mean and standard deviation as Evidence for each flagged transaction.
3. WHEN an account has fewer than 10 transactions, THE Anomaly_Detector SHALL skip Z_Score evaluation for that account and record the skip in the run log.
4. WHEN a Planted_Anomaly with an extreme amount is present in a dataset, THE Anomaly_Detector SHALL flag it in at least 90% of test runs.

### Requirement 3: Benford's Law Test

**User Story:** As an auditor, I want the distribution of leading digits tested against Benford's_Law, so that fabricated or manipulated amounts are surfaced.

#### Acceptance Criteria

1. THE Anomaly_Detector SHALL compute the observed first-digit distribution across all positive amounts and compare it to Benford's_Law using a chi-square test with a significance level of 0.05.
2. WHEN the chi-square p-value is below 0.05, THE Anomaly_Detector SHALL record an account-level flag "benford_deviation" with the chi-square statistic, p-value, and per-digit deviations as Evidence.
3. WHEN the number of positive amounts is below 100, THE Anomaly_Detector SHALL skip the Benford test and record the skip in the run log.
4. WHEN a dataset contains a Planted_Anomaly set of repeated fabricated amounts, THE Anomaly_Detector SHALL trigger "benford_deviation" in at least 90% of test runs.

### Requirement 4: Duplicate Payment Detection

**User Story:** As an auditor, I want duplicate payments identified, so that double-paid invoices are recovered.

#### Acceptance Criteria

1. WHEN two transactions share the same vendor and the same amount and are posted within 7 days of each other, THE Anomaly_Detector SHALL flag both transactions with rule "duplicate_payment".
2. THE Anomaly_Detector SHALL record the paired transaction identifiers and posting-date gap as Evidence for each duplicate flag.
3. WHEN more than 50 transactions share a vendor and amount within 7 days, THE Anomaly_Detector SHALL flag each one and note the group size in Evidence.
4. WHEN a Planted_Anomaly duplicate pair is present in a dataset, THE Anomaly_Detector SHALL flag both members of the pair.

### Requirement 5: Weekend and After-Hours Detection

**User Story:** As an auditor, I want postings made outside normal business hours flagged, so that unusual entry timing is investigated.

#### Acceptance Criteria

1. WHEN a transaction's posting date falls on a Saturday or Sunday, THE Anomaly_Detector SHALL flag it with rule "weekend_posting".
2. WHEN a transaction's posting time is After_Hours, THE Anomaly_Detector SHALL flag it with rule "after_hours_posting".
3. WHERE the business-hours window is configured differently, THE Anomaly_Detector SHALL apply the configured window instead of the default.
4. THE Anomaly_Detector SHALL record the posting day of week and time as Evidence for each timing flag.

### Requirement 6: Round-Number Detection

**User Story:** As an auditor, I want suspiciously round amounts flagged, so that estimated or fabricated entries are reviewed.

#### Acceptance Criteria

1. WHEN a transaction amount is an exact multiple of 1,000, THE Anomaly_Detector SHALL flag it with rule "round_amount".
2. WHERE a round-number threshold other than 1,000 is configured, THE Anomaly_Detector SHALL apply the configured threshold.
3. THE Anomaly_Detector SHALL record the amount and threshold as Evidence for each round-number flag.
4. WHEN a Planted_Anomaly round amount is present in a dataset, THE Anomaly_Detector SHALL flag it.

### Requirement 7: Machine-Learning Detection Layer

**User Story:** As an auditor, I want unusual patterns that rules miss detected, so that novel anomalies are not overlooked.

#### Acceptance Criteria

1. THE ML_Detector SHALL train an Isolation Forest on the numeric and encoded categorical features of the transaction dataset with a contamination rate derived from an initial rules pass.
2. THE ML_Detector SHALL flag transactions with an Isolation Forest anomaly score beyond the model's decision threshold with rule "ml_isolation_forest".
3. THE ML_Detector SHALL record the anomaly score and the features contributing to the score as Evidence for each ML flag.
4. WHEN the dataset contains fewer than 100 transactions, THE ML_Detector SHALL skip model training and record the skip in the run log.
5. WHEN a Planted_Anomaly patterned but rule-undetected anomaly is present in a dataset, THE ML_Detector SHALL flag it in at least 75% of test runs.

### Requirement 8: Explainability and Run Log

**User Story:** As an auditor, I want every flag logged with its reason, so that every finding is explainable to the client and defensible in review.

#### Acceptance Criteria

1. THE Anomaly_Detector and ML_Detector SHALL attach, to every Flag, the rule or model name, the trigger threshold, and the Evidence values that caused the trigger.
2. THE Tool SHALL write a run log recording the input file, timestamp, row counts, per-rule flag counts, and all skipped checks with reasons.
3. WHEN a Flag is emitted, THE Tool SHALL ensure the flag's Evidence is retrievable from the run's output data without re-running the analysis.
4. THE Tool SHALL store all flags for a run in a single output artifact keyed by transaction identifier.

### Requirement 9: Risk Scoring and Rollup

**User Story:** As an audit manager, I want risk scores per transaction and rolled up by account and approver, so that review effort is prioritized.

#### Acceptance Criteria

1. THE Risk_Engine SHALL assign each transaction a Risk_Score in the range 0-100 computed from its flags and the severity weight of each rule.
2. THE Risk_Engine SHALL record the per-flag point contributions in the transaction's score breakdown.
3. THE Risk_Engine SHALL produce an account-level rollup report containing, per account: transaction count, flag counts by rule, and mean and maximum Risk_Score.
4. THE Risk_Engine SHALL produce an approver-level rollup report containing, per approver: transaction count, flag counts by rule, and mean and maximum Risk_Score.
5. WHEN a transaction has no flags, THE Risk_Engine SHALL assign it a Risk_Score of 0.
6. WHEN a rollup group contains fewer than 5 transactions, THE Risk_Engine SHALL still include the group and annotate it as a small-sample group.

### Requirement 10: AI-Written Audit Summary

**User Story:** As a client, I want a plain-English summary of the top risks, so that I can understand the findings without technical knowledge.

#### Acceptance Criteria

1. WHEN a run completes, THE Summary_Writer SHALL generate a summary of at most 400 words covering the top 5 risks by aggregated Risk_Score, each citing its Evidence.
2. THE Summary_Writer SHALL generate the summary from the run's flags and rollups only, without access to raw unflagged transaction data.
3. THE Summary_Writer SHALL write the AI model call prompt and response verbatim to the run log, so that AI output can be independently verified.
4. WHEN the AI model call fails or times out, THE Summary_Writer SHALL fall back to a deterministic template summary built from the same flag data.
5. THE Summary_Writer SHALL include in every summary a statement of limitations covering rule coverage and the exploratory nature of the ML layer.
6. WHEN a number appears in the summary, THE Summary_Writer SHALL source it from the run's computed statistics, and the summary verification test SHALL confirm each such number appears in the run output.

### Requirement 11: Dashboard

**User Story:** As an auditor, I want an interactive dashboard, so that I can filter, visualize, and drill into flagged transactions.

#### Acceptance Criteria

1. THE Dashboard SHALL display a filterable, sortable table of flagged transactions showing date, account, vendor, approver, amount, risk score, and flag reasons.
2. THE Dashboard SHALL provide filters for date range, account, approver, vendor, and rule name.
3. THE Dashboard SHALL display charts for flag counts by rule, risk distribution by account, and transaction amount over time with flagged points highlighted.
4. WHEN the user selects a flagged transaction, THE Dashboard SHALL display the full Evidence for every flag on that transaction.
5. THE Dashboard SHALL display the generated audit summary text.
6. WHEN no flags match the active filters, THE Dashboard SHALL display an explicit empty-state message.

### Requirement 12: Test Dataset and Detection Tests

**User Story:** As a developer, I want a synthetic dataset with Planted_Anomalies and tests proving detection, so that the Tool's accuracy is demonstrated.

#### Acceptance Criteria

1. THE Tool SHALL include a generator producing synthetic transactions with columns date, time, account, amount, approver, vendor, and description.
2. THE generator SHALL accept a count parameter and a seed for reproducibility, and identical seeds SHALL produce identical datasets.
3. THE generator SHALL support planting anomalies of each rule type: statistical outlier, Benford violation, duplicate payment, weekend/after-hours posting, and round amount.
4. THE Tool SHALL include tests that run the full pipeline on generated datasets and assert each Planted_Anomaly type is detected with at least the recall stated in Requirements 2-7.
5. THE Tool SHALL include a test asserting that a clean synthetic dataset with no Planted_Anomalies produces no false statistical_outlier, duplicate_payment, or round_amount flags beyond 5% of rows.

### Requirement 13: Documentation

**User Story:** As an interviewer, I want a readme explaining design choices, limitations, and verification of AI output, so that I can evaluate the project.

#### Acceptance Criteria

1. THE Tool SHALL include a readme covering setup, usage, data format, design choices, and limitations.
2. THE readme SHALL document how the AI summary output was verified, referencing the run-log prompt and response capture from Requirement 10.
3. THE readme SHALL describe each rule's threshold and the rationale for it.
4. THE Tool SHALL include a one-page summary document stating the business problem, the approach, and the decisions the Tool supports.

## Non-Functional Requirements

### Requirement 14: Performance and Portability

**User Story:** As a user, I want the Tool to handle realistic dataset sizes promptly and run locally, so that I can use it on a laptop.

#### Acceptance Criteria

1. WHEN processing a dataset of 100,000 transactions, THE Tool SHALL complete loading, cleaning, all rule checks, ML detection, and risk scoring within 120 seconds on a standard laptop.
2. THE Tool SHALL run from a local Python environment with dependencies declared in a requirements file, without mandatory network access except for the AI model call.
3. WHERE an API key for the AI model is not configured, THE Tool SHALL use the deterministic template summary and note the substitution in the run log.
