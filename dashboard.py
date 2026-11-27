"""Streamlit dashboard for the AI Audit and Risk Analytics Tool."""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

OUT = Path("output")
st.set_page_config(page_title="AI Audit and Risk Analytics Tool", layout="wide")


@st.cache_data
def load_data(stamp):
    scores = pd.read_csv(OUT / "transaction_scores.csv", dtype={"tx_id": str})
    scores["date"] = pd.to_datetime(scores["date"], errors="coerce")

    with open(OUT / "flags.json", encoding="utf-8") as fh:
        flags = pd.DataFrame(json.load(fh))
    if flags.empty:
        flags = pd.DataFrame(columns=["tx_id", "rule", "threshold", "evidence"])
    flags["tx_id"] = flags["tx_id"].astype(str)
    flags["evidence"] = flags["evidence"].apply(lambda e: json.dumps(e, default=str))

    rules = (
        flags.groupby("tx_id")["rule"]
        .apply(lambda s: ", ".join(sorted(set(s))))
        .rename("rules")
    )
    scores = scores.merge(rules, on="tx_id", how="left")
    scores["rules"] = scores["rules"].fillna("")
    scores["flagged"] = scores["rules"] != ""

    accounts = pd.read_csv(OUT / "account_rollup.csv")
    approvers = pd.read_csv(OUT / "approver_rollup.csv")
    summary_path = OUT / "audit_summary.txt"
    summary = summary_path.read_text(encoding="utf-8") if summary_path.exists() else ""
    return scores, flags, accounts, approvers, summary


flags_file = OUT / "flags.json"
if not flags_file.exists():
    st.error("No results found. Run:  python cli.py data\\sample_transactions.csv")
    st.stop()

scores, flags, accounts, approvers, summary = load_data(flags_file.stat().st_mtime)

st.title("AI Audit and Risk Analytics Tool")
st.caption("Every flag comes with the rule that raised it and the evidence behind it.")

# ---------------- Sidebar filters ----------------
st.sidebar.header("Filters")

valid_dates = scores["date"].dropna()
if not valid_dates.empty:
    picked = st.sidebar.date_input(
        "Date range",
        value=(valid_dates.min().date(), valid_dates.max().date()),
    )
    if isinstance(picked, tuple) and len(picked) == 2:
        start, end = pd.Timestamp(picked[0]), pd.Timestamp(picked[1])
        scores = scores[(scores["date"] >= start) & (scores["date"] <= end)]

account_pick = st.sidebar.multiselect("Account", sorted(scores["account"].astype(str).unique()))
approver_pick = st.sidebar.multiselect("Approver", sorted(scores["approver"].astype(str).unique()))
rule_pick = st.sidebar.multiselect("Rule", sorted(flags["rule"].unique()))
min_risk = st.sidebar.slider("Minimum risk score", 0, int(max(scores["risk_score"].max(), 1)), 0)

view = scores
if account_pick:
    view = view[view["account"].astype(str).isin(account_pick)]
if approver_pick:
    view = view[view["approver"].astype(str).isin(approver_pick)]
if rule_pick:
    view = view[view["rules"].apply(lambda r: any(x in r.split(", ") for x in rule_pick))]
view = view[view["risk_score"] >= min_risk]

# ---------------- KPI row ----------------
flagged = view[view["flagged"]]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Transactions", f"{len(view):,}")
c2.metric("Flagged", f"{len(flagged):,}")
c3.metric("Flag rate", f"{(len(flagged) / len(view) * 100) if len(view) else 0:.1f}%")
c4.metric("Highest risk score", f"{view['risk_score'].max() if len(view) else 0:g}")

# ---------------- Charts ----------------
left, right = st.columns(2)
with left:
    st.subheader("Flags by rule")
    view_flags = flags[flags["tx_id"].isin(view["tx_id"])]
    if view_flags.empty:
        st.info("No flags in this selection.")
    else:
        st.bar_chart(view_flags["rule"].value_counts())
with right:
    st.subheader("Risk score distribution")
    if view.empty:
        st.info("No transactions in this selection.")
    else:
        bins = pd.cut(view["risk_score"], bins=10)
        hist = view.groupby(bins, observed=True).size()
        hist.index = hist.index.astype(str)
        st.bar_chart(hist)

left, right = st.columns(2)
with left:
    st.subheader("Highest-risk accounts")
    top_acc = accounts.sort_values("max_risk_score", ascending=False).head(10)
    st.bar_chart(top_acc.set_index("account")["max_risk_score"])
with right:
    st.subheader("Highest-risk approvers")
    top_app = approvers.sort_values("max_risk_score", ascending=False).head(10)
    st.bar_chart(top_app.set_index("approver")["max_risk_score"])

if accounts["is_small_sample"].any() or approvers["is_small_sample"].any():
    st.caption("Some accounts or approvers have very few transactions, so their scores are less reliable.")

# ---------------- Flagged transactions ----------------
st.subheader("Flagged transactions")
table = flagged.sort_values("risk_score", ascending=False)[
    ["tx_id", "date", "account", "amount", "approver", "vendor", "risk_score", "rules"]
]
st.dataframe(table, width="stretch", hide_index=True)

st.subheader("Why was it flagged?")
if table.empty:
    st.info("Nothing to explain for this selection.")
else:
    chosen = st.selectbox("Pick a transaction", table["tx_id"].tolist())
    detail = flags[flags["tx_id"] == chosen][["rule", "threshold", "evidence"]]
    st.dataframe(detail, width="stretch", hide_index=True)

# ---------------- Summary ----------------
with st.expander("Audit summary", expanded=False):
    st.text(summary or "No summary file found.")
    