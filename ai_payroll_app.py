"""
Welcome to AI PAYROLL ANOMALY DETECTION
Local Government Administration, Nigeria

Flow:  enter state, LGA and your name -> upload payroll CSV -> Analyse -> details lock -> results
Deploy: Streamlit Community Cloud (GitHub repo with this file + requirements.txt)
Libs:   streamlit pandas numpy matplotlib seaborn scikit-learn
"""

import datetime
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

SAMPLE_LGAS = ["Adavi", "Ajaokuta", "Ankpa", "Bassa", "Dekina", "Ibaji", "Idah",
               "Igalamela-Odolu", "Ijumu", "Kabba/Bunu", "Kogi (Koton Karfe)", "Lokoja",
               "Mopa-Muro", "Ofu", "Ogori/Magongo", "Okehi", "Okene", "Olamaboro",
               "Omala", "Yagba East", "Yagba West"]
SAMPLE_WEIGHTS = np.array([6, 4, 5, 4, 6, 3, 3, 3, 3, 4, 3, 9, 2, 4, 2, 4, 8, 4, 3, 3, 3], float)
SAMPLE_WEIGHTS /= SAMPLE_WEIGHTS.sum()
GRADE_BASE = {gl: 30000 + (gl - 1) * 9000 for gl in range(1, 16)}   # sample data only
REQUIRED = ["staff_id", "grade_level", "age", "days_present", "overtime_hours",
            "basic_salary", "allowances", "bank_account"]            # 'lga' is optional
NUMERIC = ["grade_level", "age", "days_present", "overtime_hours",
           "basic_salary", "allowances"]
FEATURES = ["salary_ratio", "allowance_ratio", "account_share_count",
            "days_present", "age", "net_pay", "pay_per_day_present", "overtime_hours"]

CSS = """
<style>
.hero {background: linear-gradient(135deg,#0f766e,#1d4ed8); border-radius: 16px;
       padding: 1.5rem 1.1rem; text-align: center; margin-bottom: 1rem;}
.hero h1 {color: #fff !important; font-size: 1.55rem !important; line-height: 1.3 !important;
          margin: 0 0 .5rem 0 !important; padding: 0 !important;}
.hero p {color: #e6f2ff; margin: 0; font-size: .95rem;}
.card {border: 1px solid rgba(128,128,128,.35); border-radius: 12px;
       padding: .8rem 1rem; margin: .6rem 0;}
.small {font-size: .85rem; opacity: .85;}
</style>
"""
HERO = """
<div class="hero">
  <h1>Welcome to AI PAYROLL ANOMALY DETECTION</h1>
  <p>Find ghost workers, inflated salaries and other payroll irregularities in minutes.</p>
</div>
"""


# ------------------------------------------------------------ data
@st.cache_data(show_spinner=False)
def make_sample(n=300, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "staff_id": [f"KG{200000 + i}" for i in range(n)],
        "lga": rng.choice(SAMPLE_LGAS, n, p=SAMPLE_WEIGHTS),
        "grade_level": rng.integers(1, 16, n),
        "age": rng.integers(22, 60, n),
        "days_present": rng.integers(18, 23, n),
        "overtime_hours": rng.integers(0, 15, n),
    })
    base = df["grade_level"].map(GRADE_BASE)
    df["basic_salary"] = (base * rng.normal(1.0, 0.02, n)).round(2)
    df["allowances"] = (df["basic_salary"] * rng.normal(0.35, 0.03, n)).round(2)
    df["bank_account"] = pd.unique(rng.integers(10**9, 10**10, n * 2))[:n]
    k = max(3, int(n * 0.02))
    pool = list(rng.permutation(n))
    g, inf, du, rt, al = (pool[i * k:(i + 1) * k] for i in range(5))
    df.loc[g, ["days_present", "overtime_hours"]] = 0
    df.loc[inf, "basic_salary"] = (df.loc[inf, "basic_salary"] * rng.uniform(1.8, 3.0, len(inf))).round(2)
    half = len(du) // 2
    df.loc[du[half:half * 2], "bank_account"] = df.loc[du[:half], "bank_account"].values
    df.loc[rt, "age"] = rng.integers(61, 72, len(rt))
    df.loc[al, "allowances"] = (df.loc[al, "allowances"] * rng.uniform(2.5, 4.0, len(al))).round(2)
    return df


def clean_payroll(raw, lga_name):
    """Check columns, fix types, drop unusable rows. Returns (data, notes) or raises ValueError."""
    raw = raw.copy()
    raw.columns = [str(c).strip().lower() for c in raw.columns]
    missing = [c for c in REQUIRED if c not in raw.columns]
    if missing:
        raise ValueError("Your file is missing these columns: " + ", ".join(missing))
    notes = []
    if "lga" not in raw.columns:
        raw["lga"] = lga_name.strip()
        notes.append(f"No 'lga' column found, so all staff are treated as part of {lga_name.strip()}.")
    for c in NUMERIC:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    before = len(raw)
    raw = raw.dropna(subset=REQUIRED)
    raw = raw[raw["grade_level"].between(1, 15) & (raw["basic_salary"] > 0)]
    dropped = before - len(raw)
    if dropped:
        notes.append(f"{dropped} row(s) were skipped because of missing values, "
                     "a grade level outside 1-15, or a zero salary.")
    if len(raw) < 30:
        raise ValueError("At least 30 valid staff rows are needed to train the model.")
    if len(raw) < 100:
        notes.append("Small file: results are less reliable with fewer than 100 staff.")
    return raw[["staff_id", "lga"] + REQUIRED[1:]].reset_index(drop=True), notes


def pay_columns(basic, allowances):
    gross = basic + allowances
    net = gross - (basic * 0.105 + gross * 0.07)    # pension 8% + NHF 2.5% + ~7% tax (simplified)
    return gross, net


def build_features(df):
    d = df.copy()
    medians = d.groupby("grade_level")["basic_salary"].median()   # expected pay learned from the file
    d["gross_pay"], d["net_pay"] = pay_columns(d["basic_salary"], d["allowances"])
    d["salary_ratio"] = d["basic_salary"] / d["grade_level"].map(medians)
    d["allowance_ratio"] = d["allowances"] / d["basic_salary"]
    d["account_share_count"] = d.groupby("bank_account")["staff_id"].transform("count")
    d["pay_per_day_present"] = d["gross_pay"] / (d["days_present"] + 1)
    return d, medians


def reasons(r):
    out = []
    if r["days_present"] == 0: out.append("No attendance (possible ghost worker)")
    if r["salary_ratio"] > 1.3: out.append("Salary far above grade level")
    if r["account_share_count"] > 1: out.append("Bank account shared with another staff")
    if r["age"] > 60: out.append("Past retirement age")
    if r["allowance_ratio"] > 0.7: out.append("Unusually high allowances")
    return out


# ------------------------------------------------------------ model
@st.cache_data(show_spinner=False)
def train(d, sensitivity):
    scaler = StandardScaler().fit(d[FEATURES])
    X = scaler.transform(d[FEATURES])
    model = IsolationForest(n_estimators=300, contamination=sensitivity,
                            random_state=42).fit(X)
    out = d.copy()
    out["model_flag"] = model.predict(X) == -1
    out["risk_score"] = -model.score_samples(X)
    out[["pc1", "pc2"]] = PCA(n_components=2).fit_transform(X)
    rule_text = [reasons(r) for _, r in out.iterrows()]
    out["rule_hit"] = [bool(x) for x in rule_text]
    # flagged = the model says unusual, OR a clear policy rule is broken
    out["flagged"] = out["model_flag"] | out["rule_hit"]
    out["reason"] = ["; ".join(x) if x else ("Statistical outlier - review manually" if m else "")
                     for x, m in zip(rule_text, out["model_flag"])]
    return out, scaler, model


def check_one(d, medians, scaler, model, grade, age, days, overtime, basic, allow, shared):
    gross, net = pay_columns(basic, allow)
    expected = medians.get(grade, GRADE_BASE.get(grade, basic))
    row = {"salary_ratio": basic / expected, "allowance_ratio": allow / basic,
           "account_share_count": 2 if shared else 1, "days_present": days,
           "age": age, "net_pay": net, "pay_per_day_present": gross / (days + 1),
           "overtime_hours": overtime}
    X = scaler.transform(pd.DataFrame([row])[FEATURES])
    score = float(-model.score_samples(X)[0])
    model_flag = bool(model.predict(X)[0] == -1)
    rule_reasons = reasons(row)
    flagged = model_flag or bool(rule_reasons)
    pct = float((d["risk_score"] < score).mean() * 100)
    why = rule_reasons or (["Statistical outlier - review manually"] if model_flag else [])
    return flagged, score, pct, why


def make_report(user, state, lga, d, flagged):
    lines = ["AI PAYROLL ANOMALY DETECTION - SUMMARY REPORT", "=" * 46,
             f"Prepared by : {user}", f"State       : {state}", f"LGA         : {lga}",
             f"Date        : {datetime.date.today():%d %B %Y}", "",
             f"Staff analysed      : {len(d):,}",
             f"Flagged for audit   : {len(flagged):,} ({len(flagged) / len(d) * 100:.1f}%)",
             f"Monthly pay at risk : NGN {flagged.gross_pay.sum():,.0f}", "",
             "TOP 20 HIGHEST-RISK RECORDS", "-" * 46]
    for _, r in flagged.head(20).iterrows():
        lines.append(f"{r.staff_id} | GL{int(r.grade_level)} | NGN {r.gross_pay:,.0f} | {r.reason}")
    lines += ["", "Note: flags are leads for auditors to verify, not proof of fraud."]
    return "\n".join(lines)


# ------------------------------------------------------------ charts
def chart_types(flagged):
    parts = flagged["reason"].str.split("; ").explode()
    counts = parts.value_counts()
    fig, ax = plt.subplots(figsize=(6, 3.5))
    sns.barplot(x=counts.values, y=counts.index, color="#e76f51", ax=ax)
    ax.set_title("What the system found")
    ax.set_xlabel("Staff")
    ax.set_ylabel("")
    return fig


def chart_pay(d):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(d, x="gross_pay", hue="flagged", bins=40,
                 palette={False: "#2a9d8f", True: "#e63946"}, ax=ax)
    ax.set_title("Gross pay distribution (red = flagged)")
    ax.set_xlabel("Gross pay (NGN)")
    return fig


def chart_grade(d):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(d, x="grade_level", y="gross_pay", color="#a8dadc", ax=ax)
    ax.set_title("Gross pay by grade level")
    return fig


def chart_score(d):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(d, x="risk_score", hue="flagged", bins=40,
                 palette={False: "#2a9d8f", True: "#e63946"}, ax=ax)
    ax.set_title("Risk score of every staff member")
    return fig


def chart_pca(d):
    fig, ax = plt.subplots(figsize=(6, 4))
    n, f = d[~d.flagged], d[d.flagged]
    ax.scatter(n.pc1, n.pc2, s=8, c="#2a9d8f", label="Normal")
    ax.scatter(f.pc1, f.pc2, s=26, c="#e63946", marker="x", label="Flagged")
    ax.set_title("Payroll records (PCA view)")
    ax.legend()
    return fig


def chart_lga(d):
    fig, ax = plt.subplots(figsize=(6, 4))
    c = d[d.flagged]["lga"].value_counts()
    sns.barplot(x=c.values, y=c.index, color="#457b9d", ax=ax)
    ax.set_title("Flagged staff per LGA")
    ax.set_xlabel("Flagged staff")
    return fig


def show(fig):
    st.pyplot(fig)
    plt.close("all")


# ------------------------------------------------------------ app
def main():
    st.set_page_config(page_title="AI Payroll Anomaly Detection", page_icon="🔍")
    st.markdown(CSS + HERO, unsafe_allow_html=True)
    ss = st.session_state
    locked = ss.get("locked", False)

    if not locked:
        st.subheader("Step 1: Tell us about your payroll")
        state_in = st.text_input("Name of the state", placeholder="e.g. Kogi", key="state_in")
        lga_in = st.text_input("Name of the LGA", placeholder="e.g. Lokoja", key="lga_in")
        user_in = st.text_input("Your name", placeholder="Your full name", key="user_in")

        st.subheader("Step 2: Upload your payroll CSV document")
        st.markdown(
            '<div class="card small">Your CSV needs these columns: <b>'
            + ", ".join(REQUIRED) + "</b>. An <b>lga</b> column is optional.<br>"
            "<b>Privacy:</b> this is a public demo. Remove real names and replace bank "
            "account numbers with codes (the same account must keep the same code) "
            "before uploading.</div>", unsafe_allow_html=True)
        up = st.file_uploader("Upload payroll CSV document", type="csv", key="up_in")
        use_sample_in = st.checkbox("No file yet? Try it with sample (simulated) data",
                                    key="sample_in")
        st.download_button("Download a CSV template",
                           make_sample(20).to_csv(index=False), "payroll_template.csv")
    else:
        st.subheader("🔒 Your details (locked)")
        st.text_input("Name of the state", value=ss["state"], disabled=True, key="state_lock")
        st.text_input("Name of the LGA", value=ss["lga"], disabled=True, key="lga_lock")
        st.text_input("Your name", value=ss["user"], disabled=True, key="user_lock")
        st.text_input("Payroll file", value=ss["file_label"], disabled=True, key="file_lock")
        st.caption("These details are locked for this report. "
                   "Tap 'Start a new report' below to enter new ones.")

    with st.expander("Settings"):
        sensitivity = st.slider("Sensitivity (% flagged by the AI model)", 1, 15, 7)

    if not locked:
        if st.button("Analyse payroll", type="primary"):
            problems = [lbl for lbl, v in (("the name of the state", state_in),
                                           ("the name of the LGA", lga_in),
                                           ("your name", user_in)) if not v.strip()]
            if problems:
                st.warning("Please enter " + ", ".join(problems) + ".")
            elif up is None and not use_sample_in:
                st.warning("Please upload your payroll CSV, or tick the sample data box.")
            else:
                ss["state"] = state_in.strip()
                ss["lga"] = lga_in.strip()
                ss["user"] = user_in.strip()
                ss["file_bytes"] = up.getvalue() if up is not None else None
                ss["file_label"] = up.name if up is not None else "Sample (simulated) data"
                ss["locked"] = True
                st.rerun()
        st.stop()

    if st.button("Start a new report"):
        for k in ("locked", "state", "lga", "user", "file_bytes", "file_label"):
            ss.pop(k, None)
        st.rerun()

    state, lga, user = ss["state"], ss["lga"], ss["user"]
    try:
        if ss["file_bytes"] is not None:
            raw, notes = clean_payroll(pd.read_csv(io.BytesIO(ss["file_bytes"])), lga)
            if "lga" in raw.columns:
                same = raw["lga"].astype(str).str.strip().str.lower() == lga.strip().lower()
                if 0 < same.sum() < len(raw):
                    if st.checkbox(f"Only analyse the {int(same.sum())} staff listed under {lga}",
                                   value=True):
                        raw = raw[same].reset_index(drop=True)
        else:
            raw, notes = clean_payroll(make_sample(), lga)
            notes.append("Showing simulated sample data. Nothing here is real.")
    except Exception as e:
        st.error(str(e))
        st.stop()
    for n_ in notes:
        st.info(n_)

    feats, medians = build_features(raw)
    d, scaler, model = train(feats, sensitivity / 100)
    flagged = d[d.flagged].sort_values("risk_score", ascending=False)

    st.divider()
    st.success(f"Report for **{user.strip()}** | **{lga.strip()}** LGA, "
               f"**{state.strip()}** State | {datetime.date.today():%d %b %Y}")

    t1, t2, t3, t4, t5 = st.tabs(["📊 Overview", "📈 Charts", "🚩 Flagged staff",
                                  "🔎 Check one staff", "ℹ️ How it works"])

    with t1:
        c1, c2 = st.columns(2)
        c1.metric("Staff analysed", f"{len(d):,}")
        c2.metric("Flagged for audit", f"{len(flagged):,}")
        c3, c4 = st.columns(2)
        c3.metric("% flagged", f"{len(flagged) / len(d) * 100:.1f}%")
        c4.metric("Monthly pay at risk", f"₦{flagged.gross_pay.sum():,.0f}")
        if len(flagged):
            show(chart_types(flagged))
        st.download_button("⬇️ Download summary report (TXT)",
                           make_report(user.strip(), state.strip(), lga.strip(), d, flagged),
                           "payroll_summary_report.txt")

    with t2:
        for fn in (chart_pay, chart_grade, chart_score, chart_pca):
            show(fn(d))
        if d["lga"].nunique() > 1 and len(flagged):
            show(chart_lga(d))

    with t3:
        cols = ["staff_id", "lga", "grade_level", "gross_pay", "days_present",
                "age", "risk_score", "reason"]
        table = flagged[cols].round(2)
        st.write(f"**{len(table)} staff flagged, highest risk first**")
        st.dataframe(table, hide_index=True)
        st.download_button("⬇️ Download flagged list (CSV)",
                           table.to_csv(index=False), "flagged_payroll.csv")

    with t4:
        st.write("Type one staff member's details to see if they would be flagged "
                 "when compared with the payroll you uploaded.")
        grade = st.number_input("Grade level", 1, 15, 8)
        age = st.number_input("Age", 18, 75, 40)
        days = st.number_input("Days present this month", 0, 31, 21)
        overtime = st.number_input("Overtime hours", 0, 100, 4)
        default_basic = float(medians.get(grade, GRADE_BASE[grade]))
        basic = st.number_input("Basic salary (₦)", 1000.0, 5_000_000.0,
                                round(default_basic, 2), step=1000.0, key=f"basic_{grade}")
        allow = st.number_input("Allowances (₦)", 0.0, 5_000_000.0,
                                round(default_basic * 0.35, 2), step=1000.0, key=f"allow_{grade}")
        shared = st.checkbox("Bank account is shared with another staff member")
        if st.button("Check this staff member"):
            flag, score, pct, why = check_one(d, medians, scaler, model, grade, age,
                                              days, overtime, basic, allow, shared)
            if flag:
                st.error("🚩 FLAGGED: this record should be reviewed")
            else:
                st.success("✅ Looks normal")
            st.metric("Risk score", f"{score:.3f}")
            st.write(f"More unusual than **{pct:.0f}%** of staff in this payroll.")
            for w in why:
                st.write("- " + w)

    with t5:
        st.markdown(
            "**How the system works**\n\n"
            "1. **Data:** it reads your payroll CSV, one row per staff member.\n"
            "2. **Clues:** it works out pay compared with the same grade, allowance size, "
            "shared bank accounts and pay per day at work.\n"
            "3. **Model:** an *Isolation Forest* learns what normal records look like and "
            "scores how unusual each one is.\n"
            "4. **Rules:** clear policy breaches, such as past retirement age or no "
            "attendance with pay, are flagged too.\n"
            "5. **Result:** a ranked list with a reason for each flagged record.\n\n"
            "Flags are leads for auditors to check. They are not proof of fraud.")
        st.write("**Preview of the data used**")
        st.dataframe(raw.head(10).round(2), hide_index=True)


if __name__ == "__main__":
    main()
