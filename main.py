"""Health Insurance Premium Estimator Pro - complete single-file version.
Run: streamlit run main.py"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

FEATURES = ["age", "family_size", "bmi", "smoker", "alcohol", "exercise",
            "pre_existing", "city_tier", "cover_lakh"]
LABELS = {"age": "Age", "family_size": "Family size", "bmi": "BMI", "smoker": "Smoking",
          "alcohol": "Alcohol", "exercise": "Exercise", "pre_existing": "Pre-existing condition",
          "city_tier": "City tier", "cover_lakh": "Coverage amount"}
# "Healthy baseline" used to measure how much each factor adds to the premium
BASELINE = {"bmi": 22.0, "smoker": 0, "alcohol": 0, "exercise": 1, "pre_existing": 0}


def make_data(n=6000, seed=42):
    """Synthetic data (non-linear age effect + interactions). Swap with a real CSV if available."""
    r = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age": r.integers(18, 66, n), "family_size": r.integers(1, 7, n),
        "bmi": r.normal(26, 4.5, n).clip(16, 44), "smoker": r.binomial(1, .25, n),
        "alcohol": r.binomial(1, .3, n), "exercise": r.binomial(1, .45, n),
        "pre_existing": r.binomial(1, .2, n), "city_tier": r.integers(1, 4, n),
        "cover_lakh": r.choice([3, 5, 10, 15, 25, 50], n)})
    base = 2500 + 1800 * np.exp(df.age / 22) + df.family_size * 1900
    base = base + df.cover_lakh * 380 + (3 - df.city_tier) * 700
    risk = 1 + .35 * df.smoker + .08 * df.alcohol + .30 * df.pre_existing
    risk = risk + .02 * (df.bmi - 22).clip(0) - .08 * df.exercise + .10 * df.smoker * (df.age > 40)
    df["premium"] = (base * risk + r.normal(0, 800, n)).clip(2500).round(0)
    return df


def train_models():
    df = make_data()
    X, y = df[FEATURES], df["premium"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=1)
    models = {"Linear Regression": LinearRegression(),
              "Random Forest": RandomForestRegressor(150, max_depth=12, n_jobs=-1, random_state=1),
              "Gradient Boosting": GradientBoostingRegressor(random_state=1)}
    rows, preds = [], {}
    for name, m in models.items():
        m.fit(Xtr, ytr)
        p = m.predict(Xte)
        preds[name] = p
        rows.append({"Model": name, "R2": r2_score(yte, p), "MAE": mean_absolute_error(yte, p)})
    metrics = pd.DataFrame(rows).sort_values("R2", ascending=False).reset_index(drop=True)
    return models, metrics, (yte.values, preds), df


def predict(model, profile: dict) -> float:
    return float(model.predict(pd.DataFrame([profile])[FEATURES])[0])


def factor_impact(model, profile: dict) -> pd.Series:
    """Model-agnostic: premium change caused by each factor vs a healthy baseline."""
    now = predict(model, profile)
    out = {}
    for f, b in BASELINE.items():
        p = dict(profile); p[f] = b
        out[LABELS[f]] = now - predict(model, p)
    return pd.Series(out).sort_values()


def risk_score(p: dict) -> int:
    s = 10 + (p["age"] - 18) * .6 + p["smoker"] * 25 + p["alcohol"] * 6 + p["pre_existing"] * 18
    s += max(p["bmi"] - 22, 0) * 1.6 - p["exercise"] * 8
    return int(np.clip(s, 0, 100))


def projection(model, profile: dict, years=10, inflation=.08):
    rows = []
    for i in range(years + 1):
        q = dict(profile); q["age"] = min(profile["age"] + i, 70)
        rows.append({"Year": i, "Age": q["age"],
                     "Premium": predict(model, q) * (1 + inflation) ** i})
    return pd.DataFrame(rows)


# ==================== STREAMLIT APP ====================
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(page_title="Premium Estimator Pro", page_icon="🩺", layout="wide")

st.markdown("""
<style>
@keyframes grad {0%{background-position:0% 50%}50%{background-position:100% 50%}100%{background-position:0% 50%}}
@keyframes fadeUp {from{opacity:0;transform:translateY(20px)}to{opacity:1;transform:none}}
.hero{padding:28px;border-radius:16px;color:#fff;background:linear-gradient(120deg,#0f766e,#2563eb,#7c3aed);
background-size:300% 300%;animation:grad 8s ease infinite, fadeUp .8s ease;margin-bottom:18px}
.hero h1{margin:0;font-size:2rem}.hero p{margin:6px 0 0;opacity:.9}
.card{padding:16px;border-radius:12px;border:1px solid rgba(128,128,128,.3);animation:fadeUp .9s ease;text-align:center}
.card h2{margin:0}.card span{opacity:.7;font-size:.85rem}
</style>
<div class="hero"><h1>🩺 Health Insurance Premium Estimator Pro</h1>
<p>Machine-learning premium prediction with explainability, what-if analysis and 10-year projection</p></div>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Training models...")
def load():
    return train_models()


models, metrics, (y_true, preds), data = load()

# ---------------- Sidebar inputs ----------------
sb = st.sidebar
sb.header("Customer profile")
model_name = sb.selectbox("Model", list(models), index=list(models).index(metrics.loc[0, "Model"]))
age = sb.slider("Age", 18, 65, 30)
family = sb.number_input("Family size", 1, 6, 2)
h = sb.number_input("Height (cm)", 120, 220, 165)
w = sb.number_input("Weight (kg)", 30, 200, 65)
city = sb.selectbox("City tier", [1, 2, 3], index=1, help="1 = metro, 3 = smaller town")
cover = sb.selectbox("Coverage (₹ lakh)", [3, 5, 10, 15, 25, 50], index=2)
smoker = sb.toggle("Smoker")
alcohol = sb.toggle("Drinks alcohol")
exercise = sb.toggle("Exercises regularly", value=True)
pre = sb.toggle("Pre-existing condition")
bmi = round(w / (h / 100) ** 2, 1)
profile = dict(age=age, family_size=family, bmi=bmi, smoker=int(smoker), alcohol=int(alcohol),
               exercise=int(exercise), pre_existing=int(pre), city_tier=city, cover_lakh=cover)
model = models[model_name]
premium = predict(model, profile)
risk = risk_score(profile)

t1, t2, t3, t4, t5 = st.tabs(["💰 Estimate", "🔍 Insights", "🧪 What-if", "📈 10-year projection", "🤖 Model lab"])

# ---------------- Tab 1: estimate ----------------
with t1:
    c1, c2, c3 = st.columns(3)
    c1.markdown(f'<div class="card"><h2>₹ {premium:,.0f}</h2><span>Annual premium</span></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="card"><h2>₹ {premium/12:,.0f}</h2><span>Per month</span></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="card"><h2>{bmi}</h2><span>BMI</span></div>', unsafe_allow_html=True)
    g1, g2 = st.columns(2)
    fig = go.Figure(go.Indicator(mode="gauge+number", value=risk, title={"text": "Risk score"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#2563eb"},
               "steps": [{"range": [0, 35], "color": "#bbf7d0"}, {"range": [35, 65], "color": "#fde68a"},
                         {"range": [65, 100], "color": "#fecaca"}]}))
    fig.update_layout(height=300, margin=dict(t=60, b=10), transition={"duration": 800})
    g1.plotly_chart(fig, use_container_width=True)
    comp = pd.DataFrame({"Item": ["Your premium", "Dataset average"], "₹": [premium, data.premium.mean()]})
    g2.plotly_chart(px.bar(comp, x="Item", y="₹", color="Item", title="Compared with average").update_layout(
        height=300, showlegend=False), use_container_width=True)
    report = pd.DataFrame([{**profile, "estimated_premium": round(premium), "risk_score": risk, "model": model_name}])
    st.download_button("⬇️ Download quote (CSV)", report.to_csv(index=False), "premium_quote.csv", "text/csv")
    if st.button("🎉 Celebrate a healthy profile") and risk < 35:
        st.balloons()

# ---------------- Tab 2: insights ----------------
with t2:
    imp = factor_impact(model, profile)
    fig = px.bar(imp, orientation="h", title="How much each factor adds to YOUR premium (₹ vs healthy baseline)",
                 color=imp.values, color_continuous_scale="RdYlGn_r")
    fig.update_layout(height=380, coloraxis_showscale=False, showlegend=False, xaxis_title="₹ impact", yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Positive = this factor increases your premium. Computed by resetting one factor at a time.")
    rf = models["Random Forest"]
    gi = pd.Series(rf.feature_importances_, index=[LABELS[f] for f in FEATURES]).sort_values()
    st.plotly_chart(px.bar(gi, orientation="h", title="Global feature importance (Random Forest)").update_layout(
        height=350, showlegend=False, xaxis_title="importance", yaxis_title=""), use_container_width=True)

# ---------------- Tab 3: what-if ----------------
with t3:
    st.subheader("Change a habit, see the saving")
    a, b = st.columns(2)
    q_smoke = a.checkbox("Quit smoking", disabled=not smoker)
    q_alc = a.checkbox("Stop alcohol", disabled=not alcohol)
    start_ex = a.checkbox("Start exercising", disabled=bool(exercise))
    target_bmi = b.slider("Target BMI", 18.0, 35.0, float(min(bmi, 24.9)))
    new_cover = b.selectbox("Different coverage (₹ lakh)", [3, 5, 10, 15, 25, 50], index=[3, 5, 10, 15, 25, 50].index(cover))
    p2 = dict(profile, bmi=target_bmi, cover_lakh=new_cover)
    if q_smoke: p2["smoker"] = 0
    if q_alc: p2["alcohol"] = 0
    if start_ex: p2["exercise"] = 1
    new_prem = predict(model, p2)
    st.metric("New estimated premium", f"₹ {new_prem:,.0f}", f"{new_prem - premium:,.0f} vs now", delta_color="inverse")
    st.plotly_chart(px.bar(x=["Current", "After changes"], y=[premium, new_prem], color=["Current", "After changes"],
                           labels={"x": "", "y": "₹ per year"}).update_layout(height=320, showlegend=False),
                    use_container_width=True)

# ---------------- Tab 4: projection (animated) ----------------
with t4:
    infl = st.slider("Medical inflation (%)", 0, 15, 8) / 100
    proj = projection(model, profile, 10, infl)
    fig = go.Figure(
        data=[go.Scatter(x=proj.Year[:1], y=proj.Premium[:1], mode="lines+markers")],
        layout=go.Layout(xaxis=dict(range=[0, 10], title="Years from now"),
                         yaxis=dict(range=[0, proj.Premium.max() * 1.1], title="₹ per year"),
                         title="Premium growth (press Play)", height=420,
                         updatemenus=[dict(type="buttons", showactive=False, x=0.05, y=1.15,
                                           buttons=[dict(label="▶ Play", method="animate",
                                                         args=[None, {"frame": {"duration": 400, "redraw": True},
                                                                      "fromcurrent": True}])])]),
        frames=[go.Frame(data=[go.Scatter(x=proj.Year[:i + 1], y=proj.Premium[:i + 1], mode="lines+markers")])
                for i in range(len(proj))])
    st.plotly_chart(fig, use_container_width=True)
    st.metric("Total paid over 10 years", f"₹ {proj.Premium.iloc[1:].sum():,.0f}")
    st.dataframe(proj.round(0), use_container_width=True, hide_index=True)

# ---------------- Tab 5: model lab ----------------
with t5:
    st.subheader("Model comparison")
    st.dataframe(metrics.style.format({"R2": "{:.3f}", "MAE": "₹ {:,.0f}"}), use_container_width=True, hide_index=True)
    best = metrics.loc[0, "Model"]
    fig = px.scatter(x=y_true, y=preds[best], opacity=.4, labels={"x": "Actual premium", "y": "Predicted premium"},
                     title=f"Actual vs predicted ({best})")
    fig.add_shape(type="line", x0=y_true.min(), y0=y_true.min(), x1=y_true.max(), y1=y_true.max(), line=dict(dash="dash"))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Trained on a synthetic dataset of 6,000 customers. Estimates are illustrative, not an insurer quote.")
