
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import joblib
import json
import os
from datetime import datetime
import pipeline

st.set_page_config(
    page_title="FoodMood v3",
    page_icon="🍽️",
    layout="wide"
)

# ---------- STYLING ----------
st.markdown("""
<style>
#MainMenu, footer, header {visibility:hidden;}
.stApp{
background:linear-gradient(135deg,#0f172a 0%, #111827 50%, #1e293b 100%);
}

.hero{
padding:30px;
border-radius:28px;
background:rgba(255,255,255,0.06);
backdrop-filter:blur(12px);
border:1px solid rgba(255,255,255,0.08);
}

.glass{
background:rgba(255,255,255,0.05);
border:1px solid rgba(255,255,255,0.08);
border-radius:24px;
padding:22px;
}

.metric-card{
background:rgba(255,255,255,0.05);
border:1px solid rgba(255,255,255,0.08);
border-radius:20px;
padding:18px;
text-align:center;
}

.mood-card{
text-align:center;
padding:30px;
border-radius:24px;
background:rgba(255,255,255,0.08);
}

.bigemoji{
font-size:80px;
}

.small-label{
opacity:0.7;
font-size:0.9rem;
}
</style>
""", unsafe_allow_html=True)

# ---------- MODEL ----------
@st.cache_resource
def load_assets():
    model = joblib.load("foodmood_model.pkl")
    feature_cols = json.load(open("feature_cols.json"))
    return model, feature_cols

model, feature_cols = load_assets()

LOG_FILE = "meal_log.csv"

# ---------- CHART ----------
def radar_chart(n):
    labels = ["Protein","Carbs","Fat","Sugar","Fiber"]
    values = [
        n["protein"],
        n["carbs"],
        n["fat"],
        n["sugar"],
        n.get("fiber",0)
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=values,
            theta=labels,
            fill="toself"
        )
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        polar=dict(
            bgcolor="rgba(0,0,0,0)"
        ),
        showlegend=False,
        height=450
    )

    return fig

# ---------- HERO ----------
st.markdown("""
<div class='hero'>
<h1>🍽️ FoodMood</h1>
<p>AI-powered prediction of post-meal alertness, energy, and cognitive performance.</p>
</div>
""", unsafe_allow_html=True)

st.write("")

page = st.radio(
    "",
    ["Prediction","Analytics"],
    horizontal=True,
    label_visibility="collapsed"
)

# ======================================================
# PREDICTION
# ======================================================
if page == "Prediction":

    left, right = st.columns([1,1])

    with left:

        st.markdown("### Describe Your Meal")

        meal = st.text_area(
            "",
            placeholder="Chicken breast, brown rice and vegetables",
            height=140,
            label_visibility="collapsed"
        )

        examples = [
            "Chicken breast and brown rice",
            "Chocolate cake and cola",
            "Chicken biryani with raita",
            "Greek yogurt and banana"
        ]

        example = st.selectbox(
            "Quick Examples",
            ["Select example"] + examples
        )

        if example != "Select example":
            meal = example

        analyze = st.button(
            "✨ Analyze Meal",
            use_container_width=True
        )

    with right:

        if not analyze:
            st.markdown("""
            <div class='glass'>
            <h3>How it works</h3>
            <p>
            FoodMood analyzes nutritional composition,
            engineers dietary features,
            and predicts likely post-meal alertness.
            </p>
            </div>
            """, unsafe_allow_html=True)

        else:

            result = pipeline.predict_mood(
                meal,
                model,
                feature_cols
            )

            if result is None:
                st.error("Meal not recognized.")
                st.stop()

            n = result["nutrition"]

            st.markdown(
                f"""
                <div class='mood-card'>
                <div class='bigemoji'>{result['emoji']}</div>
                <h2>{result['label']}</h2>
                <div class='small-label'>
                Confidence {result['confidence']}%
                </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.progress(result["confidence"]/100)

            st.write("")

            c1,c2,c3,c4 = st.columns(4)

            c1.metric("Calories", int(n["calories"]))
            c2.metric("Protein", f"{n['protein']:.1f} g")
            c3.metric("Fat", f"{n['fat']:.1f} g")
            c4.metric("Sugar", f"{n['sugar']:.1f} g")

            st.write("")

            st.markdown("### 🧠 AI Insight")

            insight = result.get(
                "insight",
                "No insight available."
            )

            st.info(insight)

            score = result["features"]["balance_score"]

            if score > 0.85:
                grade = "Excellent"
            elif score > 0.70:
                grade = "Good"
            elif score > 0.55:
                grade = "Average"
            else:
                grade = "Poor"

            st.success(f"Meal Quality: {grade}")

            st.markdown("### Nutrition Radar")
            st.plotly_chart(
                radar_chart(n),
                use_container_width=True
            )

            row = pd.DataFrame([{
                "timestamp": datetime.now(),
                "meal": meal,
                "mood": result["label"],
                "calories": n["calories"]
            }])

            if os.path.exists(LOG_FILE):
                old = pd.read_csv(LOG_FILE)
                row = pd.concat([old,row])

            row.to_csv(LOG_FILE,index=False)

# ======================================================
# ANALYTICS
# ======================================================
else:

    st.markdown("## Personal Analytics")

    if not os.path.exists(LOG_FILE):

        st.info(
            "Analyze meals first to build analytics."
        )

    else:

        df = pd.read_csv(LOG_FILE)

        m1,m2,m3 = st.columns(3)

        m1.metric(
            "Meals Logged",
            len(df)
        )

        m2.metric(
            "Average Calories",
            round(df["calories"].mean(),0)
        )

        m3.metric(
            "Most Recent",
            df.iloc[-1]["mood"]
        )

        st.write("")

        st.markdown("### Recent Meals")

        st.dataframe(
            df.tail(20),
            use_container_width=True
        )

        if len(df) > 1:

            st.markdown("### Calories Trend")

            st.line_chart(
                df["calories"]
            )
