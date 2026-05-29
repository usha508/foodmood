
import streamlit as st
import pandas as pd
import joblib
import json
import plotly.graph_objects as go
import sys
sys.path.insert(0, ".")
import pipeline

# ── Page config
st.set_page_config(
    page_title="FoodMood",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: "DM Sans", sans-serif; }
h1, h2, h3 { font-family: "DM Serif Display", serif; }

/* Hide Streamlit chrome */
#MainMenu { visibility: hidden; }
header { visibility: hidden; }
footer { visibility: hidden; }
[data-testid="stToolbar"]   { display: none; }
[data-testid="stDecoration"]{ display: none; }
[data-testid="stHeader"]    { display: none; }
.block-container { padding-top: 1.5rem !important; }

/* Background — soft light purple */
.stApp {
    background: #1e1030;
    color: #f9d8ff;
}

/* Sidebar — slightly deeper lavender */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #27134a 0%, #1e1030 100%);
    border-right: 2px solid #7c3aed;
}

/* Mood card */
.mood-card {
    border-radius: 24px;
    padding: 32px;
    margin-bottom: 16px;
    text-align: center;
    border: 2px solid #f9a8d4;
    background: #2a1045;
    box-shadow: 0 4px 24px rgba(249,168,212,0.08);
}

/* Stat boxes */
.stat-box {
    background: #2a1045;
    border-radius: 16px;
    padding: 16px;
    border: 1.5px solid #f9a8d4;
    text-align: center;
    margin: 4px;
    box-shadow: 0 2px 10px rgba(249,168,212,0.06);
}

/* Tip boxes */
.tip-box {
    background: #2a1045;
    border-radius: 12px;
    padding: 13px 16px;
    margin: 7px 0;
    border-left: 4px solid #f472b6;
    font-size: 0.9rem;
    color: #f9a8d4;
}

/* Predict button */
.stButton > button {
    background: linear-gradient(135deg, #f472b6, #e879f9);
    color: #ffffff;
    border: none;
    border-radius: 16px;
    padding: 14px 28px;
    font-weight: 700;
    width: 100%;
    font-size: 1rem;
    letter-spacing: 0.04em;
    box-shadow: 0 4px 18px rgba(244,114,182,0.35);
    transition: all 0.2s;
}
.stButton > button:hover {
    box-shadow: 0 6px 26px rgba(244,114,182,0.5);
    opacity: 0.92;
}

/* Labels */
label { color: #f9a8d4 !important; font-size: 0.85rem !important; font-weight: 600 !important; }

/* Text area */
[data-testid="stTextArea"] textarea {
    background: #2a1045 !important;
    border: 1.5px solid #7c3aed !important;
    color: #f9d8ff !important;
    border-radius: 14px !important;
    font-size: 0.95rem !important;
}
[data-testid="stTextArea"] textarea::placeholder {
    color: #d8b4fe !important;
    opacity: 0.8;
}

/* Selectbox */
[data-testid="stSelectbox"] > div > div {
    background: #2a1045 !important;
    border: 1.5px solid #7c3aed !important;
    color: #f9d8ff !important;
    border-radius: 12px !important;
}

/* Metrics */
[data-testid="stMetric"] {
    background: #2a1045;
    border-radius: 14px;
    padding: 10px;
    border: 1.5px solid #f9a8d4;
}
[data-testid="stMetricValue"] { color: #f472b6 !important; font-weight: 700 !important; }
[data-testid="stMetricLabel"] { color: #f9a8d4 !important; }

/* Expander */
[data-testid="stExpander"] {
    background: #2a1045;
    border: 1.5px solid #f9a8d4;
    border-radius: 14px;
}

/* Divider */
hr { border-color: #f9a8d4 !important; }

/* Headings */
h1 { color: #f9a8d4 !important; }
h2 { color: #f472b6 !important; }
h3 { color: #e879f9 !important; }

/* Paragraph text */
p { color: #e9d5ff !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #1e1030; }
::-webkit-scrollbar-thumb {
    background: linear-gradient(#f472b6, #e879f9);
    border-radius: 10px;
}

/* Plotly chart area */
.js-plotly-plot .plotly .bg { fill: #fff0fb !important; }
</style>""", unsafe_allow_html=True)

# ── Load model
@st.cache_resource
def load_model():
    model        = joblib.load("foodmood_model.pkl")
    feature_cols = json.load(open("feature_cols.json"))
    return model, feature_cols

model, feature_cols = load_model()

# ── Mood colours
MOOD_COLORS = {
    "Energy Crash":     "#f472b6",
    "Sluggish":         "#e879f9",
    "Balanced & Alert": "#db2777",
}

MOOD_BG = {
    "Energy Crash":     "linear-gradient(135deg,#1e1520,#2a1a2e)",
    "Sluggish":         "linear-gradient(135deg,#141828,#1a2038)",
    "Balanced & Alert": "linear-gradient(135deg,#141e20,#1a2a2e)",
}

# ── Tips per mood
TIPS = {
    "Energy Crash": [
        "⚡ Pair sugary foods with protein or fiber to slow the glucose spike.",
        "🥜 Try adding nuts or Greek yogurt to sweet meals.",
        "💧 Stay hydrated — dehydration worsens energy crashes.",
        "⏰ Avoid high-sugar meals before important work or study sessions.",
    ],
    "Sluggish": [
        "🥗 Heavy, fatty meals slow digestion — blood goes to stomach, not brain.",
        "🚶 A 10-minute walk after a heavy meal helps reduce sluggishness.",
        "🍽️ Try smaller portions if you need to stay alert after eating.",
        "🌙 Heavy meals are fine at dinner when you don't need focus.",
    ],
    "Balanced & Alert": [
        "✅ Great choice! This meal supports steady energy and focus.",
        "🧠 High protein meals like this support neurotransmitter production.",
        "📚 Good time to study, work or exercise after this meal.",
        "🔄 Try to eat balanced meals like this 3x a day for stable mood.",
    ],
}

# ── Charts
def macro_chart(nutrition):
    labels = ["Protein", "Carbs", "Fat", "Sugar", "Fiber"]
    values = [nutrition["protein"], nutrition["carbs"],
              nutrition["fat"],    nutrition["sugar"], nutrition["fiber"]]
    colors = ["#00bcd4","#ff9800","#9c27b0","#f44336","#4caf50"]
    fig = go.Figure(go.Bar(
        x=labels, y=values,
        marker_color=colors,
        text=[f"{v:.1f}g" for v in values],
        textposition="outside",
        textfont=dict(color="#c8a870", size=11)
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c8a870", family="Nunito"),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="#3a2a10", title="grams"),
        margin=dict(t=20,b=10), height=260, showlegend=False
    )
    return fig

def confidence_chart(probs, feature_cols):
    mood_map = {0:"Energy Crash", 2:"Sluggish", 3:"Balanced & Alert"}
    labels = [mood_map.get(i, "Balanced") for i in range(4)
              if i in mood_map]
    # get proba for each class the model knows
    classes = list(model.classes_)
    prob_vals = []
    for mood_id in [0, 2, 3]:
        if mood_id in classes:
            idx = classes.index(mood_id)
            prob_vals.append(probs[idx] * 100)
        else:
            prob_vals.append(0)

    colors = ["#ff9800","#9c27b0","#4caf50"]
    fig = go.Figure(go.Bar(
        x=prob_vals, y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{p:.0f}%" for p in prob_vals],
        textposition="outside",
        textfont=dict(color="#c8a870", size=11)
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c8a870", family="Nunito"),
        xaxis=dict(showgrid=False, range=[0,115]),
        yaxis=dict(showgrid=False),
        margin=dict(t=10,b=10,l=10), height=200
    )
    return fig

# ── Sidebar
with st.sidebar:
    st.markdown("## 🌸 FoodMood 💕")
    st.markdown("<p style='font-size:0.75rem;color:#be185d;letter-spacing:0.1em'>✨ PREDICT YOUR MOOD AFTER EATING ✨</p>", unsafe_allow_html=True)
    st.divider()

    meal_input = st.text_area(
        "What did you eat?",
        placeholder="e.g. scrambled eggs and toast\ne.g. biryani\ne.g. chocolate cake and cola",
        height=110
    )
    time_of_day = st.selectbox(
        "When are you eating?",
        ["Morning (6–11am)", "Afternoon (12–5pm)",
         "Evening (6–9pm)",  "Night (10pm+)"]
    )
    predict_btn = st.button("Predict My Mood →")
    st.divider()
    st.markdown("<p style='font-size:0.72rem;color:#be185d'>💜 Powered by Gradient Boosting ML<br>🌸 58 foods · 17 features · 74.7% accuracy</p>", unsafe_allow_html=True)

# ── Main page
if not predict_btn:
    st.markdown("# 🌸 How will you feel 2 hours from now?")
    st.markdown("Describe your meal and get a 💕 **personalised mood & energy prediction** powered by machine learning ✨")
    st.markdown("")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""<div class='stat-box'>
            <div style='font-size:2rem'>⚡💥</div>
            <div style='font-weight:600;color:#f472b6;margin:6px 0'>Energy Crash</div>
            <div style='font-size:0.8rem;opacity:0.6'>High sugar meals cause a spike then a sharp drop</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class='stat-box'>
            <div style='font-size:2rem'>😴🌀</div>
            <div style='font-weight:600;color:#e879f9;margin:6px 0'>Sluggish</div>
            <div style='font-size:0.8rem;opacity:0.6'>Heavy or fatty meals slow you down</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class='stat-box'>
            <div style='font-size:2rem'>😊⚖️</div>
            <div style='font-weight:600;color:#db2777;margin:6px 0'>Balanced & Alert</div>
            <div style='font-size:0.8rem;opacity:0.6'>Well-balanced meals keep you focused and calm</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 🌸 Try these examples:")
    examples = [
        "scrambled eggs and toast",
        "chocolate cake and cola",
        "chicken breast and rice",
        "biryani",
        "dal and rice",
        "cheeseburger and fries",
    ]
    cols = st.columns(3)
    for i, ex in enumerate(examples):
        with cols[i % 3]:
            st.markdown(f"<div class='tip-box'>🍽️ {ex}</div>", unsafe_allow_html=True)

else:
    if not meal_input.strip():
        st.error("Please describe your meal in the sidebar first.")
        st.stop()

    # Run prediction
    with st.spinner("Analysing your meal..."):
        result = pipeline.predict_mood(meal_input.strip(), model, feature_cols)

    if result is None:
        st.error(f"Sorry, I don't recognise any foods in '{meal_input}'. Try being more specific, e.g. 'chicken and rice'.")
        st.markdown("**Foods I know:**")
        food_list = ", ".join(sorted(pipeline.FOOD_DB.keys()))
        st.markdown(f"<small style='opacity:0.6'>{food_list}</small>", unsafe_allow_html=True)
        st.stop()

    mood_label = result["label"]
    mood_emoji = result["emoji"]
    mood_color = result["color"]
    confidence = result["confidence"]
    nutrition  = result["nutrition"]
    nutrition["fiber"] = nutrition.get("fiber", 0)

    # Header
    st.markdown(f"## Prediction for: {meal_input[:50]}{'...' if len(meal_input)>50 else ''}*")
    st.markdown(f"<p style='font-size:0.75rem;opacity:0.5;letter-spacing:0.1em'>{time_of_day} · {nutrition['calories']:.0f} KCAL</p>",
                unsafe_allow_html=True)
    st.markdown("")

    # Main result + charts
    left, right = st.columns([1, 1.5])

    with left:
        bg = MOOD_BG.get(mood_label, "linear-gradient(135deg,#0d2b0d,#1a4a1a)")
        st.markdown(f"""<div class='mood-card' style='background:{bg};border:1px solid {mood_color}'>
            <p style='font-size:0.72rem;opacity:0.5;letter-spacing:0.1em;margin:0'>IN ~2 HOURS YOU WILL FEEL</p>
            <div style='font-size:3.5rem;line-height:1.2;margin:8px 0'>{mood_emoji}</div>
            <div style='font-size:1.6rem;font-weight:700;color:{mood_color};font-family:Playfair Display,serif'>{mood_label}</div>
            <p style='opacity:0.6;font-size:0.85rem;margin-top:8px'>{confidence}% confidence</p>
        </div>""", unsafe_allow_html=True)

        st.markdown("#### 💖 What to do")
        for tip in TIPS.get(mood_label, [])[:3]:
            st.markdown(f"<div class='tip-box'>{tip}</div>", unsafe_allow_html=True)

    with right:
        st.markdown("#### 🍽️ Nutrition Breakdown ✨")
        st.plotly_chart(macro_chart(nutrition),
                        use_container_width=True,
                        config={"displayModeBar": False})

        # Key stats row
        s1, s2, s3, s4 = st.columns(4)
        for col, label, val in zip(
            [s1, s2, s3, s4],
            ["Calories", "Protein", "Sugar", "Fat"],
            [f"{nutrition['calories']:.0f}",
             f"{nutrition['protein']:.0f}g",
             f"{nutrition['sugar']:.0f}g",
             f"{nutrition['fat']:.0f}g"]
        ):
            with col:
                st.markdown(f"""<div class='stat-box'>
                    <div style='font-size:1.3rem;font-weight:700;color:#f0a040'>{val}</div>
                    <div style='font-size:0.75rem;opacity:0.55'>{label}</div>
                </div>""", unsafe_allow_html=True)

    # Feature expander
    with st.expander("🔬 See what the ML model used to predict this"):
        features = result["features"]
        f1, f2, f3, f4 = st.columns(4)
        feature_display = [
            ("Protein ratio",    f"{features['protein_ratio']*100:.1f}%"),
            ("Sugar ratio",      f"{features['sugar_ratio']*100:.1f}%"),
            ("Fat ratio",        f"{features['fat_ratio']*100:.1f}%"),
            ("Balance score",    f"{features['balance_score']:.2f}"),
            ("Meal heaviness",   ["Light","Medium","Heavy"][int(features['meal_heaviness'])]),
            ("High sugar flag",  "Yes" if features['high_sugar'] else "No"),
            ("High protein flag","Yes" if features['high_protein'] else "No"),
            ("Heavy meal flag",  "Yes" if features['heavy_meal'] else "No"),
        ]
        cols = [f1, f2, f3, f4]
        for i, (name, val) in enumerate(feature_display):
            with cols[i % 4]:
                st.metric(name, val)
