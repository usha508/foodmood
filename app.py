
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import joblib
import json
import os
import sys
from datetime import datetime
sys.path.insert(0, ".")
import pipeline

st.set_page_config(
    page_title="FoodMood 🌸",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: "DM Sans", sans-serif; }
h1, h2, h3 { font-family: "DM Serif Display", serif; }

#MainMenu { visibility: hidden; }
header { visibility: hidden; }
footer { visibility: hidden; }
[data-testid="stToolbar"]    { display: none; }
[data-testid="stDecoration"] { display: none; }
[data-testid="stHeader"]     { display: none; }
.block-container { padding-top: 1.5rem !important; }

.stApp { background: #1e1030; color: #f5e6ff; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #27134a 0%, #1e1030 100%);
    border-right: 2px solid #7c3aed;
}

.mood-card {
    border-radius: 22px;
    padding: 30px;
    margin-bottom: 16px;
    text-align: center;
    border: 1px solid rgba(244,114,182,0.3);
}
.stat-box {
    background: #2a1045;
    border-radius: 14px;
    padding: 16px;
    border: 1px solid #7c3aed;
    text-align: center;
    margin: 4px;
}
.tip-box {
    background: #2a1045;
    border-radius: 10px;
    padding: 13px 16px;
    margin: 7px 0;
    border-left: 4px solid #f472b6;
    font-size: 0.9rem;
    color: #f9d8ff;
}
.log-row {
    background: #2a1045;
    border-radius: 10px;
    padding: 12px 16px;
    margin: 5px 0;
    border-left: 3px solid #e879f9;
    font-size: 0.88rem;
    color: #f5e6ff;
}
.insight-card {
    background: #2a1045;
    border-radius: 14px;
    padding: 18px 20px;
    margin: 8px 0;
    border: 1px solid #7c3aed;
}
.stButton > button {
    background: linear-gradient(135deg, #9333ea, #f472b6);
    color: #ffffff;
    border: none;
    border-radius: 14px;
    padding: 14px 28px;
    font-weight: 700;
    width: 100%;
    font-size: 1rem;
    box-shadow: 0 4px 18px rgba(244,114,182,0.3);
}
.stButton > button:hover { opacity: 0.88; }
label { color: #f9a8d4 !important; font-weight: 600 !important; }
[data-testid="stTextArea"] textarea {
    background: #2a1045 !important;
    border: 1.5px solid #7c3aed !important;
    color: #f5e6ff !important;
    border-radius: 12px !important;
}
[data-testid="stSelectbox"] > div > div {
    background: #2a1045 !important;
    border-color: #7c3aed !important;
    color: #f5e6ff !important;
    border-radius: 10px !important;
}
[data-testid="stMetric"] {
    background: #2a1045;
    border-radius: 12px;
    padding: 10px;
    border: 1px solid #7c3aed;
}
[data-testid="stMetricValue"] { color: #f472b6 !important; font-weight:700 !important; }
[data-testid="stMetricLabel"] { color: #f9a8d4 !important; }
[data-testid="stExpander"] {
    background: #2a1045;
    border: 1px solid #7c3aed;
    border-radius: 12px;
}
hr { border-color: #7c3aed !important; }
h1 { color: #f9a8d4 !important; }
h2 { color: #f472b6 !important; }
h3 { color: #e879f9 !important; }
p  { color: #f5e6ff !important; }
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #1e1030; }
::-webkit-scrollbar-thumb {
    background: linear-gradient(#9333ea, #f472b6);
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ── Load model
@st.cache_resource
def load_model():
    model        = joblib.load("foodmood_model.pkl")
    feature_cols = json.load(open("feature_cols.json"))
    return model, feature_cols

model, feature_cols = load_model()

# ── Constants
MOOD_LABELS = {
    0: ("Energy Crash",     "⚡💥", "#f472b6"),
    2: ("Sluggish",         "😴🌀", "#e879f9"),
    3: ("Balanced & Alert", "😊⚖️", "#a8f0c8"),
}
MOOD_BG = {
    "Energy Crash":     "linear-gradient(135deg,#3d1060,#5a1080)",
    "Sluggish":         "linear-gradient(135deg,#2d1055,#3d1070)",
    "Balanced & Alert": "linear-gradient(135deg,#2a1045,#3a0f65)",
}
TIPS = {
    "Energy Crash": [
        "⚡ Pair sugary foods with protein or fiber to slow the glucose spike.",
        "🥜 Try adding nuts or Greek yogurt to sweet meals.",
        "💧 Stay hydrated — dehydration worsens energy crashes.",
        "⏰ Avoid high-sugar meals before important work or study sessions.",
    ],
    "Sluggish": [
        "🥗 Heavy fatty meals slow digestion — blood goes to stomach not brain.",
        "🚶 A 10-minute walk after a heavy meal reduces sluggishness.",
        "🍽️ Try smaller portions if you need to stay alert after eating.",
        "🌙 Heavy meals are fine at dinner when you do not need focus.",
    ],
    "Balanced & Alert": [
        "✅ Great choice! This meal supports steady energy and focus.",
        "🧠 High protein meals support neurotransmitter production.",
        "📚 Good time to study, work or exercise after this meal.",
        "🔄 Try to eat balanced meals like this 3x a day for stable mood.",
    ],
}
TIME_TIPS = {
    "Morning (6-11am)":  "🌅 Best: high protein + complex carbs. Avoid sugary cereals — they cause a mid-morning crash.",
    "Afternoon (12-5pm)":"☀️ Best: balanced lunch with lean protein + veggies. Avoid heavy fats — causes afternoon slump.",
    "Evening (6-9pm)":   "🌆 Best: moderate meal, lower carbs. Protein + vegetables. Avoid large portions before sleep.",
    "Night (10pm+)":     "🌙 Best: keep it light. Small protein snack or herbal tea. Heavy meals disrupt sleep quality.",
}

# ── Log file
LOG_FILE = "meal_log.csv"

def load_log():
    if os.path.exists(LOG_FILE):
        try:
            return pd.read_csv(LOG_FILE)
        except:
            pass
    return pd.DataFrame(columns=[
        "timestamp","meal","mood","mood_label",
        "calories","protein","fat","carbs","sugar","time_of_day"
    ])

def save_to_log(meal, result, time_of_day):
    df  = load_log()
    new = pd.DataFrame([{
        "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M"),
        "meal":       meal,
        "mood":       result["mood"],
        "mood_label": result["label"],
        "calories":   result["nutrition"]["calories"],
        "protein":    result["nutrition"]["protein"],
        "fat":        result["nutrition"]["fat"],
        "carbs":      result["nutrition"]["carbs"],
        "sugar":      result["nutrition"]["sugar"],
        "time_of_day":time_of_day,
    }])
    df = pd.concat([df, new], ignore_index=True)
    df.to_csv(LOG_FILE, index=False)

# ── Charts
def macro_chart(nutrition):
    labels = ["Protein","Carbs","Fat","Sugar","Fiber"]
    values = [nutrition["protein"], nutrition["carbs"],
              nutrition["fat"],    nutrition["sugar"],
              nutrition.get("fiber",0)]
    colors = ["#f472b6","#e879f9","#c084fc","#f9a8d4","#a8f0c8"]
    fig = go.Figure(go.Bar(
        x=labels, y=values,
        marker_color=colors,
        text=[f"{v:.1f}g" for v in values],
        textposition="outside",
        textfont=dict(color="#f9a8d4", size=11)
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f9a8d4", family="DM Sans"),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="#3d1060", title="grams"),
        margin=dict(t=20,b=10), height=260, showlegend=False
    )
    return fig

def mood_history_chart(df):
    counts = df["mood_label"].value_counts()
    colors_map = {
        "Energy Crash":     "#f472b6",
        "Sluggish":         "#e879f9",
        "Balanced & Alert": "#a8f0c8",
    }
    fig = go.Figure(go.Pie(
        labels=counts.index.tolist(),
        values=counts.values.tolist(),
        hole=0.5,
        marker_colors=[colors_map.get(l,"#c084fc") for l in counts.index],
        textfont=dict(color="#f5e6ff", size=11)
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f9a8d4", family="DM Sans"),
        margin=dict(t=10,b=10), height=280,
        legend=dict(font=dict(color="#f9a8d4"))
    )
    return fig

def calories_trend_chart(df):
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").tail(10)
    colors_map = {
        "Energy Crash":     "#f472b6",
        "Sluggish":         "#e879f9",
        "Balanced & Alert": "#a8f0c8",
    }
    point_colors = [colors_map.get(m,"#c084fc") for m in df["mood_label"]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["timestamp"].dt.strftime("%d %b %H:%M"),
        y=df["calories"],
        mode="lines+markers",
        line=dict(color="#9333ea", width=2),
        marker=dict(color=point_colors, size=10,
                    line=dict(color="#1e1030", width=2)),
        fill="tozeroy",
        fillcolor="rgba(147,51,234,0.08)"
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f9a8d4", family="DM Sans"),
        xaxis=dict(showgrid=False, tickangle=-30),
        yaxis=dict(showgrid=True, gridcolor="#3d1060", title="Calories"),
        margin=dict(t=10,b=10), height=250
    )
    return fig

def time_mood_chart(df):
    order = ["Morning (6-11am)","Afternoon (12-5pm)","Evening (6-9pm)","Night (10pm+)"]
    pivot = df.groupby(["time_of_day","mood_label"]).size().reset_index(name="count")
    colors_map = {
        "Energy Crash":     "#f472b6",
        "Sluggish":         "#e879f9",
        "Balanced & Alert": "#a8f0c8",
    }
    fig = go.Figure()
    for mood in ["Energy Crash","Sluggish","Balanced & Alert"]:
        mood_data = pivot[pivot["mood_label"]==mood]
        fig.add_trace(go.Bar(
            name=mood,
            x=mood_data["time_of_day"],
            y=mood_data["count"],
            marker_color=colors_map[mood]
        ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f9a8d4", family="DM Sans"),
        barmode="stack",
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="#3d1060"),
        margin=dict(t=10,b=10), height=250,
        legend=dict(font=dict(color="#f9a8d4"))
    )
    return fig

# ── Sidebar
with st.sidebar:
    st.markdown("## 🌸 FoodMood 💕")
    st.markdown("<p style='font-size:0.75rem;color:#f9a8d4;letter-spacing:0.1em'>✨ PREDICT YOUR MOOD AFTER EATING ✨</p>",
                unsafe_allow_html=True)
    st.divider()

    page = st.radio("", ["🔮 Predict","📊 My Patterns","💡 Eating Tips"],
                    label_visibility="collapsed")
    st.divider()

    if page == "🔮 Predict":
        meal_input = st.text_area(
            "What did you eat?",
            placeholder="e.g. scrambled eggs and toast\ne.g. biryani\ne.g. chocolate cake and cola",
            height=110
        )
        time_of_day = st.selectbox(
            "When are you eating?",
            ["Morning (6-11am)","Afternoon (12-5pm)",
             "Evening (6-9pm)","Night (10pm+)"]
        )
        quantity = st.slider(
            "How much did you eat? 🍽️",
            min_value=0.5, max_value=3.0,
            value=1.0, step=0.5,
            format="%.1fx serving"
        )
        save_log = st.checkbox("Save to my meal log 📓", value=True)
        predict_btn = st.button("Predict My Mood →")
    else:
        predict_btn = False
        meal_input  = ""
        time_of_day = "Afternoon (12-5pm)"
        quantity    = 1.0

    st.divider()
    st.markdown("<p style='font-size:0.72rem;color:#f9a8d4'>💜 Powered by Gradient Boosting ML<br>🌸 58 foods · 17 features · 74.7% accuracy</p>",
                unsafe_allow_html=True)

# ════════════════════════════════════════
# PAGE: PREDICT
# ════════════════════════════════════════
if page == "🔮 Predict":
    if not predict_btn:
        st.markdown("# 🌸 How will you feel 2 hours from now?")
        st.markdown("Describe your meal and get a 💕 **personalised mood & energy prediction** powered by machine learning ✨")
        st.markdown("")

        c1, c2, c3 = st.columns(3)
        for col, emoji, label, color, desc in [
            (c1, "⚡💥", "Energy Crash",     "#f472b6", "High sugar meals cause a spike then a sharp drop"),
            (c2, "😴🌀", "Sluggish",         "#e879f9", "Heavy or fatty meals slow you down"),
            (c3, "😊⚖️", "Balanced & Alert", "#a8f0c8", "Well-balanced meals keep you focused and calm"),
        ]:
            with col:
                st.markdown(f"""<div class='stat-box'>
                    <div style='font-size:2rem'>{emoji}</div>
                    <div style='font-weight:600;color:{color};margin:6px 0'>{label}</div>
                    <div style='font-size:0.8rem;opacity:0.7'>{desc}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### 🌸 Try these examples:")
        examples = ["scrambled eggs and toast","chocolate cake and cola",
                    "chicken breast and rice","biryani","dal and rice","cheeseburger and fries"]
        cols = st.columns(3)
        for i, ex in enumerate(examples):
            with cols[i%3]:
                st.markdown(f"<div class='tip-box'>🍽️ {ex}</div>", unsafe_allow_html=True)

    else:
        if not meal_input.strip():
            st.error("Please describe your meal in the sidebar first.")
            st.stop()

        with st.spinner("Analysing your meal... 🌸"):
            result = pipeline.predict_mood(meal_input.strip(), model, feature_cols)

        if result is None:
            st.error(f"Sorry, I do not recognise any foods in '{meal_input}'. Try: chicken, rice, eggs, biryani, dal, pizza...")
            st.stop()

        # Scale by quantity
        if quantity != 1.0:
            for key in result["nutrition"]:
                result["nutrition"][key] = round(
                    result["nutrition"][key] * quantity, 1)

        if save_log:
            save_to_log(meal_input.strip(), result, time_of_day)

        mood_label = result["label"]
        mood_emoji = result["emoji"]
        mood_color = result["color"]
        confidence = result["confidence"]
        nutrition  = result["nutrition"]

        st.markdown(f"## 🍽️ Prediction for: {meal_input[:50]}{'...' if len(meal_input)>50 else ''}")
        st.markdown("<p style='font-size:0.75rem;color:#f9a8d4'>" + time_of_day + " · " + str(int(nutrition['calories'])) + " KCAL</p>", unsafe_allow_html=True)

        st.markdown("")

        left, right = st.columns([1, 1.5])

        with left:
            bg = MOOD_BG.get(mood_label, MOOD_BG["Balanced & Alert"])
            st.markdown(f"""<div class='mood-card' style='background:{bg}'>
                <p style='font-size:0.72rem;color:#f9a8d4;letter-spacing:0.1em;margin:0'>IN ~2 HOURS YOU WILL FEEL</p>
                <div style='font-size:3.5rem;line-height:1.2;margin:8px 0'>{mood_emoji}</div>
                <div style='font-size:1.6rem;font-weight:700;color:{mood_color};font-family:DM Serif Display,serif'>{mood_label}</div>
                <p style='color:#f9a8d4;font-size:0.85rem;margin-top:8px'>{confidence}% confidence</p>
            </div>""", unsafe_allow_html=True)

            st.markdown("#### 💖 What to do")
            for tip in TIPS.get(mood_label, [])[:3]:
                st.markdown(f"<div class='tip-box'>{tip}</div>", unsafe_allow_html=True)

        with right:
            st.markdown("#### 🍽️ Nutrition Breakdown ✨")
            st.plotly_chart(macro_chart(nutrition),
                            use_container_width=True,
                            config={"displayModeBar": False})

            s1,s2,s3,s4 = st.columns(4)
            for col, label, val in zip(
                [s1,s2,s3,s4],
                ["Calories","Protein","Sugar","Fat"],
                [f"{nutrition['calories']:.0f}",
                 f"{nutrition['protein']:.0f}g",
                 f"{nutrition['sugar']:.0f}g",
                 f"{nutrition['fat']:.0f}g"]
            ):
                with col:
                    st.markdown(f"""<div class='stat-box'>
                        <div style='font-size:1.3rem;font-weight:700;color:#f472b6'>{val}</div>
                        <div style='font-size:0.75rem;color:#f9a8d4'>{label}</div>
                    </div>""", unsafe_allow_html=True)

        with st.expander("🔬 See what the ML model used to predict this"):
            features = result["features"]
            f1,f2,f3,f4 = st.columns(4)
            feature_display = [
                ("Protein ratio",    f"{features['protein_ratio']*100:.1f}%"),
                ("Sugar ratio",      f"{features['sugar_ratio']*100:.1f}%"),
                ("Fat ratio",        f"{features['fat_ratio']*100:.1f}%"),
                ("Balance score",    f"{features['balance_score']:.2f}"),
                ("Meal heaviness",   ["Light","Medium","Heavy"][int(features['meal_heaviness'])]),
                ("High sugar",       "Yes" if features['high_sugar'] else "No"),
                ("High protein",     "Yes" if features['high_protein'] else "No"),
                ("Heavy meal",       "Yes" if features['heavy_meal'] else "No"),
            ]
            for i,(name,val) in enumerate(feature_display):
                with [f1,f2,f3,f4][i%4]:
                    st.metric(name, val)

# ════════════════════════════════════════
# PAGE: MY PATTERNS
# ════════════════════════════════════════
elif page == "📊 My Patterns":
    st.markdown("# 📊 My Eating Patterns")
    df = load_log()

    if df.empty:
        st.markdown("### No meals logged yet 🌸")
        st.markdown("Go to **🔮 Predict**, make some predictions with **Save to my meal log** ticked, and come back here!")
        st.markdown("")
        st.markdown("<div class='tip-box'>💡 Log at least 5 meals to see meaningful patterns</div>",
                    unsafe_allow_html=True)
    else:
        total  = len(df)
        crash  = len(df[df["mood_label"]=="Energy Crash"])
        slug   = len(df[df["mood_label"]=="Sluggish"])
        bal    = len(df[df["mood_label"]=="Balanced & Alert"])
        avg_cal= df["calories"].mean()

        st.markdown(f"**{total} meals logged** · Here are your personal food-mood patterns 🌸")
        st.markdown("")

        # Top stats
        m1,m2,m3,m4 = st.columns(4)
        for col, label, val, color in [
            (m1, "Total Meals",   str(total),         "#f472b6"),
            (m2, "Energy Crashes",str(crash),          "#f9a8d4"),
            (m3, "Balanced Meals",str(bal),            "#a8f0c8"),
            (m4, "Avg Calories",  f"{avg_cal:.0f}",    "#e879f9"),
        ]:
            with col:
                st.markdown(f"""<div class='stat-box'>
                    <div style='font-size:1.6rem;font-weight:700;color:{color}'>{val}</div>
                    <div style='font-size:0.8rem;color:#f9a8d4'>{label}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("")

        # Charts row
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🍩 Mood Distribution")
            st.plotly_chart(mood_history_chart(df),
                            use_container_width=True,
                            config={"displayModeBar":False})
        with col2:
            st.markdown("#### 📈 Calories Over Time")
            st.plotly_chart(calories_trend_chart(df),
                            use_container_width=True,
                            config={"displayModeBar":False})

        if len(df) >= 3:
            st.markdown("#### ⏰ Mood by Time of Day")
            st.plotly_chart(time_mood_chart(df),
                            use_container_width=True,
                            config={"displayModeBar":False})

        # Personal insights
        st.markdown("#### 💡 Your Personal Insights")
        if crash > bal:
            st.markdown("<div class='insight-card'>⚡ You have more Energy Crashes than Balanced meals. Try adding more protein to your meals and reducing sugary drinks.</div>",
                        unsafe_allow_html=True)
        if bal > (total * 0.5):
            st.markdown("<div class='insight-card'>🌸 Over half your meals are Balanced — great eating habits! Keep it up.</div>",
                        unsafe_allow_html=True)
        if avg_cal > 600:
            st.markdown("<div class='insight-card'>🍽️ Your average meal is quite heavy ({}kcal). Consider lighter lunches to avoid afternoon sluggishness.</div>".format(int(avg_cal)),
                        unsafe_allow_html=True)
        top_crash = df[df["mood_label"]=="Energy Crash"]["meal"].value_counts()
        if not top_crash.empty:
            st.markdown(f"<div class='insight-card'>🔴 Your most common crash meal: <b>{top_crash.index[0]}</b>. Try swapping for a lower-sugar alternative.</div>",
                        unsafe_allow_html=True)

        # Recent meals log
        st.markdown("#### 📓 Recent Meals")
        for _, row in df.tail(8).iloc[::-1].iterrows():
            emoji = {"Energy Crash":"⚡💥","Sluggish":"😴🌀","Balanced & Alert":"😊⚖️"}.get(row["mood_label"],"😊")
            color = {"Energy Crash":"#f472b6","Sluggish":"#e879f9","Balanced & Alert":"#a8f0c8"}.get(row["mood_label"],"#f9a8d4")
            st.markdown(f"""<div class='log-row'>
                <span style='color:{color};font-weight:600'>{emoji} {row['mood_label']}</span>
                &nbsp;·&nbsp; <b>{str(row['meal'])[:45]}</b>
                &nbsp;·&nbsp; {row['calories']:.0f} kcal
                &nbsp;·&nbsp; <span style='opacity:0.55'>{row['timestamp']}</span>
            </div>""", unsafe_allow_html=True)

        # Clear log button
        st.markdown("")
        if st.button("🗑️ Clear meal log"):
            os.remove(LOG_FILE)
            st.success("Log cleared!")
            st.rerun()

# ════════════════════════════════════════
# PAGE: EATING TIPS
# ════════════════════════════════════════
elif page == "💡 Eating Tips":
    st.markdown("# 💡 Eating Tips by Time of Day")
    st.markdown("Science-backed recommendations to keep your energy and mood stable all day 🌸")
    st.markdown("")

    for time_label, tip in TIME_TIPS.items():
        st.markdown(f"### {time_label}")
        st.markdown(f"<div class='tip-box'>{tip}</div>", unsafe_allow_html=True)
        st.markdown("")

    st.markdown("---")
    st.markdown("#### 🌸 The Golden Rules of Food & Mood")

    rules = [
        ("🥩 Protein = Focus",
         "Aim for 25-35% of calories from protein. It is the most stable energy source and supports neurotransmitter production."),
        ("🍬 Sugar = Crash",
         "High sugar meals cause a spike then a sharp drop. Pair sugar with fiber or protein to slow absorption."),
        ("🌾 Fiber = Stability",
         "Fiber slows glucose absorption. More fiber means a flatter energy curve and more stable mood."),
        ("🏋️ Heavy meals = Sluggish",
         "Your body diverts blood to digestion. Keep lunch medium-sized if you need to work or study after."),
        ("⏰ Timing matters",
         "The same meal affects you differently at 8am vs 10pm. Your circadian rhythm changes how you metabolise food."),
        ("💧 Hydration matters too",
         "Even mild dehydration causes fatigue and poor concentration. Drink water with every meal."),
    ]

    col1, col2 = st.columns(2)
    for i, (title, body) in enumerate(rules):
        with (col1 if i%2==0 else col2):
            st.markdown(f"""<div class='insight-card'>
                <b style='color:#f472b6'>{title}</b>
                <p style='margin:6px 0 0;font-size:0.88rem;color:#f5e6ff'>{body}</p>
            </div>""", unsafe_allow_html=True)
