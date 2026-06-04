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
    page_title="FoodMood",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
html, body, [class*="css"] { font-family: "Segoe UI", sans-serif; }
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}
[data-testid="stToolbar"] { display:none; }
[data-testid="stDecoration"] { display:none; }
.block-container { padding-top: 1rem !important; }
.stApp { background-color: #f8fafc; color: #111827; }
[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e5e7eb; }
h1,h2,h3,h4 { color: #111827 !important; }
.mood-card {
    background: white; border-radius: 18px; padding: 30px;
    border: 1px solid #e5e7eb; box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    text-align: center;
}
.stat-box {
    background: white; border-radius: 14px; padding: 16px;
    border: 1px solid #e5e7eb; text-align: center;
}
.tip-box {
    background: #f8fafc; border-left: 4px solid #111827;
    border-radius: 10px; padding: 12px 16px; margin: 8px 0;
}
.insight-card {
    background: white; border-radius: 14px;
    border: 1px solid #e5e7eb; padding: 18px; margin: 10px 0;
}
.log-row {
    background: white; border: 1px solid #e5e7eb;
    border-radius: 10px; padding: 12px; margin: 6px 0;
}
.stButton > button {
    background: #111827; color: white; border-radius: 12px;
    border: none; font-weight: 600; padding: 12px;
}
.stButton > button:hover { background: #374151; }
[data-testid="stRadio"] > div { display: flex; gap: 8px; flex-direction: row !important; }
[data-testid="stRadio"] label {
    background: #f8fafc; border: 2px solid #bfdbfe;
    border-radius: 20px !important; padding: 8px 16px !important;
    color: #111827 !important; font-weight: 600 !important;
}
[data-testid="stRadio"] label:has(input:checked) {
    background: #111827 !important; color: white !important;
    border-color: #111827 !important;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model        = joblib.load("foodmood_model.pkl")
    feature_cols = json.load(open("feature_cols.json"))
    return model, feature_cols

model, feature_cols = load_model()

MOOD_BG = {
    "Low Alertness":     "linear-gradient(135deg,#ffe0ee,#ffd6e7)",
    "Moderate Alertness":         "linear-gradient(135deg,#e3f2fd,#bbdefb)",
    "High Alertness": "linear-gradient(135deg,#e8f5e9,#c8e6c9)",
}
MOOD_COLORS = {
    "Low Alertness":     "#ff6b9d",
    "Moderate Alertness":         "#4fc3f7",
    "High Alertness": "#81c784",
}
TIPS = {
    "Low Alertness": [
        "🍓 Pair sugary foods with protein or fiber to slow the glucose spike.",
        "🥜 Try adding nuts or Greek yogurt to sweet meals.",
        "🥤 Stay hydrated — dehydration worsens energy crashes.",
        "🍱 Avoid high-sugar meals before important work or study sessions.",
    ],
    "Moderate Alertness": [
        "🥗 Heavy fatty meals slow digestion — blood goes to stomach not brain.",
        "🍽️ Try smaller portions if you need to stay alert after eating.",
        "🌿 A short walk after a heavy meal helps reduce sluggishness.",
        "🌙 Heavy meals are fine at dinner when you do not need focus.",
    ],
    "High Alertness": [
        "✅ Great choice! This meal supports steady energy and focus.",
        "🧠 High protein meals support neurotransmitter production.",
        "📚 Good time to study, work or exercise after this meal.",
        "🔄 Try to eat balanced meals like this 3x a day for stable mood.",
    ],
}
TIME_TIPS = {
    "Morning (6-11am)":   " Best: high protein + complex carbs. Avoid sugary cereals — they cause a mid-morning crash.",
    "Afternoon (12-5pm)": " Best: balanced lunch with lean protein + veggies. Avoid heavy fats — causes afternoon slump.",
    "Evening (6-9pm)":    " Best: moderate meal, lower carbs. Protein + vegetables. Avoid large portions before sleep.",
    "Night (10pm+)":      " Best: keep it light. Small protein snack or herbal tea. Heavy meals disrupt sleep quality.",
}


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

def macro_chart(nutrition):
    labels = ["Protein","Carbs","Fat","Sugar","Fiber"]
    values = [nutrition["protein"], nutrition["carbs"],
              nutrition["fat"], nutrition["sugar"], nutrition.get("fiber",0)]
    colors = ["#ff6b9d","#4fc3f7","#ce93d8","#f48fb1","#81c784"]
    fig = go.Figure(go.Bar(
        x=labels, y=values, marker_color=colors,
        text=[f"{v:.1f}g" for v in values],
        textposition="outside",
        textfont=dict(color="#374151", size=11)
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#374151", family="Segoe UI"),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="#e5e7eb", title="grams"),
        margin=dict(t=20,b=10), height=260, showlegend=False
    )
    return fig

def mood_history_chart(df):
    counts = df["mood_label"].value_counts()
    colors_map = {"Low Alertness":"#ff6b9d","Moderate Alertness":"#4fc3f7","High Alertness":"#81c784"}
    fig = go.Figure(go.Pie(
        labels=counts.index.tolist(), values=counts.values.tolist(), hole=0.5,
        marker_colors=[colors_map.get(l,"#ce93d8") for l in counts.index],
        textfont=dict(color="#374151", size=11)
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#374151"),
        margin=dict(t=10,b=10), height=280,
        legend=dict(font=dict(color="#374151"))
    )
    return fig

def calories_trend_chart(df):
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").tail(10)
    colors_map = {"Low Alertness":"#ff6b9d","Moderate Alertness":"#4fc3f7","High Alertness":"#81c784"}
    point_colors = [colors_map.get(m,"#ce93d8") for m in df["mood_label"]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["timestamp"].dt.strftime("%d %b %H:%M"), y=df["calories"],
        mode="lines+markers", line=dict(color="#111827", width=2),
        marker=dict(color=point_colors, size=10, line=dict(color="white", width=2)),
        fill="tozeroy", fillcolor="rgba(37,99,235,0.08)"
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#374151"),
        xaxis=dict(showgrid=False, tickangle=-30),
        yaxis=dict(showgrid=True, gridcolor="#e5e7eb", title="Calories"),
        margin=dict(t=10,b=10), height=250
    )
    return fig

def time_mood_chart(df):
    pivot = df.groupby(["time_of_day","mood_label"]).size().reset_index(name="count")
    colors_map = {"Low Alertness":"#ff6b9d","Moderate Alertness":"#4fc3f7","High Alertness":"#81c784"}
    fig = go.Figure()
    for mood in ["Low Alertness","Moderate Alertness","High Alertness"]:
        d = pivot[pivot["mood_label"]==mood]
        fig.add_trace(go.Bar(name=mood, x=d["time_of_day"], y=d["count"],
                             marker_color=colors_map[mood]))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#374151"), barmode="stack",
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="#e5e7eb"),
        margin=dict(t=10,b=10), height=250,
        legend=dict(font=dict(color="#374151"))
    )
    return fig

# ── Top bar
# ── Research-style header
st.title("🍽️ FoodMood")

st.caption(
    "Machine Learning Framework for Predicting Post-Meal Cognitive and Emotional States"
)
# ── Navigation tabs
page = st.radio("", ["Prediction","Analytics","Insights"],
                horizontal=True, label_visibility="collapsed")

# ── Sidebar
with st.sidebar:
    st.markdown("### FoodMood AI")
    st.markdown("""
**Model:** Gradient Boosting

**Accuracy:** 74.7%

**Features:** 17 engineered nutrition features

**Foods Supported:** 58+
    """)

st.markdown("")

# ════════════════════════
# PAGE: PREDICTION
# ════════════════════════
if page == "Prediction":
    st.markdown("""
    <div class='insight-card'>
    <h4 style='margin-top:0'>Research Contribution</h4>

    FoodMood explores how nutritional composition
    influences short-term alertness, energy levels,
    and perceived mood through machine learning.

    </div>
    """, unsafe_allow_html=True)
    col_form, col_result = st.columns([1.1, 1])

    with col_form:
        st.markdown("### 🍽️ What did you eat?")
        meal_input = st.text_area(
            "Meal",
            placeholder="""Chicken breast, brown rice and vegetables
            Chocolate cake and cola
            Chicken biryani with raita
            Greek yogurt with berries""",
            height=110, 
            label_visibility="collapsed"
        )
        time_of_day = st.selectbox(
            "🕐 When are you eating?",
            ["Morning (6-11am)","Afternoon (12-5pm)","Evening (6-9pm)","Night (10pm+)"]
        )
        quantity = st.slider(
            "🍽️ How much did you eat?",
            min_value=0.5, max_value=3.0, value=1.0, step=0.5,
            format="%.1fx serving"
        )
        save_log = st.checkbox(" Save to my meal log", value=True)
        predict_btn = st.button("✨ Predict My Mood")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Example Research Cases")
        st.markdown("#### Example Meals")

        examples = [
           ("High Alertness", "Chicken breast, brown rice and vegetables"),
           ("Low Alertness", "Chocolate cake and cola"),
           ("High Alertness", "Greek yogurt with berries"),
           ("Moderate Alertness", "Chicken biryani with raita"),
           ("High Alertness", "Paneer curry with roti")
        ]

        for mood, meal in examples:
           st.markdown(
           f"""
           <div class='tip-box'>
           <b>{mood}</b><br>
           {meal}
           </div>
           """,
           unsafe_allow_html=True
           )
                   
    with col_result:
        if not predict_btn:
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.markdown("### Ready to predict?")
            st.markdown("Describe your meal on the left and click **Predict My Mood** 🍽️")
            st.markdown("<div class='insight-card' style='text-align:center;margin-top:20px'>🌱 <b>Did you know?</b><br>What you eat directly affects your brain chemistry and energy for the next 2-4 hours.</div>",
                        unsafe_allow_html=True)
        else:
            if not meal_input.strip():
                st.error("Please describe your meal first 🍽️")
                st.stop()

            with st.spinner("Analysing your meal..."):
                result = pipeline.predict_mood(meal_input.strip(), model, feature_cols)

            if result is None:
                st.error(f"I do not recognise that meal. Try one of these:")
                known = sorted(pipeline.FOOD_DB.keys())
                cols  = st.columns(3)
                for i, food in enumerate(known):
                    with cols[i%3]:
                        st.markdown(f"<div class='tip-box' style='padding:5px 10px;font-size:0.78rem'>🍱 {food}</div>",
                                    unsafe_allow_html=True)
                st.stop()

            if quantity != 1.0:
                for key in result["nutrition"]:
                    result["nutrition"][key] = round(result["nutrition"][key] * quantity, 1)

            if save_log:
                save_to_log(meal_input.strip(), result, time_of_day)

            mood_label = result["label"]
            mood_color = MOOD_COLORS.get(mood_label, "#ff6b9d")
            confidence = result["confidence"]
            nutrition  = result["nutrition"]
            features   = result["features"]
            emoji_map  = {"Low Alertness":"⚡","Moderate Alertness":"😴","High Alertness":"✅"}
            mood_emoji = emoji_map.get(mood_label, "✅")

            st.markdown(f"""<div class='mood-card'>
                <p style='color:#6b7280;font-size:0.9rem;margin-bottom:8px'>Prediction</p>
                <div style='font-size:3rem;margin-bottom:10px'>{mood_emoji}</div>
                <h2 style='color:{mood_color};margin-bottom:10px'>{mood_label}</h2>
                <p style='font-size:1rem;color:#374151'>Confidence: <strong>{confidence}%</strong></p>
            </div>""", unsafe_allow_html=True)
            st.markdown("##### Model Confidence")

            st.progress(confidence / 100)

            st.caption(
                "Prediction certainty estimated from class probabilities generated by Gradient Boosting model."             
            )
              

            st.markdown("#### 🥗 Nutrition Breakdown")
            st.plotly_chart(macro_chart(nutrition), use_container_width=True,
                            config={"displayModeBar":False})

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Calories",   int(nutrition["calories"]))
            c2.metric("Protein (g)",int(nutrition["protein"]))
            c3.metric("Sugar (g)",  int(nutrition["sugar"]))
            c4.metric("Fat (g)",    int(nutrition["fat"]))
            
            #new section
            st.markdown("### Why this prediction?")

            st.markdown(f"""
            <div class='insight-card'>
            <b>Top Influential Factors</b><br><br>

            Protein Density ............ {features['protein_ratio']*100:.1f}%<br>
            Fat Content ................. {features['fat_ratio']*100:.1f}%<br>
            Sugar Load ................. {features['sugar_ratio']*100:.1f}%<br>
            Balance Score .............. {features['balance_score']:.2f}
            </div>
            """, unsafe_allow_html=True)
            

            st.markdown("#### 🌿 What to do")
            for tip in TIPS.get(mood_label, [])[:3]:
                st.markdown(f"<div class='tip-box'>{tip}</div>", unsafe_allow_html=True)

            if mood_label == "Low Alertness":
                st.info("High sugar content and low fiber levels suggest a rapid glucose spike followed by a likely energy drop.")
            elif mood_label == "Moderate Alertness":
                st.info("This meal is calorie-dense and relatively heavy, which may reduce alertness during digestion.")
            else:
                st.info("The nutritional profile is balanced with sufficient protein and moderate sugar, supporting sustained focus.")

            with st.expander("🔬 ML Features used for this prediction"):
                
                f1,f2,f3,f4 = st.columns(4)
                display = [
                    ("Protein %",   f"{features['protein_ratio']*100:.1f}%"),
                    ("Sugar %",     f"{features['sugar_ratio']*100:.1f}%"),
                    ("Fat %",       f"{features['fat_ratio']*100:.1f}%"),
                    ("Balance",     f"{features['balance_score']:.2f}"),
                    ("Heaviness",   ["Light","Medium","Heavy"][int(features['meal_heaviness'])]),
                    ("High sugar",  "Yes" if features['high_sugar'] else "No"),
                    ("High protein","Yes" if features['high_protein'] else "No"),
                    ("Heavy meal",  "Yes" if features['heavy_meal'] else "No"),
                ]
                for i,(n,v) in enumerate(display):
                    with [f1,f2,f3,f4][i%4]:
                        st.metric(n, v)

# ════════════════════════
# PAGE: ANALYTICS
# ════════════════════════
elif page == "Analytics":
    st.markdown("### 🍩 My Eating Patterns")
    df = load_log()

    if df.empty:
        col2 = st.columns([1,2])
        with col2:
            st.markdown("### No meals logged yet")
            st.markdown("Go to **Prediction**, make predictions with **Save to my meal log** ticked!")
            st.markdown("<div class='tip-box'>🍱 Log at least 5 meals to see meaningful patterns</div>",
                        unsafe_allow_html=True)
    else:
        total   = len(df)
        crash   = len(df[df["mood_label"]=="Low Alertness"])
        bal     = len(df[df["mood_label"]=="High Alertness"])
        avg_cal = df["calories"].mean()

        st.markdown(f"**{total} meals logged** — here are your personal food-mood patterns")
        st.markdown("")

        m1,m2,m3,m4 = st.columns(4)
        m1.metric("Total Meals",   total)
        m2.metric("Energy Crashes", crash)
        m3.metric("Balanced Meals", bal)
        m4.metric("Avg Calories",  f"{avg_cal:.0f}")
        st.markdown("")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Mood Distribution")
            st.plotly_chart(mood_history_chart(df), use_container_width=True,
                            config={"displayModeBar":False})
        with col2:
            st.markdown("#### Calories Over Time")
            st.plotly_chart(calories_trend_chart(df), use_container_width=True,
                            config={"displayModeBar":False})

        if len(df) >= 3:
            st.markdown("#### Mood by Time of Day")
            st.plotly_chart(time_mood_chart(df), use_container_width=True,
                            config={"displayModeBar":False})

        st.markdown("#### Your Personal Insights")
        if crash > bal:
            st.markdown("<div class='insight-card'>⚡ You have more Energy Crashes than Balanced meals. Try adding more protein and reducing sugary drinks.</div>",
                        unsafe_allow_html=True)
        if bal > (total * 0.5):
            st.markdown("<div class='insight-card'>✅ Over half your meals are Balanced — great eating habits!</div>",
                        unsafe_allow_html=True)
        if avg_cal > 600:
            st.markdown(f"<div class='insight-card'>🍽️ Your average meal is quite heavy ({int(avg_cal)} kcal). Consider lighter lunches.</div>",
                        unsafe_allow_html=True)

        top_crash = df[df["mood_label"]=="Low Alertness"]["meal"].value_counts()
        if not top_crash.empty:
            st.markdown(f"<div class='insight-card'>🔴 Most common crash meal: <b>{top_crash.index[0]}</b>. Try a lower-sugar alternative.</div>",
                        unsafe_allow_html=True)

        st.markdown("#### Best Meal For Right Now")
        hour = datetime.now().hour
        if 6 <= hour < 11:
            rec = "🥚 Try: scrambled eggs and toast, oatmeal, or Greek yogurt"
        elif 11 <= hour < 17:
            rec = "🥗 Try: chicken and rice, dal and rice, or grilled salmon"
        elif 17 <= hour < 21:
            rec = "🍲 Try: chicken and vegetables, soup, or paneer with roti"
        else:
            rec = "🌙 Try: Greek yogurt, small oatmeal, or warm milk"
        st.markdown(f"<div class='insight-card'>{rec}</div>", unsafe_allow_html=True)

        st.markdown("#### Recent Meals")
        for _, row in df.tail(8).iloc[::-1].iterrows():
            emoji = {"Low Alertness":"⚡","Moderate Alertness":"😴","High Alertness":"✅"}.get(row["mood_label"],"✅")
            color = {"Low Alertness":"#ff6b9d","Moderate Alertness":"#4fc3f7","High Alertness":"#81c784"}.get(row["mood_label"],"#ff6b9d")
            st.markdown(f"<div class='log-row'><span style='color:{color};font-weight:700'>{emoji} {row['mood_label']}</span> &nbsp;·&nbsp; <b>{str(row['meal'])[:45]}</b> &nbsp;·&nbsp; {row['calories']:.0f} kcal &nbsp;·&nbsp; <span style='opacity:0.5'>{row['timestamp']}</span></div>",
                        unsafe_allow_html=True)

        st.markdown("")
        if st.button("🗑️ Clear my meal log"):
            if os.path.exists(LOG_FILE):
                os.remove(LOG_FILE)
            st.rerun()

# ════════════════════════
# PAGE: NUTRITION INTELLIGENCE
# ════════════════════════
elif page == "Nutrition Intelligence":
    st.markdown("### 🥑 Eating Tips by Time of Day")
    st.markdown("Science-backed recommendations to keep your energy and mood stable all day.")
    st.markdown("")

    for time_label, tip in TIME_TIPS.items():
        st.markdown(f"#### {time_label}")
        st.markdown(f"<div class='tip-box'>{tip}</div>", unsafe_allow_html=True)
        st.markdown("")

    st.markdown("---")
    st.markdown("#### The Golden Rules of Food and Mood")
    rules = [
        ("🥩 Protein = Focus",   "Aim for 25-35% of calories from protein. Most stable energy source."),
        ("🍬 Sugar = Crash",     "High sugar causes a spike then sharp drop. Pair with fiber or protein."),
        ("🌾 Fiber = Stability", "Fiber slows glucose absorption. More fiber means flatter energy curve."),
        ("🍲 Heavy = Sluggish",  "Body diverts blood to digestion. Keep lunch medium if you need focus."),
        ("🕐 Timing matters",    "Same meal at 8am vs 10pm affects you differently due to body clock."),
        ("🥤 Hydration too",     "Mild dehydration causes fatigue. Drink water with every meal."),
    ]
    col1, col2 = st.columns(2)
    for i,(title,body) in enumerate(rules):
        with (col1 if i%2==0 else col2):
            st.markdown(f"<div class='insight-card'><b style='color:#111827'>{title}</b><p style='margin:6px 0 0;font-size:0.88rem;color:#374151'>{body}</p></div>",
                        unsafe_allow_html=True)
