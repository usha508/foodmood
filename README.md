**🍽️ FoodMood — ML-Powered Nutrition & Alertness Prediction**
![Python](https://img.shields.io/badge/Python-3.12-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red) ![ML](https://img.shields.io/badge/Model-Gradient%20Boosting-green) ![Accuracy](https://img.shields.io/badge/Accuracy-74.7%25-orange)
> Predict your post meal alertness, focus, and cognitive energy using machine learning (before you even take a bite).

**📌 Overview:**

FoodMood is an end to end machine learning web application that predicts how a meal will affect your short term cognitive state whether it will leave you energised and focused, moderately alert, or sluggish and crashed. Built with a Gradient Boosting classifier, selected after benchmarking against Logistic Regression and Random Forest, trained on nutritional features, FoodMood bridges the gap between nutrition science and practical daily decision making.
This project was developed as part of an exploration into how nutritional composition influences short term alertness, energy levels, and perceived mood a research area with growing relevance in cognitive science and personalised health.

**🎯 Research Motivation:**

Existing nutrition apps focus on long-term health metrics (calories, macros for weight loss). FoodMood addresses a different and underexplored problem: how does what you eat right now affect your brain in the next 2–4 hours?
This has practical implications for:

-->Students optimising meals around study sessions

-->Professionals managing energy through the workday

-->Athletes timing nutrition around training

-->Anyone experiencing unexplained afternoon energy crashes.

##  Model Selection

Three models were trained and evaluated before selecting the final classifier:

1)
Model: Logistic Regression

Accuarcy: 61%

Notes: Underfit : linear boundaries insufficient for nutritional feature interactions 

2)
Model:Random Forest

Accuracy:71% 

Notes:Good performance but slightly lower accuracy and higher inference time 

3)
Model:Gradient Boosting

Accuracy:74.7%

Best accuracy : selected as final model

Gradient Boosting was chosen for its ability to capture non linear relationships between nutritional features and alertness classes, and its superior performance on the validation set.

### Why 74.7% and not higher?

The accuracy ceiling is an honest reflection of the dataset's nature, not a 
modelling failure. Three factors explain it:

1. **Labels are heuristic, not empirical** :the ground truth (High/Moderate/Low 
Alertness) was derived from nutritional science literature, not from real 
physiological measurements like glucose monitors or EEG alertness scores. 
Noisy labels put a hard ceiling on any model's accuracy regardless of algorithm.

2. **Individual biology is unpredictable** : the same meal affects two people 
differently based on metabolism, sleep, stress, and gut microbiome. A model 
trained on population-level nutrition data cannot fully capture this variance, 
and nor should it claim to.

3. **74.7% is meaningfully above chance** : a random classifier on 3 balanced 
classes scores 33.3%. A majority-class baseline scores 45%. At 74.7%, the 
model is capturing genuine nutritional signal. The gap between Random Forest 
(71%) and Gradient Boosting (74.7%) also confirms the result is not noise.

Future work with real user reported mood labels and continuous glucose data 
would be the correct path to pushing accuracy above 85%.

**Machine Learning Pipeline:**

**Model**:

Algorithm: Gradient Boosting Classifier (scikit-learn)

Accuracy: 74.7% (3-class classification)

Features: 17 engineered nutritional features

Supported foods: 58+ common Indian and international foods

Feature Engineering:

Raw nutritional values (calories, protein, fat, carbs, sugar, fiber) are transformed into 17 meaningful features including:
Feature	Description:

'protein_ratio'	Protein as % of total calories

'sugar_ratio'	Sugar as % of total calories

'fat_ratio'	Fat as % of total calories

'balance_score'	Composite score of macronutrient balance

'meal_heaviness'	Encoded scale: Light / Medium / Heavy

'high_sugar'	Binary flag for high glycaemic load

'high_protein'	Binary flag for protein-dense meals

'heavy_meal'	Binary flag for calorie-dense meals

**Target Classes:**

Class	Meaning:

🟢 High Alertness	Balanced meal — sustained energy and focus

🟡 Moderate Alertness	Heavy meal — mild post-meal sluggishness

🔴 Low Alertness	High sugar meal — energy spike followed by crash

**Application Features:**

**Prediction Page:**

Natural language meal input ("chicken biryani with raita")

Time of day aware recommendations

Serving size adjustment (0.5x to 3x)

Detailed nutrition breakdown with interactive charts

Explainability panel showing top influential ML features

Personalised tips based on predicted mood

**Analytics Page:**

Meal logging with persistent CSV storage

Mood distribution pie chart

Calorie trend over last 10 meals

Mood by time of day stacked bar chart

Personal insights (crash rate, balanced meal rate, avg calories)

Most common crash-causing meal identification

**Insights Page:**

Science backed eating tips by time of day

Six golden rules of food and mood

Practical alternatives for common bad choices

**Getting Started:**

Prerequisites:

bash

Python 3.9+

Installation:

bash

git clone https://github.com/usha508/foodmood.git

cd foodmood

pip install -r requirements.txt

Run on Google Colab

'''python
!pip install streamlit pyngrok -q
from pyngrok import ngrok
ngrok.set_auth_token("YOUR_NGROK_TOKEN")'''

'''import subprocess, time, os
os.chdir("/content/foodmood")
subprocess.Popen(["python", "-m", "streamlit", "run", "app.py",
                  "--server.port", "8501", "--server.headless", "true"],
                 cwd="/content/foodmood")
time.sleep(6)
print(ngrok.connect(8501))'''


**Project Structure:**

foodmood/

│

├── app.py                  # Streamlit web application
                            
├── pipeline.py             # ML pipeline: food DB, feature extraction, prediction
                      
├── foodmood_model.pkl      # Trained Gradient Boosting model
                        
├── feature_cols.json       # Feature column names for inference

├── meal_log.csv            # Auto-generated user meal log (gitignored)

├── requirements.txt        # Python dependencies

└── README.md


**Requirements:**

streamlit

pandas

plotly

scikit-learn

joblib


**Strengths:**

1)End to end pipeline : covers data, feature engineering, model training, and a production-style deployed UI.

2)Explainability : every prediction shows which nutritional features drove the result, not just a label.

3)Practical framing : targets a real, underexplored problem (short-term cognitive impact of food) rather than a generic classification task.

4)Indian food support : includes 20+ Indian dishes (biryani, dal, paneer, roti, idli) rarely found in Western nutrition ML datasets.

5)Personalisation : meal logging and analytics build a personal food-mood profile over time.

6)Time aware recommendations : advice changes based on morning/afternoon/evening context.

**Limitations:**

1)Small food database : only 58+ foods are recognised. Mixed or novel meals may not be identified. A production system would integrate a full nutrition API (e.g. USDA FoodData Central, Nutritionix).

2)No ground truth mood labels : the training labels (High/Moderate/Low Alertness) are derived from nutritional science heuristics, not from actual self-reported mood data collected from users. This is the most significant scientific limitation.

3)Individual variation ignored : factors like gut microbiome, metabolism rate, sleep quality, stress, and medication all affect how food impacts mood. The model treats all users identically.

4)No temporal context : the model predicts based on a single meal in isolation. In reality, what you ate 3 hours ago matters too.

5)Binary feature flags : features like 'high_sugar' and 'heavy_meal' use fixed thresholds that may not generalise across all body types and activity levels.

6)74.7% accuracy ceiling : performance is limited by the heuristic labels and small food set. A larger dataset with real physiological measurements (glucose monitors, EEG alertness scores) would substantially improve this.

 **Future Work:**
Integrate USDA FoodData Central API for unlimited food recognition

Collect real user mood labels via post meal surveys to replace heuristic labels

Add physiological features (sleep hours, activity level) as user inputs

Experiment with deep learning on continuous glucose monitoring data

Multi meal temporal modelling (what you ate today affects tomorrow)

Expand to 200+ Indian regional foods

**Author:**

Built by **Usha Tejasa Nunepally** as an independent ML research project exploring the intersection of nutrition science and machine learning.

**License:**

MIT License : free to use, modify, and build upon with attribution.

**UI structure**

## 📸 Screenshots

### Home
![Home](images/homescreen.png)

### Prediction — Form & Result
![Prediction](images/prediction_form.png)

### Prediction — Nutrition Breakdown
![Nutrition](images/prediction_nutrition.png)

### Prediction — Why This Prediction?
![Factors](images/prediction_factors.png)

### Prediction — ML Features
![ML Features](images/prediction_MLfeatures.png)

### Analytics — Overview
![Analytics](images/analytics_overview.png)

### Analytics — Insights
![Insights](images/analytics_insights.png)

### Analytics — Data
![Data](images/analytics_data.png)

### Insights — Tips
![Tips](images/insights_tips.png)

### Insights — Rules
![Rules](images/insights_rules.png)
