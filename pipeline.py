import pandas as pd

MOOD_LABELS = {
    0: ("Low Alertness", "😴", "#f97316"),
    2: ("Moderate Alertness", "🙂", "#8b5cf6"),
    3: ("High Alertness", "⚡", "#22c55e"),
}

# Keep your existing FOOD_DB exactly as-is from your current file.
# Paste the full FOOD_DB from your original pipeline.py below this line.
FOOD_DB = {
    "chicken breast":  {"calories":165,"protein":31,"fat":3.6,"carbs":0,"sugar":0,"fiber":0},
    "salmon":          {"calories":208,"protein":20,"fat":13,"carbs":0,"sugar":0,"fiber":0},
    "grilled salmon":  {"calories":208,"protein":20,"fat":13,"carbs":0,"sugar":0,"fiber":0},
    "eggs":            {"calories":155,"protein":13,"fat":11,"carbs":1.1,"sugar":1.1,"fiber":0},
    "egg":             {"calories":155,"protein":13,"fat":11,"carbs":1.1,"sugar":1.1,"fiber":0},
    "scrambled eggs":  {"calories":149,"protein":10,"fat":11,"carbs":1.6,"sugar":1.4,"fiber":0},
    "tuna":            {"calories":116,"protein":26,"fat":1,"carbs":0,"sugar":0,"fiber":0},
    "beef":            {"calories":250,"protein":26,"fat":15,"carbs":0,"sugar":0,"fiber":0},
    "tofu":            {"calories":76,"protein":8,"fat":4.8,"carbs":1.9,"sugar":0.5,"fiber":0.3},
    "white rice":      {"calories":130,"protein":2.7,"fat":0.3,"carbs":28,"sugar":0,"fiber":0.4},
    "brown rice":      {"calories":123,"protein":2.7,"fat":1,"carbs":25,"sugar":0,"fiber":1.8},
    "rice":            {"calories":130,"protein":2.7,"fat":0.3,"carbs":28,"sugar":0,"fiber":0.4},
    "oatmeal":         {"calories":71,"protein":2.5,"fat":1.5,"carbs":12,"sugar":0.3,"fiber":1.7},
    "oats":            {"calories":389,"protein":17,"fat":7,"carbs":66,"sugar":1,"fiber":10},
    "bread":           {"calories":265,"protein":9,"fat":3.2,"carbs":49,"sugar":5,"fiber":2.7},
    "toast":           {"calories":265,"protein":9,"fat":3.2,"carbs":49,"sugar":5,"fiber":2.7},
    "pasta":           {"calories":131,"protein":5,"fat":1.1,"carbs":25,"sugar":0.6,"fiber":1.8},
    "chapati":         {"calories":297,"protein":8,"fat":5,"carbs":52,"sugar":1,"fiber":3},
    "roti":            {"calories":297,"protein":8,"fat":5,"carbs":52,"sugar":1,"fiber":3},
    "naan":            {"calories":310,"protein":9,"fat":5,"carbs":57,"sugar":3,"fiber":2},
    "broccoli":        {"calories":34,"protein":2.8,"fat":0.4,"carbs":7,"sugar":1.7,"fiber":2.6},
    "spinach":         {"calories":23,"protein":2.9,"fat":0.4,"carbs":3.6,"sugar":0.4,"fiber":2.2},
    "salad":           {"calories":20,"protein":1.5,"fat":0.3,"carbs":3.5,"sugar":1.5,"fiber":2},
    "vegetables":      {"calories":65,"protein":2.5,"fat":0.3,"carbs":13,"sugar":4,"fiber":3},
    "banana":          {"calories":89,"protein":1.1,"fat":0.3,"carbs":23,"sugar":12,"fiber":2.6},
    "apple":           {"calories":52,"protein":0.3,"fat":0.2,"carbs":14,"sugar":10,"fiber":2.4},
    "mango":           {"calories":60,"protein":0.8,"fat":0.4,"carbs":15,"sugar":13.7,"fiber":1.6},
    "cheeseburger":    {"calories":303,"protein":17,"fat":14,"carbs":25,"sugar":7,"fiber":1},
    "burger":          {"calories":295,"protein":17,"fat":14,"carbs":24,"sugar":6,"fiber":1},
    "fries":           {"calories":312,"protein":3.4,"fat":15,"carbs":41,"sugar":0.3,"fiber":3.8},
    "french fries":    {"calories":312,"protein":3.4,"fat":15,"carbs":41,"sugar":0.3,"fiber":3.8},
    "pizza":           {"calories":266,"protein":11,"fat":10,"carbs":33,"sugar":3.6,"fiber":2.3},
    "fried chicken":   {"calories":260,"protein":24,"fat":15,"carbs":9,"sugar":0,"fiber":0.5},
    "chocolate cake":  {"calories":371,"protein":5,"fat":16,"carbs":54,"sugar":36,"fiber":2},
    "cake":            {"calories":357,"protein":4.6,"fat":14,"carbs":54,"sugar":36,"fiber":1},
    "cola":            {"calories":37,"protein":0,"fat":0,"carbs":10,"sugar":10,"fiber":0},
    "soda":            {"calories":37,"protein":0,"fat":0,"carbs":10,"sugar":10,"fiber":0},
    "juice":           {"calories":46,"protein":0.7,"fat":0.2,"carbs":11,"sugar":9,"fiber":0.2},
    "coffee":          {"calories":2,"protein":0.3,"fat":0,"carbs":0,"sugar":0,"fiber":0},
    "milk":            {"calories":61,"protein":3.2,"fat":3.3,"carbs":4.8,"sugar":5,"fiber":0},
    "butter":          {"calories":717,"protein":0.9,"fat":81,"carbs":0.1,"sugar":0.1,"fiber":0},
    "butter chicken":  {"calories":150,"protein":14,"fat":8,"carbs":7,"sugar":3,"fiber":0.5},
    "cheese":          {"calories":402,"protein":25,"fat":33,"carbs":1.3,"sugar":0.5,"fiber":0},
    "yogurt":          {"calories":59,"protein":3.5,"fat":0.4,"carbs":3.6,"sugar":3.2,"fiber":0},
    "greek yogurt":    {"calories":97,"protein":9,"fat":5,"carbs":3.6,"sugar":3.6,"fiber":0},
    "honey":           {"calories":304,"protein":0.3,"fat":0,"carbs":82,"sugar":82,"fiber":0},
    "dal":             {"calories":116,"protein":9,"fat":0.4,"carbs":20,"sugar":1,"fiber":8},
    "biryani":         {"calories":200,"protein":8,"fat":7,"carbs":27,"sugar":2,"fiber":1.5},
    "samosa":          {"calories":262,"protein":4,"fat":14,"carbs":30,"sugar":1.5,"fiber":2.5},
    "idli":            {"calories":58,"protein":2,"fat":0.4,"carbs":12,"sugar":0.5,"fiber":0.5},
    "dosa":            {"calories":133,"protein":3.4,"fat":3.7,"carbs":22,"sugar":0.5,"fiber":0.9},
    "paneer":          {"calories":265,"protein":18,"fat":20,"carbs":3,"sugar":1,"fiber":0},
    "ice cream":       {"calories":207,"protein":3.5,"fat":11,"carbs":24,"sugar":21,"fiber":0.7},
    "sandwich":        {"calories":250,"protein":12,"fat":8,"carbs":35,"sugar":5,"fiber":2},
    "soup":            {"calories":72,"protein":3.8,"fat":2.8,"carbs":8,"sugar":2,"fiber":1.5},
    "noodles":         {"calories":138,"protein":4.5,"fat":2,"carbs":25,"sugar":0.4,"fiber":1.5},
    "sweet":           {"calories":380,"protein":2,"fat":8,"carbs":75,"sugar":60,"fiber":0},
    "rice and dal":    {"calories":140,"protein":6,"fat":0.7,"carbs":27,"sugar":0.5,"fiber":3}
    }

def find_food(food_name):
    food_lower = food_name.lower().strip()
    if food_lower in FOOD_DB:
        return FOOD_DB[food_lower], food_lower

    for key in FOOD_DB:
        if key in food_lower or food_lower in key:
            return FOOD_DB[key], key

    return None, None

def get_nutrition_for_meal(meal_text):
    separators = [" and ", " with ", ", ", " plus "]

    foods = [meal_text]

    for sep in separators:
        new_foods = []
        for food in foods:
            new_foods.extend(food.split(sep))
        foods = new_foods

    foods = [f.strip() for f in foods if f.strip()]

    total = {
        "calories": 0,
        "protein": 0,
        "fat": 0,
        "carbs": 0,
        "sugar": 0,
        "fiber": 0,
    }

    for food in foods:
        result, _ = find_food(food)
        if result:
            for k in total:
                total[k] += result[k]

    return total

def engineer_features(nutrition):
    cal = nutrition["calories"] or 1

    return {
        "protein_ratio": (nutrition["protein"] * 4) / cal,
        "carb_ratio": (nutrition["carbs"] * 4) / cal,
        "fat_ratio": (nutrition["fat"] * 9) / cal,
        "sugar_ratio": nutrition["sugar"] / (nutrition["carbs"] + 0.1),
        "balance_score": 1 - (
            abs((nutrition["protein"] * 4 / cal) - 0.30)
            + abs((nutrition["carbs"] * 4 / cal) - 0.40)
            + abs((nutrition["fat"] * 9 / cal) - 0.30)
        ) / 2,
        "meal_heaviness": 0 if cal < 300 else (1 if cal < 600 else 2),
        "protein_per_calorie": nutrition["protein"] / cal,
        "sugar_per_calorie": nutrition["sugar"] / cal,
        "fat_per_calorie": nutrition["fat"] / cal,
        "carb_per_calorie": nutrition["carbs"] / cal,
        "sugar_protein_ratio": nutrition["sugar"] / (nutrition["protein"] + 0.1),
        "fat_protein_ratio": nutrition["fat"] / (nutrition["protein"] + 0.1),
        "high_sugar": int(nutrition["sugar"] > 20),
        "high_protein": int(nutrition["protein"] > 25),
        "heavy_meal": int(nutrition["calories"] > 500),
        "calories": nutrition["calories"],
        "sugar_absolute": nutrition["sugar"],
    }

def generate_insight(result):
    f = result["features"]

    insights = []

    if f["high_protein"]:
        insights.append(
            "This meal is rich in protein and may support sustained energy."
        )

    if f["high_sugar"]:
        insights.append(
            "The sugar content could lead to a short-term energy spike followed by a crash."
        )

    if f["heavy_meal"]:
        insights.append(
            "The calorie density suggests this meal may feel heavy during digestion."
        )

    if f["balance_score"] > 0.80:
        insights.append(
            "The macronutrient profile is relatively balanced."
        )

    if not insights:
        insights.append(
            "This meal has a moderate nutritional profile without major extremes."
        )

    return " ".join(insights)

def predict_mood(meal_text, model, feature_cols):

    nutrition = get_nutrition_for_meal(meal_text)

    if not nutrition or nutrition["calories"] == 0:
        return None

    features = engineer_features(nutrition)

    cal = nutrition["calories"] or 1
    sugar_pct = (nutrition["sugar"] * 4) / cal
    fat_pct = (nutrition["fat"] * 9) / cal

    # Keep only strong override rules
    if nutrition["sugar"] > 35 or sugar_pct > 0.40:
        pred = 0

    elif fat_pct > 0.60 and cal > 400:
        pred = 2

    else:
        X = pd.DataFrame([features])[feature_cols]
        pred = int(model.predict(X)[0])

    label, emoji, color = MOOD_LABELS.get(
        pred,
        ("Moderate Alertness", "🙂", "#8b5cf6"),
    )

    probs = model.predict_proba(
        pd.DataFrame([features])[feature_cols]
    )[0]

    result = {
        "mood": pred,
        "label": label,
        "emoji": emoji,
        "color": color,
        "confidence": round(float(max(probs)) * 100, 1),
        "nutrition": nutrition,
        "features": features,
    }

    result["insight"] = generate_insight(result)

    return result
