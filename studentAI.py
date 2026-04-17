import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report


# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Student Performance Analyzer", layout="centered")

st.title("📊 Student Performance Analyzer")

st.markdown("""
### 🚀 Overview
This application is a **hybrid AI system** designed to evaluate student performance using:

🧠 **Rule-Based Intelligence**
- Deterministic scoring based on academic behavior

🤖 **Machine Learning Model (Logistic Regression)**
- Predicts performance category using learned patterns

📊 **Evaluation & Explainability**
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- Feature Importance Analysis

---

### 🎯 Goal
To combine human-like rules + machine learning for more **reliable and interpretable academic prediction systems**.
""")

# ---------- SIDEBAR ----------
st.sidebar.title("🧠 System Architecture")
st.sidebar.write("""
Rule Engine → Score Calculation  
ML Model → Logistic Regression  
Fusion Layer → Final Decision  
Explainability → Feature Importance  
Evaluation → Classification Metrics
""")


# ---------- RULE-BASED ----------
def calculate_score(inputs):
    study, sleep, screen, attendance, practice = inputs

    # LOGIC

    score = (
    study * 0.30 +
    sleep * 0.15 +
    attendance * 0.20 +
    practice * 0.20 +
    (1 - screen) * 0.15
) * 100

    return round(max(0, min(score, 100)), 2)


def classify(score):
    if score >= 70:
        return "Good"
    elif score >= 40:
        return "Moderate"
    else:
        return "High Risk"


# ---------- ML MODEL ----------
X = np.array([
    [0.6,0.7,0.3,0.8,0.7],
    [0.3,0.5,0.8,0.5,0.4],
    [0.8,0.8,0.2,0.9,0.85],
    [0.4,0.6,0.7,0.6,0.5],
    [0.7,0.7,0.3,0.75,0.8],
    [0.5,0.6,0.2,0.6,0.74],
    [0.2,0.4,0.9,0.4,0.3],
    [0.9,0.8,0.1,0.95,0.9],
    [0.85,0.75,0.2,0.9,0.88],
    [0.25,0.45,0.85,0.5,0.35],
    [0.65,0.7,0.4,0.8,0.78],
    [0.3,0.5,0.7,0.55,0.45]
])

y = np.array([2,0,2,1,2,1,0,2, 2,0,2,1])

# TRAIN-TEST SPLIT (REAL ML)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3,stratify= y, random_state=42
)

model = LogisticRegression(max_iter=200, random_state=42)
model.fit(X_train, y_train)

# REAL ACCURACY
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro',zero_division= 0)  ### “This warning occurs due to class imbalance or small dataset.
                                                                               ### I handled it using zero_division and stratified splitting to ensure fair evaluation.”
recall = recall_score(y_test, y_pred, average='macro',zero_division= 0)
f1 = f1_score(y_test, y_pred, average='macro',zero_division= 0)
report = classification_report(y_test, y_pred, target_names=["High Risk","Moderate","Good"])



def predict_student(inputs):
    pred = model.predict([inputs])[0]
    return ["High Risk", "Moderate", "Good"][pred]


def generate_suggestion(prediction):
    if prediction == "High Risk":
        return "🚨 Increase study, reduce screen time, improve attendance."
    elif prediction == "Moderate":
        return "⚠ Improve consistency."
    else:
        return "✅ Keep it up!"


# ---------- UI ----------
st.title("🎓 Student Performance Indicator")
st.write("Hybrid Model: Rule-Based + Logistic Regression")

st.subheader("📊 Enter Student Data")

study = st.slider("📚 Study Hours", 0, 10)
sleep = st.slider("😴 Sleep Hours", 0, 10)
screen = st.slider("📱 Screen Time", 0, 10)
attendance = st.slider("🏫 Attendance (%)", 0, 100)
practice = st.slider("📝 Practice Score (%)", 0, 100)


# ---------- ANALYSIS ----------

if st.button("🔍 Analyze Performance"):

    # NORMALIZED INPUT
    inputs = [
        study/10,
        sleep/10,
        screen/10,
        attendance/100,
        practice/100
    ]
    # ML probability first 
    proba = model.predict_proba([inputs])[0]
    confidence = max(proba) * 100
    ml_status = ["High Risk", "Moderate", "Good"][np.argmax(proba)]

# Rule-based
    score = calculate_score(inputs)
    rule_status = classify(score)


    if confidence >= 75:
        confidence_msg = "🟢 High confidence"
    elif confidence >= 60:
        confidence_msg = "🟡 Medium confidence"
    else:
        confidence_msg = "🔴 Low confidence"

    st.write(f"Confidence Level: {confidence_msg}")
     # Default = ML
    final_status = ml_status

# If ML is weak → trust rule
    if confidence < 60:
        final_status = rule_status

# If both disagree strongly → go safe
    elif ml_status != rule_status:
        final_status = "Moderate"

    st.subheader("📈 Results")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Score", score)
        st.write(f"📊 Rule-Based Prediction: **{rule_status}**")

    with col2:
        st.metric("Confidence", f"{confidence:.1f}%")
        st.write(f"🤖 ML Prediction: **{ml_status} ({confidence:.1f}%)**")
        
    st.subheader("🧠 Final AI Decision")
    st.write(f"Final Status: **{final_status}**")


    st.success(generate_suggestion(final_status))

    # GRAPH
    st.subheader("📊 Performance Breakdown")

    labels = ["Study", "Sleep", "Screen (low better)", "Attendance", "Practice"]

    normalized_values = inputs  # already normalized

    fig, ax = plt.subplots()
    ax.bar(labels, normalized_values)
    ax.set_ylabel("Normalized Scale (0 to 1)")
    plt.tight_layout()
    st.pyplot(fig)

    # FIXED FEATURE IMPORTANCE
    st.subheader("📊 Feature Importance")

    #importance = np.mean(np.abs(model.coef_), axis=0)
    importance = np.abs(model.coef_).mean(axis=0)
    importance = importance / importance.sum()

    fig2, ax2 = plt.subplots()
    ax2.barh(labels, importance)
    plt.tight_layout()
    st.pyplot(fig2)

    st.subheader("📊 Confusion Matrix")

    fig3, ax3 = plt.subplots()
    ax3.imshow(cm)
    ax3.set_xlabel("Predicted")
    ax3.set_ylabel("Actual")

# Label ticks
    ax3.set_xticks([0,1,2])
    ax3.set_yticks([0,1,2])
    ax3.set_xticklabels(["High Risk","Moderate","Good"])
    ax3.set_yticklabels(["High Risk","Moderate","Good"])

    for i in range(len(cm)):
      for j in range(len(cm)):
        ax3.text(j, i, cm[i, j], ha="center", va="center")

    st.pyplot(fig3)



    # MODEL INFO
    st.subheader("📊 Model Performance")

    col3, col4 = st.columns(2)

    with col3:
      st.metric("Accuracy", f"{round(accuracy*100,2)}%")
      st.metric("Precision", f"{round(precision*100,2)}%")

    with col4:
      st.metric("Recall", f"{round(recall*100,2)}%")
      st.metric("F1 Score", f"{round(f1*100,2)}%")
    
    st.subheader("📄 Classification Report")
    st.text(report)

    st.markdown("")
    st.warning("⚠ Model trained on small synthetic dataset (demo purpose only)")
    st.write("📊 Model: Logistic Regression")

# ---------- FOOTER ----------
st.markdown("---")
st.markdown("Built using Python, Streamlit & Machine Learning")
