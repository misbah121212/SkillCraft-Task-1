import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ----------------------------------------
# 🌈 PAGE CONFIGURATION + STYLING
# ----------------------------------------
st.set_page_config(page_title="🏠 House Price Predictor", layout="wide")

st.markdown("""
    <style>
    body {
        background: linear-gradient(to right top, #c6ffdd, #fbd786, #f7797d);
    }
    .main {
        background-color: rgba(255,255,255,0.9);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }
    h1, h2, h3 {
        text-align: center;
        color: #333333;
        font-family: 'Trebuchet MS', sans-serif;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------------------------
# 🏡 TITLE
# ----------------------------------------
st.title("🏠 House Price Prediction Dashboard")
st.markdown("### SkillCraft Internship - Task 01 | By **Misba Sikandar** 💫")

# ----------------------------------------
# 📂 LOAD & PREPARE DATA
# ----------------------------------------
@st.cache_data
def load_data():
    data = pd.read_csv("train.csv")
    data = data[["GrLivArea", "BedroomAbvGr", "FullBath", "SalePrice"]]
    data.fillna(data.median(), inplace=True)
    return data

data = load_data()

# ----------------------------------------
# 🔍 SPLIT INTO TRAIN & TEST
# ----------------------------------------
X = data[["GrLivArea", "BedroomAbvGr", "FullBath"]]
y = data["SalePrice"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ----------------------------------------
# 📈 LAYOUT - TWO COLUMNS
# ----------------------------------------
col1, col2 = st.columns(2)

# -------- LEFT COLUMN (DATA + GRAPHS) --------
with col1:
    st.subheader("📊 Dataset Overview")
    st.dataframe(data.head())

    # Correlation Heatmap
    st.subheader("🔥 Feature Correlation Heatmap")
    fig1, ax1 = plt.subplots(figsize=(5,3))
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm", ax=ax1)
    st.pyplot(fig1)

    # Actual vs Predicted
    st.subheader("🎯 Actual vs Predicted Prices")
    fig2, ax2 = plt.subplots(figsize=(5,3))
    ax2.scatter(y_test, y_pred, alpha=0.7, color="#4B8BBE")
    ax2.set_xlabel("Actual Prices")
    ax2.set_ylabel("Predicted Prices")
    ax2.set_title("Actual vs Predicted")
    st.pyplot(fig2)

# -------- RIGHT COLUMN (PREDICTION + METRICS) --------
with col2:
    st.subheader("⚙️ Model Evaluation")
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.metric("MAE", f"{mae:,.2f}")
    st.metric("MSE", f"{mse:,.2f}")
    st.metric("R² Score", f"{r2:.4f}")

    st.markdown("---")
    st.subheader("💰 Predict New House Price")

    sqft = st.number_input("🏡 Living Area (Square Feet)", min_value=500, max_value=10000, value=1500, step=50)
    bedrooms = st.number_input("🛏️ Bedrooms", min_value=1, max_value=10, value=3)
    bathrooms = st.number_input("🛁 Bathrooms", min_value=1, max_value=5, value=2)

    if st.button("🔍 Predict Price"):
        input_data = pd.DataFrame({
            "GrLivArea": [sqft],
            "BedroomAbvGr": [bedrooms],
            "FullBath": [bathrooms]
        })
        prediction = model.predict(input_data)
        st.success(f"🏡 **Estimated Price:** ${prediction[0]:,.2f}")

# ----------------------------------------
# 🧾 FOOTER
# ----------------------------------------
st.markdown("---")
st.caption("✨ Developed by Misba Sikandar | SkillCraft Internship 2025 | Powered by Streamlit & scikit-learn ✨")
