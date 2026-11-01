
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="ChurnIQ | Customer Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- LOAD MODELS ----------------

BASE_DIR = Path(__file__).resolve().parent

model = joblib.load(BASE_DIR / "churn_model.pkl")
preprocessor = joblib.load(BASE_DIR / "preprocessor.pkl")

DATA_PATH = BASE_DIR / "dataset" / "telco_customer_churn.csv"

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #090E1A;
    color: #F1F5F9;
}

[data-testid="stSidebar"] {
    background: #111827;
    border-right: 1px solid #263248;
}

[data-testid="stSidebar"] p {
    color: #CBD5E1;
}

h1, h2, h3 {
    color: #F8FAFC !important;
}

.hero {
    padding: 28px;
    border-radius: 18px;
    background: linear-gradient(120deg, #172554, #312E81, #172554);
    border: 1px solid #3730A3;
    margin-bottom: 25px;
}

.hero h1 {
    font-size: 34px;
    margin-bottom: 8px;
}

.hero p {
    color: #C7D2FE;
    margin: 0;
}

.metric-card {
    background: #151F32;
    padding: 22px;
    border: 1px solid #293750;
    border-radius: 15px;
    min-height: 125px;
}

.metric-label {
    color: #94A3B8;
    font-size: 13px;
}

.metric-value {
    color: #F8FAFC;
    font-size: 29px;
    font-weight: 800;
    margin-top: 12px;
}

.metric-note {
    color: #818CF8;
    font-size: 12px;
    margin-top: 5px;
}

.section-title {
    font-size: 21px;
    font-weight: 700;
    margin-top: 25px;
    margin-bottom: 15px;
}

.stButton > button {
    background: linear-gradient(90deg, #6366F1, #8B5CF6);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 12px 24px;
    font-weight: 700;
    width: 100%;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #4F46E5, #7C3AED);
    color: white;
    border: none;
}

div[data-testid="stForm"] {
    background: #151F32;
    padding: 22px;
    border: 1px solid #293750;
    border-radius: 16px;
}

div[data-testid="stMetric"] {
    background: #151F32;
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #293750;
}

[data-testid="stProgressBar"] > div > div {
    background: #818CF8;
}

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# ---------------- SIDEBAR ----------------

with st.sidebar:
    st.markdown("# ✦ ChurnIQ")
    st.caption("CUSTOMER INTELLIGENCE PLATFORM")

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Dashboard",
            "Predict Churn",
            "Analytics",
            "Model Performance",
            "About Project"
        ]
    )

    st.divider()

    st.markdown("### 🤖 AI Engine")
    st.success("Model Loaded")

    st.caption("Logistic Regression")
    st.caption("Version 1.0")

    st.divider()
    st.caption("Developed with Python & Streamlit")


# ---------------- LOAD DATA ----------------

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

try:
    df = load_data()
except Exception:
    df = None


# ---------------- HELPER: HERO ----------------

def hero(title, subtitle):
    st.markdown(f"""
    <div class="hero">
        <h1>{title}</h1>
        <p>{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


def metric_card(label, value, note):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-note">{note}</div>
    </div>
    """, unsafe_allow_html=True)


# ---------------- DASHBOARD ----------------

if page == "Dashboard":

    hero(
        "Customer Intelligence Dashboard",
        "Understand customer behavior. Predict churn. Make smarter decisions."
    )

    if df is not None:

        total_customers = len(df)
        churned = (df["Churn"] == "Yes").sum()
        retained = (df["Churn"] == "No").sum()
        churn_rate = churned / total_customers * 100

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            metric_card("TOTAL CUSTOMERS", f"{total_customers:,}", "Dataset records")

        with c2:
            metric_card("CHURNED CUSTOMERS", f"{churned:,}", "Customers lost")

        with c3:
            metric_card("RETAINED CUSTOMERS", f"{retained:,}", "Customers retained")

        with c4:
            metric_card("CHURN RATE", f"{churn_rate:.2f}%", "Overall churn percentage")

        st.markdown('<div class="section-title">Customer Overview</div>',
                    unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            churn_data = df["Churn"].value_counts().reset_index()
            churn_data.columns = ["Status", "Customers"]

            fig = px.pie(
                churn_data,
                names="Status",
                values="Customers",
                hole=0.65,
                title="Customer Churn Distribution",
                color="Status",
                color_discrete_map={
                    "Yes": "#F87171",
                    "No": "#818CF8"
                }
            )
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#151F32",
                plot_bgcolor="#151F32",
                font_color="#F8FAFC"
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            contract_data = (
                df.groupby("Contract")["Churn"]
                .apply(lambda x: (x == "Yes").mean() * 100)
                .reset_index(name="Churn Rate")
            )

            fig = px.bar(
                contract_data,
                x="Contract",
                y="Churn Rate",
                color="Contract",
                title="Churn Rate by Contract",
                color_discrete_sequence=["#818CF8", "#A78BFA", "#38BDF8"]
            )
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#151F32",
                plot_bgcolor="#151F32",
                font_color="#F8FAFC",
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("Dataset not found. Check the dataset folder.")


# ---------------- PREDICTION ----------------

elif page == "Predict Churn":

    hero(
        "AI Churn Prediction",
        "Enter customer details to predict whether the customer may leave."
    )

    st.markdown("### 👤 Customer Information")

    with st.form("prediction_form"):

        st.markdown("#### Personal Details")

        c1, c2, c3 = st.columns(3)

        with c1:
            gender = st.selectbox("Gender", ["Female", "Male"])

        with c2:
            senior = st.selectbox("Senior Citizen", [0, 1])

        with c3:
            partner = st.selectbox("Partner", ["Yes", "No"])

        c1, c2 = st.columns(2)

        with c1:
            dependents = st.selectbox("Dependents", ["Yes", "No"])

        with c2:
            tenure = st.number_input(
                "Tenure (Months)", 0, 72, 12
            )

        st.divider()
        st.markdown("#### 📞 Services")

        c1, c2 = st.columns(2)

        with c1:
            phone = st.selectbox("Phone Service", ["Yes", "No"])
            multiple = st.selectbox(
                "Multiple Lines",
                ["No", "Yes", "No phone service"]
            )
            internet = st.selectbox(
                "Internet Service",
                ["DSL", "Fiber optic", "No"]
            )
            security = st.selectbox(
                "Online Security",
                ["Yes", "No", "No internet service"]
            )
            backup = st.selectbox(
                "Online Backup",
                ["Yes", "No", "No internet service"]
            )
            protection = st.selectbox(
                "Device Protection",
                ["Yes", "No", "No internet service"]
            )

        with c2:
            support = st.selectbox(
                "Tech Support",
                ["Yes", "No", "No internet service"]
            )
            tv = st.selectbox(
                "Streaming TV",
                ["Yes", "No", "No internet service"]
            )
            movies = st.selectbox(
                "Streaming Movies",
                ["Yes", "No", "No internet service"]
            )
            contract = st.selectbox(
                "Contract",
                ["Month-to-month", "One year", "Two year"]
            )
            paperless = st.selectbox(
                "Paperless Billing", ["Yes", "No"]
            )
            payment = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)"
                ]
            )

        st.divider()
        st.markdown("#### Billing Details")

        c1, c2 = st.columns(2)

        with c1:
            monthly = st.number_input(
                "Monthly Charges",
                min_value=18.0,
                max_value=120.0,
                value=70.0
            )

        with c2:
            total = st.number_input(
                "Total Charges",
                min_value=0.0,
                max_value=9000.0,
                value=840.0
            )

        submit = st.form_submit_button(
            "⚡ Predict Customer Churn",
            use_container_width=True
        )

    if submit:

        input_data = pd.DataFrame([{
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple,
            "InternetService": internet,
            "OnlineSecurity": security,
            "OnlineBackup": backup,
            "DeviceProtection": protection,
            "TechSupport": support,
            "StreamingTV": tv,
            "StreamingMovies": movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total
        }])

        try:
            processed_data = preprocessor.transform(input_data)
            prediction = model.predict(processed_data)[0]

            probability = None

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(processed_data)[0]
                classes = list(model.classes_)
                probability = probabilities[classes.index("Yes")] * 100

            st.divider()
            st.markdown("### Prediction Result")

            if prediction == "Yes":

                st.error("⚠️ Customer is likely to Churn!")

            else:

                st.success("✅ Customer is likely to Stay!")

            if probability is not None:
                st.markdown("#### Churn Probability")
                st.progress(int(probability))
                st.metric("Probability of Churn", f"{probability:.2f}%")

                if probability >= 50:
                    st.warning("This customer has a higher predicted churn risk.")
                else:
                    st.info("This customer has a lower predicted churn risk.")

        except Exception as e:
            st.error(f"Prediction error: {e}")


# ---------------- ANALYTICS ----------------

elif page == "Analytics":

    hero(
        "Customer Analytics",
        "Explore patterns and discover what drives customer churn."
    )

    if df is not None:

        c1, c2 = st.columns(2)

        with c1:
            fig = px.histogram(
                df,
                x="tenure",
                color="Churn",
                nbins=25,
                title="Churn by Customer Tenure",
                color_discrete_sequence=["#818CF8", "#F87171"]
            )
            fig.update_layout(template="plotly_dark",
                              paper_bgcolor="#151F32")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig = px.box(
                df,
                x="Churn",
                y="MonthlyCharges",
                color="Churn",
                title="Monthly Charges vs Churn",
                color_discrete_sequence=["#818CF8", "#F87171"]
            )
            fig.update_layout(template="plotly_dark",
                              paper_bgcolor="#151F32")
            st.plotly_chart(fig, use_container_width=True)

        contract_data = (
            df.groupby("Contract")["Churn"]
            .apply(lambda x: (x == "Yes").mean() * 100)
            .reset_index(name="Churn Rate")
        )

        fig = px.bar(
            contract_data,
            x="Contract",
            y="Churn Rate",
            title="Churn Percentage by Contract Type",
            color="Churn Rate",
            color_continuous_scale="Purples"
        )
        fig.update_layout(template="plotly_dark",
                          paper_bgcolor="#151F32")
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("Dataset not found.")


# ---------------- MODEL PERFORMANCE ----------------

elif page == "Model Performance":

    hero(
        "Model Performance",
        "Evaluation results of the trained machine learning models."
    )

    st.markdown("### Logistic Regression")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card("ACCURACY", "80.55%", "Overall correctness")

    with c2:
        metric_card("PRECISION", "65.72%", "Positive prediction quality")

    with c3:
        metric_card("RECALL", "55.88%", "Churn detection rate")

    with c4:
        metric_card("F1 SCORE", "60.40%", "Precision-recall balance")

    st.markdown("### Model Comparison")

    comparison = pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "Decision Tree",
            "Random Forest"
        ],
        "F1 Score": [
            0.6040,
            0.4776,
            0.5323
        ]
    })

    fig = px.bar(
        comparison,
        x="Model",
        y="F1 Score",
        color="Model",
        title="F1 Score Comparison",
        text_auto=".3f",
        color_discrete_sequence=[
            "#818CF8", "#A78BFA", "#38BDF8"
        ]
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#151F32",
        plot_bgcolor="#151F32",
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Logistic Regression achieved the highest F1 score "
        "among the three evaluated models."
    )


# ---------------- ABOUT PROJECT ----------------

elif page == "About Project":

    hero(
        "About ChurnIQ",
        "An AI-powered customer churn prediction and analytics platform."
    )

    st.markdown("### Project Objective")

    st.write(
        "The objective is to predict whether a telecom customer "
        "is likely to leave the service using machine learning."
    )

    st.markdown("### Dataset Information")

    st.write("""
    - Dataset: IBM Telco Customer Churn
    - Total Records: 7,043
    - Total Columns: 21
    - Target Variable: Churn
    - ML Task: Binary Classification
    """)

    st.markdown("### Machine Learning Models")

    st.write("""
    1. Logistic Regression
    2. Decision Tree
    3. Random Forest
    """)

    st.markdown("### Technology Stack")

    st.write("""
    - Python
    - Pandas and NumPy
    - Matplotlib and Seaborn
    - Scikit-learn
    - Streamlit
    - Plotly
    - Joblib
    """)

    st.markdown("### Project Workflow")

    st.write("""
    Data Collection → Data Cleaning → EDA → Preprocessing →
    Model Training → Model Evaluation → Churn Prediction
    """)

# ---------------- FOOTER ----------------

st.divider()

st.markdown(
    "<center style='color:#64748B'>"
    "ChurnIQ | AI-Powered Customer Intelligence | Built with Python"
    "</center>",
    unsafe_allow_html=True
)
