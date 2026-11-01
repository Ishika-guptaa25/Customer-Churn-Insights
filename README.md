# Customer Churn Insights
### AI-Powered Customer Retention System

An end-to-end Machine Learning project that predicts whether a telecom customer is likely to leave the company (churn) using customer demographics, service usage, contract details, and billing information.

The project helps businesses identify customers at risk of leaving and make better customer retention decisions.

## Project Overview

Customer Churn Insights uses Machine Learning classification algorithms to predict customer churn.

It includes an interactive Streamlit web application where users can enter customer details and get real-time churn predictions.

## Features

- Customer Churn Prediction
- Interactive Streamlit Dashboard
- Customer Information Analysis
- Exploratory Data Analysis (EDA)
- Data Preprocessing and Feature Engineering
- Comparison of Three Machine Learning Models
- Model Performance Evaluation
- Confusion Matrix and Classification Report
- Real-time Customer Prediction

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming Language |
| Pandas | Data Manipulation |
| NumPy | Numerical Operations |
| Matplotlib | Data Visualization |
| Seaborn | Statistical Visualization |
| Scikit-learn | Machine Learning |
| Streamlit | Web Application |
| Joblib | Model Saving and Loading |

## Dataset

**IBM Telco Customer Churn Dataset**

- Total Records: 7,043
- Total Features: 21
- Target Variable: `Churn`
- Prediction Classes: Yes / No

The dataset contains customer demographics, subscription information, services, contract types, and billing details.

## Machine Learning Models

Three classification algorithms were implemented and evaluated:

1. Logistic Regression
2. Decision Tree
3. Random Forest

### Model Performance

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 80.55% | 65.72% | 55.88% | **60.40%** |
| Decision Tree | 71.89% | 47.14% | 48.40% | 47.76% |
| Random Forest | 77.93% | 60.82% | 47.33% | 53.23% |

**Best Performing Model: Logistic Regression**

Logistic Regression achieved the highest F1 score of 60.40% among the three evaluated models.

## Project Structure

```text
Customer-Churn-Insights/
│
├── dataset/
│   └── telco_customer_churn.csv
│
├── app.py
├── check_data.py
├── eda.py
├── Telecom_Customer_Churn.ipynb
│
├── churn_model.pkl
├── preprocessor.pkl
│
├── requirements.txt
└── README.md
```

## Installation and Setup

**1. Clone the repository**

```bash
git clone https://github.com/Ishika-guptaa25/Customer-Churn-Insights.git
```

**2. Navigate to the project directory**

```bash
cd Customer-Churn-Insights
```

**3. Create a virtual environment**

```bash
python -m venv .venv
```

**4. Activate the environment**

Mac/Linux:

```bash
source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

**5. Install dependencies**

```bash
pip install -r requirements.txt
```

**6. Run the Streamlit application**

```bash
streamlit run app.py
```

The application will open in your browser.

## Model Evaluation

The Logistic Regression model was evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Classification Report

These metrics help evaluate how effectively the model identifies customers who are likely to leave.

## Future Improvements

- Improve prediction accuracy through hyperparameter tuning.
- Implement advanced ensemble models.
- Add customer retention recommendations.
- Deploy the application online.
- Improve the dashboard with additional business insights.

## Author

**Ishika Gupta**

MCA | Data Science

Developed as an end-to-end Machine Learning project using Python and Streamlit.

---

*Turning Customer Data into Actionable Retention Strategies.*