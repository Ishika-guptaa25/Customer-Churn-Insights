# Customer Churn Insights
### AI-Powered Customer Retention System

An end-to-end Machine Learning project that predicts whether a telecom customer is likely to leave the company (churn) using customer demographics, service usage, contract details, and billing information.

## Project Overview

Customer Churn Insights uses Machine Learning classification algorithms to predict customer churn.

It includes an interactive Streamlit web application where users can enter customer details and get real-time churn predictions.


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


### Model Performance: Before vs After

| Metric | Before Feature Engineering | After Feature Engineering  |
|---|---:|---:|
| Accuracy | 80.62% | 91.55% |
| Precision | 65.83% | 85.72% |
| Recall | 56.15% | 89.88% |
| F1 Score | 60.61% | 87.40% |

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

## Future Improvements

- Improve prediction accuracy through advanced model optimization.
- Experiment with additional ensemble learning algorithms.
- Add customer retention recommendations.
- Deploy the application online.
- Improve the dashboard with additional business insights.

## Author

**Ishika Gupta**

*Turning Customer Data into Actionable Retention Strategies.*
