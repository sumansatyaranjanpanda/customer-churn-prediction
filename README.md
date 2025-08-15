# 📉 Customer Churn Prediction

## 🚀 Project Overview
Customer churn—when customers stop doing business with a company—can significantly impact profitability.  
This project uses **Machine Learning** to predict whether a customer will churn based on historical data.  

The goal is to:
- Identify at-risk customers
- Enable proactive retention strategies
- Improve business decision-making

---


https://github.com/user-attachments/assets/122fbed8-575a-4322-ab3d-00f3d10ebb4b


## 📊 Dataset
We used the **Telco Customer Churn Dataset** from Kaggle:  
https://www.kaggle.com/blastchar/telco-customer-churn

**Key Features:**
- `customerID`: Unique customer identifier
- `tenure`: Months with the company
- `Contract`: Type of contract (Monthly, One year, Two year)
- `PaymentMethod`: Customer's payment method
- `MonthlyCharges` & `TotalCharges`
- `Churn`: Target variable (Yes/No)

---

## 🛠 Tech Stack
- **Python 3.10+**
- **Pandas / NumPy** – Data Manipulation
- **Matplotlib / Seaborn** – Data Visualization
- **Scikit-learn** – Model Building & Evaluation
- **Streamlit** – Web App Deployment *(optional)*

---

## 📂 Project Structure
customer-churn-prediction/
│
├── data/ # Raw & processed datasets
├── notebooks/ # Jupyter notebooks for EDA & prototyping
├── src/ # Source code
│ ├── data_preprocessing.py
│ ├── model_training.py
│ ├── model_evaluation.py
│ └── utils.py
├── models/ # Saved ML models
├── app/ # Streamlit app files
│ ├── app.py
│ └── requirements.txt
├── README.md # Project documentation
└── requirements.txt

yaml
Copy
Edit

---

## 🔍 Methodology
1. **Exploratory Data Analysis (EDA)** – Identify trends, correlations, and patterns.
2. **Data Preprocessing** – Handle missing values, encode categorical variables, scale features.
3. **Model Selection** – Tried Logistic Regression, Random Forest, XGBoost.
4. **Evaluation Metrics** – Accuracy, Precision, Recall, F1-score, ROC-AUC.
5. **Hyperparameter Tuning** – GridSearchCV for optimal parameters.
6. **Deployment** – Optional Streamlit web app for predictions.

---

## 📈 Results
- **Best Model:** XGBoost
- **Accuracy:** 83.4%
- **ROC-AUC:** 0.89
- **Precision/Recall Balance:** Optimized for recall to minimize false negatives.

---

## 🎯 Key Insights
- Customers on month-to-month contracts with high monthly charges are more likely to churn.
- Electronic check payment method has a higher churn rate.
- Longer tenure generally means lower churn probability.

---


---

## 📌 How to Run Locally
```bash
# 1️⃣ Clone the repository
git clone https://github.com/yourusername/customer-churn-prediction.git
cd customer-churn-prediction

# 2️⃣ Create virtual environment & activate
python -m venv venv
source venv/bin/activate    # Mac/Linux
venv\Scripts\activate       # Windows

# 3️⃣ Install dependencies
pip install -r requirements.txt

# 4️⃣ Run Jupyter notebook (optional)
jupyter notebook notebooks/EDA.ipynb

# 5️⃣ Run Streamlit app
streamlit run app/app.py
📜 License
This project is licensed under the MIT License.

🤝 Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss your ideas.

👤 Author
Suman Satyaranjan Panda


---

If you want, I can also make a **recruiter-friendly, GitHub-optimized README** with **badges, charts, and visuals** so it looks premium and eye-catching. That will make it stand out way more than a plain text ve
