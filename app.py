from flask import Flask, request, render_template
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

# Route for home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for prediction form and result
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        try:
            # Collect form data
            data = CustomData(
                gender=request.form['gender'],
                SeniorCitizen=int(request.form['SeniorCitizen']),
                Partner=request.form['Partner'],
                Dependents=request.form['Dependents'],
                tenure=int(request.form['tenure']),
                PhoneService=request.form['PhoneService'],
                MultipleLines=request.form['MultipleLines'],
                InternetService=request.form['InternetService'],
                OnlineSecurity=request.form['OnlineSecurity'],
                OnlineBackup=request.form['OnlineBackup'],
                DeviceProtection=request.form['DeviceProtection'],
                TechSupport=request.form['TechSupport'],
                StreamingTV=request.form['StreamingTV'],
                StreamingMovies=request.form['StreamingMovies'],
                Contract=request.form['Contract'],
                PaperlessBilling=request.form['PaperlessBilling'],
                PaymentMethod=request.form['PaymentMethod'],
                MonthlyCharges=float(request.form['MonthlyCharges']),
                TotalCharges=float(request.form['TotalCharges'])
            )

            # Convert to DataFrame
            input_df = data.get_data_as_dataframe()

            # Load prediction pipeline
            pipeline = PredictPipeline()
            prediction_result = pipeline.predict(input_df)

            # Format prediction
            prediction_text = "✅ Customer is likely to churn." if prediction_result[0] == 1 else "🟢 Customer is not likely to churn."

            return render_template('home.html', prediction_text=prediction_text)

        except Exception as e:
            error_message = f"⚠️ Error occurred: {str(e)}"
            return render_template('home.html', prediction_text=error_message)

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000,debug=True)
