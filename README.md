# 🏠 House Price Prediction Web App

A Machine Learning + Flask based web application that predicts house prices based on various features like area, number of bedrooms, bathrooms, stories, furnishing status, parking, and more.

---

## 📂 Project Structure

House Price Prediction/
│
├── Dataset/
│ └── Housing.csv # Dataset used for training
│
├── static/ # CSS styling files
│ ├── result.css
│ └── style.css
│
├── templates/ # HTML templates for UI
│ ├── index.html # Input form
│ └── result.html # Prediction result page
│
├── app.py # Flask app for serving model
├── model_building.py # Script to train & save model
├── house_price_model.joblib # Trained ML model
├── scaler.joblib # Feature scaler
├── encoder.joblib # Label/OneHot encoder
├── requirements.txt # Project dependencies
├── runtime.txt # Python runtime version
└── README.md # Project documentation


---

## 🚀 Features
- Simple and interactive **web interface** for predictions  
- Trained ML model using **Scikit-learn**  
- **Flask** backend to handle requests  
- Encoders & scalers stored for preprocessing  
- Clean UI with **HTML + CSS**  

---

## ⚙️ Tech Stack
- **Python 3.11.5**  
- **Flask 3.0.3**  
- **Gunicorn 23.0.0**  
- **Pandas 2.2.2**  
- **NumPy 2.1.0**  
- **Scikit-learn 1.6.1**  
- **Joblib 1.4.2**

---

## 🛠️ Installation & Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/house-price-prediction.git
   cd house-price-prediction

2. Create a virtual environment & activate it:
python -m venv venv
venv\Scripts\activate   # For Windows
source venv/bin/activate # For Linux/Mac

3. Install dependencies:
pip install -r requirements.txt

4. Run the Flask app:
python app.py

5. Open your browser at:
http://127.0.0.1:5000/


📊 Dataset
The dataset used is Housing.csv, containing features such as:

area
bedrooms
bathrooms
stories
mainroad (yes/no)
guestroom (yes/no)
basement (yes/no)
hotwaterheating (yes/no)
airconditioning (yes/no)
parking
prefarea (yes/no)
furnishingstatus (furnished, semi-furnished, unfurnished)
price (target variable)


🎯 How it Works

User enters house details on the index.html form
Input data is preprocessed (scaled + encoded)
ML model (RandomForestRegressor or similar) predicts the house price
Result is displayed on the result.html page

📸 Screenshots
<img width="1308" height="640" alt="image" src="https://github.com/user-attachments/assets/989f8966-c113-4bd6-bd6c-b41ef05fa885" />

