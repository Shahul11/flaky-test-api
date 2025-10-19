


```markdown
# 🧪 Flaky Test Prediction System

This project predicts whether a software test run is **flaky** or **stable** using machine learning. It includes data preprocessing, model training, evaluation, and a Flask API for real-time predictions.



## 📁 Project Structure


Shahul_Hameed_Project/
├── data/
│   └── sample_flaky_test_dataset_5000.csv
├── models/
│   └── flaky_model.pkl
├── src/
│   ├── preprocess.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── predict_api.py

```
---

## 🚀 Features

- Preprocesses flaky test data with feature engineering
- Trains a Random Forest classifier
- Evaluates model performance with metrics and SHAP plots
- Serves predictions via a Flask API (`/predict` endpoint)

---

## 🔧 How to Run Locally

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

```bash
python src/preprocess.py
python src/train_model.py
```

### 3. Evaluate the model

```bash
python src/evaluate_model.py
```

### 4. Run the API

```bash
python src/predict_api.py
```

---

## 📬 API Usage

### Endpoint: `/predict`  
**Method:** `POST`  
**Content-Type:** `application/json`

### Sample Request:

```json
{
  "TestName": "test_login",
  "FailureRate": 0.8,
  "DurationVariance": 300,
  "EnvVolatility": 2.5,
  "TimeOfDay": "Evening"
}
```

### Sample Response:

```json
{
  "input": {...},
  "prediction": "Flaky"
}
```

---

## 📈 Model Performance

- Accuracy: _XX%_
- Precision: _XX%_
- Recall: _XX%_
- SHAP plots included in `evaluate_model.py`

---

## 🛠️ Future Improvements

- Replace crude encoders with saved `LabelEncoder` objects
- Add batch prediction via CSV upload
- Deploy to Hugging Face Spaces or Render
- Add Streamlit dashboard for interactive use

---

## 👨‍💻 Author

**Shahul Hameed**  
Focused on real-world ML projects, clean code, and collaborative learning.

---

## 📜 License


