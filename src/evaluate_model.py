# src/evaluate_model.py

from sklearn.metrics import classification_report, confusion_matrix
import shap
import joblib
from sklearn.preprocessing import LabelEncoder

from preprocess import preprocess_data

def evaluate_model(csv_path, model_path='../models/flaky_model.pkl'):
    df = preprocess_data(csv_path)

    # Encode
    df['TestNameEncoded'] = LabelEncoder().fit_transform(df['TestName'])
    df['TimeOfDayEncoded'] = LabelEncoder().fit_transform(df['TimeOfDay'].astype(str))

    features = ['TestNameEncoded', 'FailureRate', 'DurationVariance', 'EnvVolatility', 'TimeOfDayEncoded']
    X = df[features]
    y = df['FlakyLabel']

    # Load model
    model = joblib.load(model_path)
    y_pred = model.predict(X)

    print(classification_report(y, y_pred))
    print(confusion_matrix(y, y_pred))

    # # SHAP
    # explainer = shap.TreeExplainer(model)
    # shap_values = explainer.shap_values(X)
    # shap.summary_plot(shap_values[1], X)

if __name__ == "__main__":
        evaluate_model("../data/sample_flaky_test_dataset_5000.csv")