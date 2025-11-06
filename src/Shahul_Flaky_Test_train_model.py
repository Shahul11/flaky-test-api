# src/Shahul_Flaky_Test_train_model.py

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib
from Shahul_Flaky_Test_preprocess import preprocess_data

def train_model(csv_path, model_path):
    df = preprocess_data(csv_path)

    # Encode categorical features
    le_test = LabelEncoder()
    df['TestNameEncoded'] = le_test.fit_transform(df['TestName'])

    le_time = LabelEncoder()
    df['TimeOfDayEncoded'] = le_time.fit_transform(df['TimeOfDay'].astype(str))

    # Feature selection
    features = ['TestNameEncoded', 'FailureRate', 'DurationVariance', 'EnvVolatility', 'TimeOfDayEncoded']
    X = df[features]
    y = df['FlakyLabel']  # or use 'FlakyHeuristic'

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

    # Train model
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # Save model
    joblib.dump(rf, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
        csv_path = "../data/sample_flaky_test_dataset_5000.csv"  # Update path if needed
        model_path = "../models/flaky_model.pkl"

        train_model(csv_path, model_path)