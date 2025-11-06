import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def preprocess_data(csv_path):
    df = pd.read_csv(csv_path)

    # Clean and standardize
    df['Status'] = df['Status'].str.lower().str.strip()
    df['ErrorMessage'] = df['ErrorMessage'].fillna('None')
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    # Feature: Failure Rate
    status_counts = df.groupby(['TestName', 'Status']).size().unstack(fill_value=0)
    status_counts['TotalRuns'] = status_counts.sum(axis=1)
    status_counts['FailureRate'] = status_counts.get('fail', 0) / status_counts['TotalRuns']
    df = df.merge(status_counts[['FailureRate']], on='TestName', how='left')

    # Feature: Duration Variance
    duration_stats = df.groupby('TestName')['Duration_ms'].agg(['mean', 'std']).rename(columns={'std': 'DurationVariance'})
    df = df.merge(duration_stats[['DurationVariance']], on='TestName', how='left')

    # Feature: Time-of-Day
    df['Hour'] = df['Timestamp'].dt.hour
    df['TimeOfDay'] = pd.cut(df['Hour'], bins=[0,6,12,18,24], labels=['Night','Morning','Afternoon','Evening'], right=False)
    print(df[['Hour', 'TimeOfDay']].head(10))

    # Feature: Environment Volatility
    env_combo = df.groupby('TestName')[['OS', 'Browser']].nunique()
    env_combo['EnvVolatility'] = env_combo['OS'] + env_combo['Browser']
    df = df.merge(env_combo[['EnvVolatility']], on='TestName', how='left')

    # Heuristic Label
    df['FlakyHeuristic'] = ((df['FailureRate'] > 0.3) & (df['Status'] == 'pass')).astype(int)

    # Count values
    flaky_counts = df['FlakyHeuristic'].value_counts().sort_index()

    # Plot
    # sns.barplot(x=flaky_counts.index, y=flaky_counts.values, palette='Set2')
    # plt.xticks([0, 1], ['Not Flaky (0)', 'Flaky (1)'])
    # plt.title('Distribution of FlakyHeuristic')
    # plt.xlabel('FlakyHeuristic Label')
    # plt.ylabel('Number of Test Runs')
    # plt.show()

    # plt.figure(figsize=(6, 6))
    # plt.pie(
    #     flaky_counts.values,
    #     labels=['Not Flaky', 'Flaky'],
    #     autopct='%1.1f%%',
    #     colors=['#66c2a5', '#fc8d62'],
    #     startangle=140
    # )
    # plt.title('FlakyHeuristic Distribution')
    # plt.axis('equal')
    # plt.show()

    return df




if __name__ == "__main__":
    # Provide the path to your CSV file
    csv_path = "../data/sample_flaky_test_dataset_5000.csv"

    # Run the preprocessing
    processed_df = preprocess_data(csv_path)

    # Preview the result
    print(processed_df.head())         # Show first few rows
    print(processed_df.columns)        # Show all column names
    print(processed_df['FlakyHeuristic'].value_counts())  # Check heuristic distribution
    print(processed_df['FlakyHeuristic'])