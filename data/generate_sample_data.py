import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# Get the absolute path to the workspace root directory
current_file = os.path.abspath(__file__)
workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
DATA_DIR = os.path.join(workspace_root, "pipelines", "data")

print(f"Workspace root: {workspace_root}")
print(f"Data directory: {DATA_DIR}")

# Create data directory if it doesn't exist
os.makedirs(DATA_DIR, exist_ok=True)

# 시계열 데이터 생성
def generate_time_series():
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)
    
    df = pd.DataFrame({
        'date': dates,
        'sales': np.random.normal(1000, 100, len(dates)),
        'visitors': np.random.normal(500, 50, len(dates)),
        'temperature': np.random.normal(20, 5, len(dates)),
        'category': np.random.choice(['A', 'B', 'C'], len(dates))
    })
    
    # 계절성 추가
    df['sales'] += np.sin(np.arange(len(dates)) * 2 * np.pi / 365) * 200
    df['visitors'] += np.sin(np.arange(len(dates)) * 2 * np.pi / 365) * 100
    
    output_path = os.path.join(DATA_DIR, 'sales_data.csv')
    df.to_csv(output_path, index=False)
    print(f"Generated sales_data.csv at {output_path}")
    print(f"File size: {os.path.getsize(output_path)} bytes")

# 다변량 데이터 생성
def generate_multivariate_data():
    np.random.seed(42)
    n_samples = 1000
    
    df = pd.DataFrame({
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.normal(0, 1, n_samples),
        'feature3': np.random.normal(0, 1, n_samples),
        'feature4': np.random.normal(0, 1, n_samples),
        'target': np.random.normal(0, 1, n_samples)
    })
    
    # 상관관계 추가
    df['feature2'] = df['feature1'] * 0.5 + df['feature2'] * 0.5
    df['target'] = df['feature1'] * 0.3 + df['feature2'] * 0.3 + df['feature3'] * 0.4 + np.random.normal(0, 0.1, n_samples)
    
    output_path = os.path.join(DATA_DIR, 'analysis_data.csv')
    df.to_csv(output_path, index=False)
    print(f"Generated analysis_data.csv at {output_path}")
    print(f"File size: {os.path.getsize(output_path)} bytes")

if __name__ == "__main__":
    print(f"Current working directory: {os.getcwd()}")
    print(f"Script location: {os.path.abspath(__file__)}")
    generate_time_series()
    generate_multivariate_data()