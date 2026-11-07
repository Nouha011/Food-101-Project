import pandas as pd

def get_calories(food_name, csv_path):
    df = pd.read_csv(csv_path)
    # Find the row where the class name matches the AI prediction
    row = df[df['class'].str.lower() == food_name.lower()]
    if not row.empty:
        return row.iloc[0]['kcal_per_100']
    return "Unknown"