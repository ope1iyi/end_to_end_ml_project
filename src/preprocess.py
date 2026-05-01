import pandas as pd
import numpy as np
from pathlib import Path

RAW_DATA_PATH = Path("Dataset/raw/Life Expectancy Data.csv")
PROCESSED_DATA_PATH = Path("Dataset/processed/life_expectancy_cleaned.csv")

SELECTED_FEATURES = ['schooling',
                'income_composition_of_resources',
                'log_gdp',
                'bmi',
                'diphtheria',
                'alcohol',
                'log_thinness_5-9_years',
                'adult_mortality',
                'log_hiv/aids']

TARGET = 'life_expectancy'


def load_data(path)-> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    return df

def handle_missing_values(df: pd.DataFrame):
    drop_rows = ['life_expectancy', 'adult_mortality', 'bmi', 'diphtheria',
           'polio', 'thinness__1-19_years', 'thinness_5-9_years']

    df = df.dropna(subset=drop_rows)
    
    df['income_composition_of_resources'] = df.groupby('status')['income_composition_of_resources'].transform(lambda x: x.fillna(x.mean()))
    df['population'] = df.groupby('status')['population'].transform(lambda x: x.fillna(x.mean()))
    df['gdp'] = df.groupby('status')['gdp'].transform(lambda x: x.fillna(x.mean()))
    df['schooling'] = df.groupby('status')['schooling'].transform(lambda x: x.fillna(x.mean()))
    df['alcohol'] = df.groupby('country')['alcohol'].transform(lambda x: x.fillna(x.mean()))
    df['total_expenditure'] = df.groupby('status')['total_expenditure'].transform(lambda x: x.fillna(x.mean()))
    df['hepatitis_b'] = df.groupby('status')['hepatitis_b'].transform(lambda x: x.fillna(x.mean()))
    return df

def feature_engineering(df: pd.DataFrame)-> pd.DataFrame:
    df['log_gdp'] = np.log1p(df['gdp'])
    df['log_hiv/aids'] = np.log1p(df['hiv/aids'])
    df['log_thinness_5-9_years'] = np.log1p(df['thinness_5-9_years'])
    return df

def select_features(df: pd.DataFrame):
    return df[SELECTED_FEATURES + [TARGET]]


def save_data(df: pd.DataFrame, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)




def main():
    df = load_data(RAW_DATA_PATH)
    df = handle_missing_values(df)
    df = feature_engineering(df)
    df = select_features(df)
    save_data(df, PROCESSED_DATA_PATH)

    print(f"Preprocessing complete. Saved to {PROCESSED_DATA_PATH}")
    print(f"Final shape: {df.shape}")


if __name__ == "__main__":
    main()