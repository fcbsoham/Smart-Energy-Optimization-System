import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

def train_model():

    df = pd.read_csv("smart_energy_subset.csv")

    le = LabelEncoder()
    df["primary_use"] = le.fit_transform(df["primary_use"])

    df = df.dropna()

    X = df.drop("meter_reading", axis=1)
    y = df["meter_reading"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    return model