# Imports
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def main():
    # Load data
    housing = fetch_california_housing(as_frame=True)
    X = housing.data
    y = housing.target

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Model
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("mlp", MLPRegressor(
                hidden_layer_sizes=(64, 32),
                early_stopping=True,
                random_state=42,
                max_iter=500
            ))
        ]
    )

    model.fit(X_train, y_train)

    print("Model trained successfully")


if _name_ == "_main_":
    main()
