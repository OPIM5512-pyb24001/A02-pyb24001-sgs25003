from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split


def main():
    # Load California Housing dataset
    housing = fetch_california_housing(as_frame=True)
    X = housing.data
    y = housing.target

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)


if __name__ == "__main__":
    main()
