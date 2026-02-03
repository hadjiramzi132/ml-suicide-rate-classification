import argparse
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def load_data(path: str, target: str):
    df = pd.read_csv(path)

    df = df.dropna().copy()

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found. Available: {list(df.columns)}")

    y = df[target]
    X = df.drop(columns=[target])

    X = pd.get_dummies(X, drop_first=True)

    return X, y


def build_model(name: str, k: int, seed: int):
    name = name.lower()

    if name == "knn":
        return KNeighborsClassifier(n_neighbors=k)

    if name == "logreg":
        return LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed)

    if name == "rf":
        return RandomForestClassifier(
            n_estimators=300,
            random_state=seed,
            class_weight="balanced",
            n_jobs=-1
        )

    raise ValueError("Unknown model. Choose from: knn, logreg, rf")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/suicide_rate.csv")
    parser.add_argument("--target", type=str, default="label")
    parser.add_argument("--model", type=str, default="knn", choices=["knn", "logreg", "rf"])
    parser.add_argument("--k", type=int, default=7)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    X, y = load_data(args.data, args.target)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y if y.nunique() > 1 else None
    )

    scaler = StandardScaler(with_mean=False)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = build_model(args.model, args.k, args.seed)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    print("\n=== RESULTS ===")
    print("Model:", args.model)
    print("Accuracy:", round(accuracy_score(y_test, pred), 4))
    print("F1-score:", round(f1_score(y_test, pred, average="weighted"), 4))
    print("\nConfusion matrix:\n", confusion_matrix(y_test, pred))
    print("\nClassification report:\n", classification_report(y_test, pred))


if __name__ == "__main__":
    main()
