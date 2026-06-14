import json

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC


DATA_PATH = "data/emotion_dataset_final.csv"
MODEL_PATH = "emotion_model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"
METRICS_PATH = "model_metrics.json"

LABEL_MAP = {
    "joy": "기쁨",
    "sadness": "슬픔",
    "anger": "분노",
    "anxiety": "불안",
    "depression": "우울",
}


def read_dataset(path):
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp949")


def build_vectorizer():
    return TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )


def main():
    df = read_dataset(DATA_PATH)
    df = df[df["emotion"].isin(LABEL_MAP)].copy()
    df["emotion"] = df["emotion"].map(LABEL_MAP)

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["emotion"],
        test_size=0.2,
        random_state=42,
        stratify=df["emotion"],
    )

    vectorizer = build_vectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    candidates = [
        ("linear_svc_c0.5", LinearSVC(C=0.5, class_weight="balanced")),
        ("linear_svc_c1.0", LinearSVC(C=1.0, class_weight="balanced")),
        ("linear_svc_c2.0", LinearSVC(C=2.0, class_weight="balanced")),
    ]

    best = None
    results = []

    for name, model in candidates:
        model.fit(X_train_vec, y_train)
        pred = model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, pred)
        macro_f1 = f1_score(y_test, pred, average="macro")
        results.append({"name": name, "accuracy": accuracy, "macro_f1": macro_f1})
        print(f"{name} 정확도: {accuracy:.4f}, macro F1: {macro_f1:.4f}")

        if best is None or macro_f1 > best["macro_f1"]:
            best = {
                "name": name,
                "model": model,
                "pred": pred,
                "accuracy": accuracy,
                "macro_f1": macro_f1,
            }

    print("\n최종 선택:", best["name"])
    print("정확도:", best["accuracy"])
    print("macro F1:", best["macro_f1"])
    print(classification_report(y_test, best["pred"]))

    joblib.dump(best["model"], MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "selected_model": best["name"],
                "accuracy": best["accuracy"],
                "macro_f1": best["macro_f1"],
                "candidates": results,
                "labels": sorted(df["emotion"].unique().tolist()),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("모델 저장 완료")


if __name__ == "__main__":
    main()
