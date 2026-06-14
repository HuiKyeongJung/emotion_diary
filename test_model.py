from emotion_predictor import load_artifacts, predict_emotion


model, vectorizer = load_artifacts()

print("감정 테스트 시작")
print("종료하려면 exit 입력")
print("-" * 30)

while True:
    text = input("입력 문장: ")

    if text.lower() == "exit":
        print("테스트 종료")
        break

    result = predict_emotion(text, model, vectorizer)

    print("입력 문장:", text)
    print("예측 감정:", result["emotion"])
    print("신뢰도:", result["confidence"])
    print("판단 방식:", result["source"])
    print("-" * 30)
