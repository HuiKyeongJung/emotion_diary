import joblib

# 저장된 모델 불러오기
model = joblib.load("emotion_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# 테스트 문장
text = ["오늘 너무 행복하고 기분이 좋아"]

# 문장을 숫자로 변환
text_vectorized = vectorizer.transform(text)

# 감정 예측
prediction = model.predict(text_vectorized)

print("입력 문장:", text[0])
print("예측 감정:", prediction[0])