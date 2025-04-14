import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image, ImageOps
import io

# 페이지 세팅
st.set_page_config(page_title="손글씨 숫자 인식기", layout="centered")

st.title("✍️ 손글씨 숫자 인식기")
st.write("손글씨 숫자를 업로드하면 AI가 인식해줘요!")

# 모델 로딩
@st.cache_resource
def load_digit_model():
    model = load_model('mnist_model.h5')
    return model

model = load_digit_model()

# 이미지 업로드
uploaded_file = st.file_uploader("🖼️ 숫자 손글씨 이미지를 업로드해주세요", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')  # 흑백으로 변환
    st.image(image, caption='업로드된 이미지', use_column_width=True)

    # 전처리: 28x28 사이즈, 흰배경/검은글씨로 정규화
    image = ImageOps.invert(image)  # 검은 배경 → 흰 배경
    image = image.resize((28, 28))
    image_array = np.array(image).astype("float32") / 255.0
    image_array = image_array.reshape(1, 784)  # (1, 28*28)

    # 예측
    prediction = model.predict(image_array)
    predicted_label = np.argmax(prediction)

    st.markdown(f"### 🧠 AI가 예측한 숫자: **{predicted_label}**")
