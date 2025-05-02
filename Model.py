import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Carregar modelo treinado (deve estar na mesma pasta que este arquivo .py)
model = load_model("cifar10_model.h5")

# Classes do CIFAR-10
class_names = ["avião", "automóvel", "pássaro", "gato", "veado", "cachorro", "sapo", "cavalo", "navio", "caminhão"]

# Média e desvio padrão do CIFAR-10
mean = 125.3
std = 63.0

def preprocess_user_image(image_file, mean, std):
    original_img = Image.open(image_file).convert('RGB')  # Salva a original
    img = original_img.resize((32, 32))  # Redimensiona só para o modelo
    img_array = np.array(img)
    img_array = (img_array - mean) / std
    img_array = np.expand_dims(img_array, axis=0)
    return img_array, original_img


def classify_user_image(model, img_array):
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_names[predicted_class_index]
    return predicted_class_name

# Interface Streamlit
st.title("🔍 Reconhecimento de Imagens - CIFAR-10")
st.markdown("### Este modelo reconhece 10 categorias do CIFAR-10:")
st.markdown("""
- ✈️ Avião  
- 🚗 Automóvel  
- 🐦 Pássaro  
- 🐱 Gato  
- 🦌 Veado  
- 🐶 Cachorro  
- 🐸 Sapo  
- 🐴 Cavalo  
- 🚢 Navio  
- 🚛 Caminhão
""")

uploaded_file = st.file_uploader("Envie uma imagem:", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img_array, original_img = preprocess_user_image(uploaded_file, mean, std)
    predicted_class = classify_user_image(model, img_array)

    st.image(original_img, caption="Imagem enviada", use_container_width=True)

    st.success(f"🧠 Classe prevista: **{predicted_class}**")
