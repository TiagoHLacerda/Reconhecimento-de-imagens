import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

# Carregar modelo treinado
model = load_model("cifar10_model.h5")

# Classes do CIFAR-10
class_names = [
    "avião", "automóvel", "pássaro", "gato", "veado",
    "cachorro", "sapo", "cavalo", "navio", "caminhão"
]

# Média e desvio padrão do CIFAR-10 (calculados a partir dos dados de treino)
mean = 125.3
std = 63.0

def preprocess_user_image(image_file, mean, std):
    # Abrir imagem enviada e converter para RGB (3 canais)
    original_img = Image.open(image_file).convert('RGB')

    # Redimensionar para 32x32 pixels, como no CIFAR-10
    resized_img = original_img.resize((32, 32), Image.Resampling.LANCZOS)

    # Converter para array e normalizar com a mesma média e std do treino
    img_array = np.array(resized_img).astype('float32')
    if img_array.shape != (32, 32, 3):
        raise ValueError(f"Shape inesperado: {img_array.shape}. Esperado (32, 32, 3)")

    img_array = (img_array - mean) / std
    img_array = np.expand_dims(img_array, axis=0)  # (1, 32, 32, 3)

    return img_array, original_img

def classify_user_image(model, img_array):
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    return predicted_class_index, predictions[0]

# --- Interface Streamlit ---
st.title("🔍 Reconhecimento de Imagens - CIFAR-10")

st.markdown("Nos treinamentos com imagens do repositorio cifar 10 atingiu 87% de accuraci nas imagens")

st.markdown("### Este modelo reconhece as seguintes categorias:")

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

# Upload da imagem
uploaded_file = st.file_uploader("📷 Envie uma imagem (tirada ou da web):", type=["jpg", "jpeg", "png"])

# Processamento e exibição
if uploaded_file is not None:
    img_array, original_img = preprocess_user_image(uploaded_file, mean, std)
    class_index, probs = classify_user_image(model, img_array)
    class_name = class_names[class_index]

    # Mostrar imagem original
    st.image(original_img, caption="📸 Imagem original enviada", use_column_width=True)

    # Mostrar classe prevista
    st.success(f"🧠 Classe prevista: **{class_name}**")

    # Mostrar probabilidades para cada classe
    st.subheader("📊 Confiança do modelo para cada classe:")
    for i, prob in enumerate(probs):
        st.write(f"{class_names[i]}: {prob:.4f}")
        st.progress(float(prob))
