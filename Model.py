import streamlit as st
from PIL import Image
import numpy as np
import io
from tensorflow.keras.models import load_model

# Carregar modelo treinado
model = load_model("cifar10_model.h5")

# Classes do CIFAR-10
class_names = ["avião", "automóvel", "pássaro", "gato", "veado", "cachorro", "sapo", "cavalo", "navio", "caminhão"]

# Média e desvio padrão do dataset
mean = 125.3
std = 63.0

def preprocess_user_image(image_file, mean, std):
    # Abrir imagem e converter para RGB
    original_img = Image.open(image_file).convert('RGB')

    # Redimensionar para 32x32
    resized_img = original_img.resize((32, 32), Image.Resampling.LANCZOS)

    # Comprimir imagem para reduzir tamanho (<50 KB)
    buffer = io.BytesIO()
    resized_img.save(buffer, format="JPEG", optimize=True, quality=30)  # Qualidade ajustável (30~85)
    buffer.seek(0)

    # Reabrir imagem comprimida
    compressed_img = Image.open(buffer).convert("RGB")

    # Mostrar tamanho da imagem comprimida
    st.caption(f"📦 Tamanho da imagem após compressão: {buffer.getbuffer().nbytes / 1024:.2f} KB")

    # Converter para array e normalizar
    img_array = np.array(compressed_img).astype('float32')
    if img_array.shape != (32, 32, 3):
        raise ValueError(f"Shape inesperado: {img_array.shape}. Esperado (32, 32, 3)")

    img_array = (img_array - mean) / std
    img_array = np.expand_dims(img_array, axis=0)

    return img_array, original_img

def classify_user_image(model, img_array):
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    return predicted_class_index, predictions[0]

# --- Interface Streamlit ---
st.title("🔍 Reconhecimento de Imagens - CIFAR-10")
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

uploaded_file = st.file_uploader("📷 Envie uma imagem (tirada ou da web):", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img_array, original_img = preprocess_user_image(uploaded_file, mean, std)
    class_index, probs = classify_user_image(model, img_array)
    class_name = class_names[class_index]

    st.image(original_img, caption="📸 Imagem original enviada", use_column_width=True)
    st.success(f"🧠 Classe prevista: **{class_name}**")

    st.subheader("📊 Confiança do modelo para cada classe:")
    for i, prob in enumerate(probs):
        st.write(f"{class_names[i]}: {prob:.4f}")
        st.progress(float(prob))
