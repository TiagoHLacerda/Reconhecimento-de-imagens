import streamlit as st
from PIL import Image, ImageEnhance
import numpy as np
from tensorflow.keras.models import load_model

# Carregar modelo treinado
model = load_model("cifar10_model.h5")

# Classes do CIFAR-10
class_names = ["avião", "automóvel", "pássaro", "gato", "veado", "cachorro", "sapo", "cavalo", "navio", "caminhão"]

# Estatísticas CIFAR-10
mean = 125.3
std = 63.0

def center_crop_with_zoom(img: Image.Image, scale: float = 0.6) -> Image.Image:
    """Faz crop central com zoom. Scale < 1 foca no centro."""
    width, height = img.size
    new_width = int(width * scale)
    new_height = int(height * scale)
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    return img.crop((left, top, left + new_width, top + new_height))

def preprocess_user_image(image_file, mean, std):
    # Abrir imagem e converter para RGB
    original_img = Image.open(image_file).convert('RGB')

    # Melhorar contraste e brilho
    original_img = ImageEnhance.Contrast(original_img).enhance(1.2)
    original_img = ImageEnhance.Brightness(original_img).enhance(1.1)

    # Crop central com zoom
    cropped_img = center_crop_with_zoom(original_img, scale=0.6)

    # Redimensionar com suavização moderna
    resized_img = cropped_img.resize((32, 32), Image.Resampling.LANCZOS)

    # Normalizar
    img_array = np.array(resized_img).astype('float32')
    img_array = (img_array - mean) / std
    img_array = np.expand_dims(img_array, axis=0)

    return img_array, original_img, resized_img

def classify_user_image(model, img_array):
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    return predicted_class_index, predictions[0]

# --- Interface do App ---
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

# Upload da imagem
uploaded_file = st.file_uploader("📷 Envie uma imagem tirada ou da web", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img_array, original_img, resized_img = preprocess_user_image(uploaded_file, mean, std)
    class_index, probs = classify_user_image(model, img_array)
    class_name = class_names[class_index]

    st.image(original_img, caption="📸 Imagem original", use_column_width=True)
    st.image(resized_img, caption="📦 Imagem usada pelo modelo (32x32)", width=128)

    st.success(f"🧠 Classe prevista: **{class_name}**")

    st.subheader("📊 Confiança do modelo para cada classe:")
    for i, prob in enumerate(probs):
        st.write(f"{class_names[i]}: {prob:.4f}")
        st.progress(float(prob))
