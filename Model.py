import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

# Carregar modelo
model = load_model("cifar10_model.h5")

# Classes CIFAR-10
class_names = ["avião", "automóvel", "pássaro", "gato", "veado", "cachorro", "sapo", "cavalo", "navio", "caminhão"]

# Estatísticas do CIFAR-10
mean = 125.3
std = 63.0

def center_crop(img: Image.Image) -> Image.Image:
    """Faz crop centralizado quadrado."""
    width, height = img.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return img.crop((left, top, left + side, top + side))

def preprocess_user_image(image_file, mean, std):
    # Abrir imagem e forçar para RGB (mesmo se já for)
    original_img = Image.open(image_file).convert('RGB')

    # Crop centralizado quadrado
    cropped_img = center_crop(original_img)

    # Reduzindo para 32x32 com suavização moderna
    resized_img = cropped_img.resize((32, 32), Image.Resampling.LANCZOS)

    # Garantir que tenha shape (32, 32, 3)
    img_array = np.array(resized_img).astype('float32')
    if img_array.shape != (32, 32, 3):
        raise ValueError(f"Erro: imagem com shape inesperado {img_array.shape}. Esperado (32, 32, 3)")

    # Normalizar
    img_array = (img_array - mean) / std

    # Adicionar dimensão do batch: (1, 32, 32, 3)
    img_array = np.expand_dims(img_array, axis=0)

    return img_array, original_img



def classify_user_image(model, img_array):
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    return predicted_class_index, predictions[0]

# Interface
st.title("🔍 Reconhecimento de Imagens - CIFAR-10")
st.markdown("### Este modelo reconhece as categorias abaixo:")
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

uploaded_file = st.file_uploader("📷 Envie uma imagem:", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img_array, original_img = preprocess_user_image(uploaded_file, mean, std)
    class_index, probs = classify_user_image(model, img_array)
    class_name = class_names[class_index]

    # Exibir imagem original
    st.image(original_img, caption="Imagem original enviada", use_container_width=True)

    # Resultado
    st.success(f"🧠 Classe prevista: **{class_name}**")

    # Mostrar probabilidades
    st.subheader("📊 Confiança do modelo para cada classe:")
    for i, prob in enumerate(probs):
        st.write(f"{class_names[i]}: {prob:.4f}")
        st.progress(float(prob))
