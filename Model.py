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

# Média e desvio padrão do CIFAR-10
mean = 125.3
std = 63.0

def preprocess_user_image(cropped_img, mean, std):
    resized_img = cropped_img.resize((32, 32), Image.Resampling.LANCZOS)
    img_array = np.array(resized_img).astype('float32')
    img_array = (img_array - mean) / std
    img_array = np.expand_dims(img_array, axis=0)
    return img_array, resized_img

def classify_user_image(model, img_array):
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    return predicted_class_index, predictions[0]

# Interface do app
st.title("🔍 Reconhecimento de Imagens - CIFAR-10")

st.markdown("Nos testes com CIFAR-10, o modelo atingiu **87% de acurácia**.")
st.markdown("### Classes reconhecidas:")
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
    original_img = Image.open(uploaded_file)
    if original_img.mode != "RGB":
        original_img = original_img.convert("RGB")

    st.image(original_img, caption="📸 Imagem original enviada", use_column_width=True)

    img_width, img_height = original_img.size

    st.markdown("🔧 **Defina a área de recorte do objeto:**")
    left = st.slider("Esquerda", 0, img_width - 1, 0)
    top = st.slider("Topo", 0, img_height - 1, 0)
    width = st.slider("Largura", 10, img_width - left, min(100, img_width - left))
    height = st.slider("Altura", 10, img_height - top, min(100, img_height - top))

    # Recortar e exibir
    cropped_img = original_img.crop((left, top, left + width, top + height))
    st.image(cropped_img, caption="📐 Recorte selecionado", use_column_width=False)

    # Classificar
    img_array, resized_img = preprocess_user_image(cropped_img, mean, std)
    class_index, probs = classify_user_image(model, img_array)
    class_name = class_names[class_index]

    st.success(f"🧠 Classe prevista: **{class_name}**")

    st.subheader("📊 Confiança do modelo para cada classe:")
    for i, prob in enumerate(probs):
        st.write(f"{class_names[i]}: {prob:.4f}")
        st.progress(float(prob))
