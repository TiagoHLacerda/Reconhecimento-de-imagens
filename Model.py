import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from streamlit_drawable_canvas import st_canvas

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
    if img_array.shape != (32, 32, 3):
        raise ValueError(f"Shape inesperado: {img_array.shape}. Esperado (32, 32, 3)")
    img_array = (img_array - mean) / std
    img_array = np.expand_dims(img_array, axis=0)
    return img_array, resized_img

def classify_user_image(model, img_array):
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    return predicted_class_index, predictions[0]

# --- Interface Streamlit ---
st.title("🔍 Reconhecimento de Imagens - CIFAR-10")

st.markdown("Nos treinamentos com imagens do repositório CIFAR-10, o modelo atingiu **87% de acurácia**.")
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

uploaded_file = st.file_uploader("📷 Envie uma imagem:", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    original_img = Image.open(uploaded_file)
    if original_img.mode != "RGB":
        original_img = original_img.convert("RGB")
    st.markdown("🖱️ **Selecione o objeto principal desenhando um retângulo sobre a imagem abaixo:**")

    background_img = original_img.copy()

    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 255, 0.2)",
        stroke_width=2,
        background_image=background_img,
        update_streamlit=True,
        height=original_img.height,
        width=original_img.width,
        drawing_mode="rect",
        key="canvas"
    )

    if canvas_result.json_data and len(canvas_result.json_data["objects"]) > 0:
        obj = canvas_result.json_data["objects"][-1]
        left = int(obj["left"])
        top = int(obj["top"])
        width = int(obj["width"])
        height = int(obj["height"])

        cropped_img = original_img.crop((left, top, left + width, top + height))
        st.image(cropped_img, caption="📐 Área selecionada (entrada para o modelo)", use_column_width=False)

        img_array, resized_img = preprocess_user_image(cropped_img, mean, std)
        class_index, probs = classify_user_image(model, img_array)
        class_name = class_names[class_index]

        st.success(f"🧠 Classe prevista: **{class_name}**")

        st.subheader("📊 Confiança do modelo para cada classe:")
        for i, prob in enumerate(probs):
            st.write(f"{class_names[i]}: {prob:.4f}")
            st.progress(float(prob))
    else:
        st.warning("⬅️ Desenhe um retângulo sobre a imagem para selecionar o objeto que deseja classificar.")
