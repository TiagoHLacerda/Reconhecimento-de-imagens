from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Função para pré-processar a imagem fornecida pelo usuário
def preprocess_user_image(img_path, mean, std):
    # Carregar a imagem
    img = Image.open(img_path).convert('RGB')  # Garante 3 canais (RGB)

    # Redimensionar para 32x32 pixels
    img = img.resize((32, 32))

    # Converter para array numpy
    img_array = np.array(img)

    # Normalizar usando a mesma média e desvio padrão do treinamento
    img_array = (img_array - mean) / std

    # Adicionar dimensão do batch
    img_array = np.expand_dims(img_array, axis=0)  # De (32,32,3) para (1,32,32,3)

    return img_array, img

# Função para classificar a imagem e exibir o resultado
def classify_user_image(model, img_array, class_names, original_img):
    # Fazer a previsão
    predictions = model.predict(img_array)

    # Pegar o índice da classe com maior probabilidade
    predicted_class_index = np.argmax(predictions, axis=1)[0]

    # Obter o nome da classe
    predicted_class_name = class_names[predicted_class_index]

    # Exibir a imagem junto com a classe prevista
    plt.imshow(original_img)
    plt.title(f"Classe prevista: {predicted_class_name}")
    plt.axis('off')
    plt.show()

    print(f"A imagem foi classificada como: {predicted_class_name}")


if __name__ == "__main__":
    # Caminho da imagem fornecido pelo usuário
    user_img_path = input("Digite o caminho da imagem que deseja classificar: ")

    # Definir class_names aqui, antes de usar na função classify_user_image
    class_names = ["avião", "automóvel", "pássaro", "gato", "veado", "cachorro", "sapo", "cavalo", "navio", "caminhão"]

    # Pré-processar a imagem
    img_array, original_img = preprocess_user_image(user_img_path, mean, std)

    # Classificar e exibir o resultado
    classify_user_image(model, img_array, class_names, original_img)