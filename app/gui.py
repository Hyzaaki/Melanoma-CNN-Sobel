import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from preprocessamento import aplicar_filtro_sobel, correlacao2d, convolucao2d, normaliza
from skimage import exposure
from scipy.ndimage import gaussian_filter

# Carrega o modelo treinado
modelo = tf.keras.models.load_model("modelo/modelo_treinado.h5")

# Interface
def selecionar_imagem():
    caminho = filedialog.askopenfilename()
    if not caminho:
        return

    imagem_original = Image.open(caminho).resize((150, 150))

    # Aplica filtro de Sobel (pré-processamento do trabalho)
    imagem_filtrada = aplicar_filtro_sobel(caminho)
    imagem_filtrada = imagem_filtrada.resize((150, 150))

    img_array = np.array(imagem_filtrada) / 255.0
    img_array = img_array.reshape((1, 150, 150, 3))

    pred = modelo.predict(img_array)[0][0]

    # Adaptação para o novo tema (0 = benigno, 1 = melanoma)
    resultado = "Lesão Benigna" if pred < 0.5 else "Melanoma"

    img_tk = ImageTk.PhotoImage(imagem_original)
    painel.configure(image=img_tk)
    painel.image = img_tk
    resultado_label.config(text=f"Resultado: {resultado}")

    # VISUALIZAÇÃO EXTRA: Correlação e Convolução (não afeta o modelo)
    matriz_img = np.array(imagem_original.convert("L"))
    img_eq = exposure.equalize_hist(matriz_img)
    matriz_eq = (img_eq * 255).astype(np.uint8)
    matriz_suave = gaussian_filter(matriz_eq, sigma=1)

    sobel_h = np.array([[1, 0, -1],
                        [2, 0, -2],
                        [1, 0, -1]])

    sobel_v = np.array([[1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]])

    img_corr = normaliza(np.hypot(
        correlacao2d(matriz_suave, sobel_h),
        correlacao2d(matriz_suave, sobel_v)
    ))

    img_conv = normaliza(np.hypot(
        convolucao2d(matriz_suave, sobel_h),
        convolucao2d(matriz_suave, sobel_v)
    ))

    # Soma entre correlação e convolução com Sobel
    soma = normaliza((img_corr.astype(np.float32) + img_conv.astype(np.float32)) / 2)


    # Mostrar os resultados em matplotlib (sem interferir na predição)
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title("Soma Corr + Conv")
    plt.imshow(soma, cmap='gray')
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Correlação - Sobel")
    plt.imshow(img_corr, cmap='gray')
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Convolução - Sobel")
    plt.imshow(img_conv, cmap='gray')
    plt.axis("off")
    
    plt.tight_layout()
    plt.show()

# Configuração da janela
janela = tk.Tk()
janela.title("Classificador de Lesões de Pele")

btn = tk.Button(janela, text="Selecionar Imagem", command=selecionar_imagem)
btn.pack()

painel = tk.Label(janela)
painel.pack()

resultado_label = tk.Label(janela, text="", font=("Arial", 12, "bold"))
resultado_label.pack(pady=10)

janela.mainloop()
