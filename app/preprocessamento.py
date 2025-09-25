from PIL import Image
import numpy as np
from skimage import exposure
from scipy.ndimage import gaussian_filter

def correlacao2d(imagem, kernel):
    altura, largura = imagem.shape
    kh, kw = kernel.shape
    offset_h, offset_w = kh // 2, kw // 2
    saida = np.zeros_like(imagem, dtype=float)

    for i in range(offset_h, altura - offset_h):
        for j in range(offset_w, largura - offset_w):
            regiao = imagem[i - offset_h:i + offset_h + 1, j - offset_w:j + offset_w + 1]
            resultado = np.sum(regiao * kernel)
            saida[i, j] = resultado

    return saida

def convolucao2d(imagem, kernel):
    kernel_rotacionado = np.flipud(np.fliplr(kernel))
    return correlacao2d(imagem, kernel_rotacionado)

def normaliza(img):
    img = img - np.min(img)
    if np.max(img) != 0:
        img = img / np.max(img)
    return (img * 255).astype(np.uint8)

def aplicar_filtro_sobel(img_path):
    img = Image.open(img_path).convert("L")
    matriz_img = np.array(img)

    # Equaliza o histograma para melhor contraste
    img_eq = exposure.equalize_hist(matriz_img)
    matriz_eq = (img_eq * 255).astype(np.uint8)

    # Suaviza com filtro gaussiano
    matriz_suave = gaussian_filter(matriz_eq, sigma=1)

    sobel_h = np.array([[1, 0, -1],
                        [2, 0, -2],
                        [1, 0, -1]])
    sobel_v = np.array([[1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]])

    # Aplicar convolução usando sua função aprimorada
    conv_h = convolucao2d(matriz_suave, sobel_h)
    conv_v = convolucao2d(matriz_suave, sobel_v)
    magnitude = normaliza(np.hypot(conv_h, conv_v))

    return Image.fromarray(magnitude).convert("RGB")
