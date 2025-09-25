import os
from PIL import Image
from preprocessamento import aplicar_filtro_sobel

entrada = "dataset/normal"
saida = "dataset/sobel"

for classe in ["benign", "melanoma"]:
    pasta_entrada = os.path.join(entrada, classe)
    pasta_saida = os.path.join(saida, classe)
    os.makedirs(pasta_saida, exist_ok=True)

    for nome_arquivo in os.listdir(pasta_entrada):
        caminho_entrada = os.path.join(pasta_entrada, nome_arquivo)
        caminho_saida = os.path.join(pasta_saida, nome_arquivo)

        try:
            img_sobel = aplicar_filtro_sobel(caminho_entrada)
            img_sobel.save(caminho_saida)
            print(f"✅ Processada: {nome_arquivo}")
        except Exception as e:
            print(f"❌ Erro em {nome_arquivo}: {e}")
