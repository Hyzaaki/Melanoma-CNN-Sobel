# Detecção de Melanoma com Redes Neurais Convolucionais e Processamento de Imagens

Este projeto utiliza **Redes Neurais Convolucionais (CNNs)** para classificar imagens de lesões de pele, distinguindo entre lesões benignas e **melanoma**. O pipeline de detecção combina técnicas de processamento digital de imagens, como a **Convolução** e **Correlação** com o filtro de Sobel, para realçar bordas e padrões morfológicos antes da classificação.

O objetivo do trabalho é demonstrar a aplicabilidade de conceitos fundamentais de visão computacional em um problema real e desafiador: o auxílio ao diagnóstico médico de melanoma.

---

## Estrutura do Projeto

O repositório está organizado da seguinte forma:

.
├── app/
│   ├── gerar_dataset_sobel.py   # Script para pré-processar o dataset com filtro de Sobel.
│   ├── gui.py                  # Interface gráfica para testar o modelo em novas
imagens.
│   ├── preprocessamento.py     # Funções manuais de correlação, convolução e
filtro de Sobel.
│   └── treinar_modelo.py       # Script para treinar a CNN.
├── dataset/                    # Contém os dados de entrada e saída.
│   ├── normal/                 # Imagens originais (benignas e melanoma).
│   └── sobel/                  # Imagens processadas com o filtro de Sobel.
├── docs/                       # Onde fica localizado o relatório sobre esse trabalho  
│   └── Relatorio.pdf           
├── img_testes/                 # Imagens de exemplo para testes.
├── modelo/                     # Onde o modelo treinado será salvo.
├── README.md                   # Este arquivo.
└── requirements.txt            # Dependências do projeto.

## Fundamentação Técnica

* **Processamento de Imagens**: O pré-processamento manual das imagens é uma etapa crucial. Ele inclui conversão para tons de cinza, equalização de histograma, suavização com filtro Gaussiano e, por fim, a aplicação do **filtro de Sobel** para detecção de bordas[cite: 36, 138, 139].
* **Correlação e Convolução**: As operações de correlação e convolução 2D são a base da extração de características em imagens[cite: 9, 10]. O projeto implementa manualmente essas operações para fins didáticos, aplicando-as ao filtro de Sobel para realçar as bordas das lesões[cite: 12, 56].
* **CNN (Rede Neural Convolucional)**: Uma arquitetura CNN simples é usada para a classificação final. O modelo é composto por camadas convolucionais (para extrair features), de pooling (para reduzir a dimensionalidade) e camadas densas (para a classificação binária)[cite: 174, 186, 187, 188, 190].

---

## Conjunto de Dados

O projeto utiliza o **Skin Cancer MNIST: HAM10000**, uma base de dados pública com mais de 10.000 imagens dermatoscópicas[cite: 124, 126]. Para o escopo deste trabalho, foram selecionadas as classes `melanoma` (maligno) e `lesões benignas`[cite: 128].

---

## Como Executar

### Pré-requisitos
Certifique-se de ter o **Python 3.10.11** instalado. Recomenda-se o uso de um ambiente virtual.

### Instalação
Clone o repositório e instale as dependências:
```bash
git clone <https://github.com/Hyzaaki/Melanoma-CNN-Sobel.git>
cd <https://github.com/Hyzaaki/Melanoma-CNN-Sobel.git>
pip install -r requirements.txt
```
---

### Passo a Passo
1. **Pré-processar o dataset**: Execute o script para aplicar o filtro de Sobel nas imagens originais.
```bash
python app/gerar_dataset_sobel.py
```
Isso criará a pasta dataset/sobel com as imagens já processadas.

2. **Treinar o modelo**: Use o script de treinamento para criar o modelo de classificação.
```bash
python app/treinar_modelo.py
```
O modelo será salvo em **modelo/modelo_treinado.h5** após 15 épocas, com um desempenho de validação de aproximadamente 76% de acurácia.

3. **Executar a interface gráfica**: Para testar o modelo em uma nova imagem, inicie a interface.
```bash
python app/gui.py
```
A GUI permite selecionar uma imagem, visualizar as transformações de correlação e convolução, e obter a previsão do modelo treinado.

---

## Documentação

Para uma análise técnica detalhada da metodologia, fundamentação teórica e resultados, consulte o relatório completo do projeto:

[**Relatório Completo do Projeto**](docs/Relatorio.pdf)

---

### Contribuições

Sinta-se à vontade para abrir issues ou enviar pull requests. Melhorias incluem:
- Ajustes de hiperparâmetros da CNN.
- Exploração de outras arquiteturas de rede.
- Implementação de outras técnicas de pré-processamento.

---

### Autoria
- Isaac Alves Schuenck
- Pontifícia Universidade Católica de Minas Gerais (PUC-MG)