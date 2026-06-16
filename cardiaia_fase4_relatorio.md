# CardioIA — Fase 4
## Relatório: Pipeline de Pré-processamento e Visão Computacional

**Projeto:** CardioIA — Assistente Cardiológico Virtual  
**Fase:** 4 — Visão Computacional  
**Disciplina:** Cognitive Computing  

---

## 1. Dataset Selecionado

**Chest X-Ray Images (Pneumonia)** — Kaggle (kaggle.com/paultimothymooney/chest-xray-pneumonia)

O dataset contém aproximadamente 5.800 imagens de raio-X de tórax divididas em duas classes: **NORMAL** e **PNEUMONIA**. A escolha justifica-se por três fatores: (a) tamanho viável para treinamento no Google Colab sem requerer hardware especializado; (b) qualidade clínica real, com imagens coletadas de pacientes pediátricos; e (c) alinhamento com o projeto CardioIA, cujo foco é diagnóstico assistido por IA em contextos de saúde cardiovascular e pulmonar.

| Split | NORMAL | PNEUMONIA | Total |
|---|---|---|---|
| Treino (80%) | ~1.082 | ~3.468 | ~4.550 |
| Validação (20%) | ~270 | ~867 | ~1.137 |
| Teste | 234 | 390 | 624 |

> O val set original do dataset possui apenas 16 imagens (insuficiente para validação robusta). Optou-se por reservar 20% do conjunto de treino para validação via `validation_split=0.2`.

---

## 2. Pipeline de Pré-processamento

### 2.1 Redimensionamento

Todas as imagens foram redimensionadas para **224 × 224 pixels**. Essa dimensão foi escolhida por ser o padrão de entrada do VGG16 e ResNet, os modelos pré-treinados utilizados na Parte 2. O redimensionamento garante compatibilidade direta entre o pipeline de dados e os modelos.

### 2.2 Normalização

Os valores de pixel foram normalizados de [0, 255] para **[0, 1]** via divisão por 255 (`rescale=1./255`). A normalização acelera a convergência do gradiente e melhora a estabilidade numérica durante o treinamento.

### 2.3 Conversão de Formato

Todas as imagens foram convertidas para **RGB (3 canais)**, mesmo as originalmente em escala de cinza. Isso garante compatibilidade com os modelos CNN que esperam 3 canais como entrada.

### 2.4 Data Augmentation (conjunto de treino)

Para aumentar a diversidade dos dados de treino e reduzir overfitting, foram aplicadas as seguintes transformações aleatórias:

| Transformação | Valor |
|---|---|
| Rotação | ±10 graus |
| Shift horizontal | 10% da largura |
| Shift vertical | 10% da altura |
| Flip horizontal | Ativado |

O augmentation é aplicado **apenas ao conjunto de treino**, garantindo que validação e teste avaliem o desempenho real do modelo em imagens sem transformação.

### 2.5 Justificativa das Escolhas

A escolha de rotações e shifts moderados (10%) reflete a natureza clínica das imagens: raios-X de tórax possuem orientação padronizada, portanto augmentation agressivo seria clinicamente incoerente (ex.: flip vertical tornaria a imagem anatomicamente impossível).

---

## 3. Modelos Implementados

### 3.1 CNN do Zero

Arquitetura simples com 3 blocos convolucionais progressivos (32 → 64 → 128 filtros) seguidos de uma camada densa com dropout. Treinada por até 15 épocas com EarlyStopping (paciência = 3).

**Justificativa:** Serve como linha de base (baseline) para comparação com Transfer Learning, demonstrando o que é possível alcançar sem conhecimento prévio.

### 3.2 Transfer Learning com VGG16

Base convolucional do VGG16 pré-treinada no ImageNet com pesos **congelados**, seguida de nova cabeça de classificação (`GlobalAveragePooling2D → Dense(256) → Dropout(0.5) → Dense(1, sigmoid)`). Treinada por até 10 épocas.

**Justificativa:** O VGG16 foi treinado em mais de 1 milhão de imagens e aprendeu representações visuais generalizáveis (bordas, texturas, padrões) que transferem bem para imagens médicas, reduzindo a necessidade de grande quantidade de dados.

---

## 4. Métricas de Avaliação

As métricas foram escolhidas para refletir a natureza clínica do problema:

- **Acurácia:** desempenho geral
- **Precisão:** proporção de diagnósticos positivos que são corretos
- **Recall (Sensibilidade):** capacidade de detectar casos reais de pneumonia — **métrica mais crítica** em contexto clínico, pois falsos negativos têm impacto direto na saúde do paciente
- **F1-Score:** equilíbrio entre precisão e recall
- **Matriz de Confusão:** visualização completa dos acertos e erros por classe

---

## 5. Interface de Apresentação

O protótipo de interface foi implementado diretamente no notebook Colab utilizando **ipywidgets**, permitindo ao usuário fazer upload de uma imagem de raio-X e obter o diagnóstico classificado pelo melhor modelo (VGG16 com Transfer Learning) com indicação de confiança percentual.

---

## Considerações Finais

O protótipo demonstra, em ambiente acadêmico controlado, como técnicas de Visão Computacional podem ser aplicadas ao diagnóstico médico assistido por IA. A abordagem adota boas práticas de ética em IA para saúde: transparência nos resultados (exibição da confiança), reconhecimento das limitações do modelo (treinado em dados simulados/públicos) e não substituição do julgamento clínico profissional.
