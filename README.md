# Sistema Inteligente de Classificação da Qualidade Nutricional de Solução Hidropônica utilizando Perceptron

## Descrição

Este projeto implementa um **Perceptron** em **MATLAB** para realizar a classificação da qualidade nutricional de uma solução hidropônica.

O sistema utiliza informações obtidas de sensores ambientais para determinar se uma solução possui **boa qualidade (1)** ou **má qualidade (-1)**, aplicando um modelo de aprendizado supervisionado.

O objetivo é demonstrar a aplicação de Redes Neurais Artificiais na agricultura de precisão, auxiliando na identificação da qualidade da solução utilizada em sistemas hidropônicos.

---

## Objetivos

- Implementar um Perceptron do zero em MATLAB.
- Realizar treinamento supervisionado.
- Classificar novas amostras.
- Avaliar o desempenho através de métricas.
- Visualizar o comportamento do treinamento utilizando gráficos.

---

## Estrutura do Projeto

```
Projeto/
│
├── Dados/
│   ├── hidroponia_treinamento.csv
│   └── hidroponia_teste.csv
│
├── perceptron_hidroponia.m
│
└── README.md
```

---

## Variáveis de Entrada

O modelo utiliza cinco atributos:

| Variável | Descrição |
|----------|-----------|
| Temperatura da Água | Temperatura da solução hidropônica |
| Temperatura Ambiente | Temperatura do ambiente |
| Umidade | Umidade relativa do ar |
| pH | Potencial hidrogeniônico da solução |
| EC | Condutividade Elétrica |

---

## Funcionamento do Sistema

O algoritmo realiza as seguintes etapas:

1. Leitura dos dados de treinamento e teste.
2. Conversão das classes para o formato do Perceptron (-1 e 1).
3. Normalização dos dados.
4. Inclusão do termo de Bias.
5. Inicialização aleatória dos pesos.
6. Treinamento utilizando a Regra de Aprendizagem do Perceptron.
7. Classificação das amostras de teste.
8. Avaliação do desempenho.
9. Geração de gráficos para análise dos resultados.

---

## Parâmetros Utilizados

| Parâmetro | Valor |
|-----------|-------|
| Taxa de Aprendizagem | 0.01 |
| Máximo de Épocas | 100 |
| Função de Ativação | Degrau |
| Classes | -1 (Ruim) / 1 (Boa) |

---

## Métricas Calculadas

Ao final do treinamento são exibidos:

- Número de acertos
- Número de erros
- Taxa de erro
- Acurácia
- Pesos finais
- Distribuição das classes
- Variável mais influente
- Diagnóstico do modelo

---

## Gráficos Gerados

O programa gera automaticamente seis gráficos:

### 1. Convergência do Perceptron

Mostra a redução do erro ao longo das épocas de treinamento.

---

### 2. Evolução dos Pesos

Apresenta a atualização dos pesos durante o treinamento.

---

### 3. Classe Real × Classe Prevista

Compara as classes reais com as classes previstas pelo modelo.

---

### 4. Distribuição 3D das Amostras

Visualização tridimensional utilizando:

- Temperatura da Água
- pH
- Condutividade Elétrica

---

### 5. Distribuição das Classes

Quantidade de amostras pertencentes a cada classe.

---

### 6. Importância das Variáveis

Exibe o valor absoluto dos pesos aprendidos, indicando quais variáveis possuem maior influência na classificação.

---

## Como Executar

1. Abra o MATLAB.
2. Coloque a pasta `Dados` no mesmo diretório do script.
3. Execute:

```matlab
perceptron_hidroponia
```

Os resultados serão exibidos no terminal e os gráficos serão gerados automaticamente.

---

## Requisitos

- MATLAB R2020 ou superior
- Toolbox básica do MATLAB

---

## Conceitos Aplicados

- Inteligência Artificial
- Redes Neurais Artificiais
- Perceptron
- Aprendizado Supervisionado
- Normalização de Dados
- Classificação Binária
- Agricultura Inteligente

---

## Autor

**EricIkeda1**

---
