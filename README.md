---

📌 Sistema de Detecção de Fraudes com Random Forest

📖 Visão Geral

Este projeto implementa um sistema de detecção de fraudes em transações financeiras usando Machine Learning.
O modelo utiliza Random Forest para classificar transações como fraudulentas (1) ou legítimas (0).

O código inclui:

Geração de dados sintéticos simulando transações financeiras.

Treinamento de um modelo Random Forest para detectar fraudes.

Avaliação do modelo com métricas como acurácia e matriz de confusão.

Visualizações interativas para análise dos resultados.

Interface gráfica (GUI) para personalizar os parâmetros do modelo.

Salvamento do modelo treinado para uso futuro.



---

⚙️ Tecnologias Utilizadas

Python

Machine Learning (scikit-learn)

Visualização de Dados (Matplotlib, Seaborn)

Jupyter Notebook (com ipywidgets para interface gráfica)



---

🚀 Como Executar

1️⃣ Instalar Dependências

pip install pandas numpy matplotlib seaborn scikit-learn joblib ipywidgets

2️⃣ Executar o Código

Abra o Jupyter Notebook e execute o script.
A interface gráfica permitirá ajustar os parâmetros e treinar o modelo.


---

📂 Estrutura do Código

📦 fraud_detection
 ┣ 📜 fraud_detection.ipynb    # Código principal (Jupyter Notebook)
 ┣ 📜 fraud_detection_model.pkl # Modelo treinado salvo
 ┗ 📜 README.md                # Documentação do projeto


---

📌 Funcionalidades

📊 1. Geração de Dados Sintéticos

Função: generate_data(n=10000, fraud_rate=0.02)
📌 Cria um conjunto de dados fictício com transações financeiras.

📜 Colunas geradas:

amount → Valor da transação (1 a 1000).

time → Hora do dia da transação (0 a 24h).

customer_age → Idade do cliente (18 a 80 anos).

transaction_location → Código do local da transação.

merchant_category → Código da categoria do comerciante.

is_fraud → Indica se a transação é fraude (1) ou não (0).



---

🎯 2. Treinamento do Modelo

Função: train_model(df, n_estimators=100)
📌 Treina um modelo Random Forest para classificar transações como fraudulentas ou legítimas.

📜 Etapas:

1. Divide os dados em treino (80%) e teste (20%).


2. Treina um RandomForestClassifier.


3. Retorna o modelo treinado e os conjuntos de treino/teste.




---

📊 3. Avaliação do Modelo

Função: evaluate_model(model, X_test, y_test)
📌 Mede o desempenho do modelo com métricas de classificação.

📜 Métricas calculadas: ✔ Acurácia → Percentual de previsões corretas.
✔ Matriz de Confusão → Compara previsões x valores reais.
✔ Importância das Características → Indica quais variáveis mais influenciam o modelo.


---

📈 4. Visualização dos Resultados

Função: display_results(accuracy, report, cm, feature_importance, X_test, y_test, y_pred)
📌 Exibe gráficos interativos e métricas de desempenho.

📜 Gráficos gerados:

Importância das Características (Gráfico de barras).

Matriz de Confusão (Heatmap).

Distribuição de Transações por valor, idade do cliente e hora do dia.

Gráfico de dispersão (detecção de fraude por valor e horário).


💾 O modelo é salvo automaticamente como 'fraud_detection_model.pkl'.


---

🖥️ 5. Interface Gráfica (GUI)

Função: create_ui()
📌 Permite que o usuário ajuste os parâmetros e treine o modelo com um clique.

📜 Opções disponíveis:

Tamanho da amostra (1.000 a 50.000 transações).

Taxa de fraude (1% a 10%).

Número de árvores (estimadores) no Random Forest (10 a 200).

Botão "Treinar Modelo" para iniciar o treinamento.



---

🔧 Dependências Necessárias

pandas, numpy, matplotlib, seaborn, scikit-learn, joblib, ipywidgets


---
