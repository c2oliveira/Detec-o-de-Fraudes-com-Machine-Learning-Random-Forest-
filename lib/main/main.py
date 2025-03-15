import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from IPython.display import display, HTML
import ipywidgets as widgets
from IPython.display import clear_output

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

def generate_data(n=10000, fraud_rate=0.02):
    np.random.seed(42)
    return pd.DataFrame({
        'amount': np.random.uniform(1, 1000, n),
        'time': np.random.uniform(0, 24, n),
        'customer_age': np.random.randint(18, 100, n),  # Agora a idade máxima é 100
        'transaction_location': np.random.randint(1, 100, n),
        'merchant_category': np.random.randint(1, 50, n),
        'is_fraud': np.random.choice([0, 1], n, p=[1-fraud_rate, fraud_rate])
    })

def train_model(df, n_estimators=100):
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(columns=['is_fraud']), df['is_fraud'],
        test_size=0.2, random_state=42
    )
    
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)
    
    return model, X_train, X_test, y_train, y_test

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    
    feature_importance = pd.DataFrame({
        'feature': X_test.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return accuracy, report, cm, feature_importance, y_pred

def display_results(accuracy, report, cm, feature_importance, X_test, y_test, y_pred):
    clear_output(wait=True)
    
    display(HTML(f"<h2 style='color:#4A90E2; font-family:Arial;'>Modelo de Detecção de Fraudes</h2>"))
    display(HTML(f"<div style='background:#F8F9FA; padding:15px; border-radius:5px; margin:10px 0;'>"
                f"<h3 style='margin:0; color:#333; font-family:Arial;'>Acurácia do Modelo: {accuracy:.4f}</h3></div>"))
    
    plt.figure(figsize=(10, 5))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title('Quais fatores mais influenciam na detecção de fraudes?', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Transação Normal', 'Fraude'],
                yticklabels=['Transação Normal', 'Fraude'])
    plt.xlabel('Previsão do Modelo')
    plt.ylabel('Valor Real')
    plt.title('Erro de Classificação: Comparação entre Previsões e Valores Reais', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    report_df = pd.DataFrame(report).transpose()
    
    translations = {
        '0': 'Transação Normal',
        '1': 'Fraude',
        'accuracy': 'Acurácia',
        'macro avg': 'Média Macro',
        'weighted avg': 'Média Ponderada'
    }
    
    report_df.index = [translations.get(idx, idx) for idx in report_df.index]
    report_df.columns = ['Precisão', 'Revocação', 'F1-Score', 'Amostras']
    
    styled_df = report_df.style.background_gradient(cmap='Blues', subset=['Precisão', 'Revocação', 'F1-Score'])
    styled_df = styled_df.format({
        'Precisão': '{:.4f}',
        'Revocação': '{:.4f}',
        'F1-Score': '{:.4f}',
        'Amostras': '{:.0f}'
    })
    
    display(HTML("<h3 style='font-family:Arial; color:#333;'>Desempenho do Modelo</h3>"))
    display(styled_df)
    
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))

    sns.histplot(X_test['amount'][y_test == 0], color='blue', alpha=0.5, label='Transação Normal', ax=ax[0, 0], kde=True)
    sns.histplot(X_test['amount'][y_test == 1], color='red', alpha=0.5, label='Fraude', ax=ax[0, 0], kde=True)
    ax[0, 0].set_title('Distribuição de Valores das Transações')
    ax[0, 0].set_xlabel('Valor da Transação')
    ax[0, 0].legend()

    sns.histplot(X_test['customer_age'][y_test == 0], color='blue', alpha=0.5, label='Transação Normal', ax=ax[0, 1], kde=True)
    sns.histplot(X_test['customer_age'][y_test == 1], color='red', alpha=0.5, label='Fraude', ax=ax[0, 1], kde=True)
    ax[0, 1].set_title('Idade dos Clientes em Transações Normais e Fraudes')
    ax[0, 1].set_xlabel('Idade do Cliente')
    ax[0, 1].legend()

    scatter = ax[1, 1].scatter(X_test['amount'], X_test['time'], c=y_pred, cmap='coolwarm', alpha=0.6)
    ax[1, 1].set_title('Relação entre o Valor e o Horário da Transação')
    ax[1, 1].set_xlabel('Valor da Transação')
    ax[1, 1].set_ylabel('Hora do Dia')
    legend1 = ax[1, 1].legend(*scatter.legend_elements(), title="Previsão")
    ax[1, 1].add_artist(legend1)

    plt.tight_layout()
    plt.show()
    
    joblib.dump(model, 'fraud_detection_model.pkl')
    display(HTML("<div style='background:#E8F5E9; padding:10px; border-radius:5px; margin:10px 0;'>"
                "<p style='margin:0; font-family:Arial;'>✓ Modelo salvo como 'fraud_detection_model.pkl'</p></div>"))

def create_ui():
    sample_size_slider = widgets.IntSlider(value=10000, min=1000, max=50000, step=1000, description='Tamanho da amostra:')
    fraud_rate_slider = widgets.FloatSlider(value=0.02, min=0.01, max=0.1, step=0.01, description='Taxa de fraude (%):')
    n_estimators_slider = widgets.IntSlider(value=100, min=10, max=200, step=10, description='Número de estimadores:')
    run_button = widgets.Button(description='Treinar Modelo', button_style='primary', icon='play')
    output = widgets.Output()

    def run_model(b):
        with output:
            clear_output()
            df = generate_data(n=sample_size_slider.value, fraud_rate=fraud_rate_slider.value)
            model, X_train, X_test, y_train, y_test = train_model(df, n_estimators=n_estimators_slider.value)
            accuracy, report, cm, feature_importance, y_pred = evaluate_model(model, X_test, y_test)
            display_results(accuracy, report, cm, feature_importance, X_test, y_test, y_pred)

    run_button.on_click(run_model)
    
    display(sample_size_slider, fraud_rate_slider, n_estimators_slider, run_button, output)

if __name__ == "__main__":
    create_ui()
