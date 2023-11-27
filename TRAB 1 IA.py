import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Supondo que você tenha um DataFrame chamado 'dados' com colunas como 'Nome', 'Marca', 'anoModelo', 'anoReferencia', 'Valor', etc.
# Carregue seus dados (substitua 'seu_arquivo.csv' pelo nome do seu arquivo CSV ou ajuste conforme necessário).
dados = pd.read_csv('seu_dataset_com_classes.csv')

# Obtenha o ano atual (substitua 2023 pelo ano atual real).
ano_atual = 2022

# 1. Calcule a taxa de desvalorização em relação a cada ano.
dados['Taxa_Desvalorizacao'] = ((dados['valor'] - dados.groupby(['marca', 'modelo', 'anoModelo'])['valor'].transform('mean')) / dados.groupby(['marca', 'modelo', 'anoModelo'])['valor'].transform('mean')) * 100


# 2. Calcular a desvalorização percentual desde o ano de lançamento até o ano atual.
dados['Desvalorizacao_Percentual'] = ((dados['valor'] - dados['valor']) / dados['valor']) * 100



# 3. Adicione uma nova coluna 'Classificacao' com base na taxa de desvalorização média.
dados['Classificacao'] = dados['Taxa_Desvalorizacao'].apply(
    lambda taxa: 'Baixa Desvalorizacao' if taxa < -20 else 'Alta Desvalorizacao'
)

# 4. Exiba o resultado.
print(dados[['modelo', 'marca', 'anoModelo', 'anoReferencia', 'Taxa_Desvalorizacao', 'Classificacao']])

# 5. Salve o DataFrame modificado em um novo arquivo CSV.
dados.to_csv('seu_arquivo_modificado.csv', index=False)

# 6. Preparar os dados para o SVM.
X = dados[['anoReferencia', 'Desvalorizacao_Percentual']]  # Recursos de entrada
y = dados['Classificacao']  # Rótulos de saída

# 7. Dividir os dados em conjuntos de treinamento e teste.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 8. Padronizar os recursos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 9. Criar e treinar o modelo SVM.
svm_model = SVC(kernel='linear', C=1.0)  # Kernel é ajustável.
svm_model.fit(X_train_scaled, y_train)

# 10. Fazer previsões no conjunto de teste.
y_pred = svm_model.predict(X_test_scaled)

# 11. Avaliar o desempenho do modelo.
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# 12. Exibir resultados.
print(f'Acuracia: {accuracy}')
print(f'Matriz de Confusao:\n{conf_matrix}')
print(f'Relatorio de Classificacao:\n{class_report}')

# 13. Exibir o DataFrame com as colunas adicionadas.
print(dados[['modelo', 'marca', 'anoModelo', 'anoReferencia', 'Desvalorizacao_Percentual', 'Classificacao']])

# Agora você pode verificar a quantidade de instâncias para cada classe.
quantidade_alta_desvalorizacao = y_test.value_counts()['Alta Desvalorizacao']
quantidade_baixa_desvalorizacao = y_test.value_counts()['Baixa Desvalorizacao']

print(f"Quantidade de instâncias 'Alta Desvalorização': {quantidade_alta_desvalorizacao}")
print(f"Quantidade de instâncias 'Baixa Desvalorização': {quantidade_baixa_desvalorizacao}")