O seguinte Script Apresentado se trata de um IDS utilizando o algoritmo KNN adicionando o Machine Learning para melhor performace, Para a realização do script é nescessario do seguinte arquivo:
[Uploading PDFMalware2022_pp.csv…]()


import numpy as np

import pandas as pd

import tensorflow as tf

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

from sklearn.preprocessing import StandardScaler
 
# Carrega os dados
df = pd.read_csv("/content/PDFMalware2022_pp.csv", dtype={"Class": int})
 
# Exibe informações do dataframe
df.head()
df.describe()
df.info()
![df1](https://github.com/user-attachments/assets/3665e8c9-1fa3-49ec-87da-ad990cd40486)

 
# Divide os dados em duas partes (partA e partB)
partA, partB = train_test_split(df, test_size=0.9, random_state=42)
 
# Separa as features e os rótulos para o treinamento na partA
y = partA["Class"]

X = partA.drop("Class", axis=1)
 
# Divide os dados de partA em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Normaliza os dados usando StandardScaler
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
 
# Define o modelo Perceptron Multicamadas (MLP) com camadas ocultas
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Saída binária
])
 ![df2](https://github.com/user-attachments/assets/dec3a803-5e61-48d7-a980-f0063ff7faa5)

# Compila o modelo com um learning rate menor
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
 
# Define o Early Stopping para evitar overfitting
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
 
# Treina o modelo
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])
 ![df3](https://github.com/user-attachments/assets/95fe641b-5fa2-48e4-8db3-d719a9d37a22)

# Avalia o modelo na partB
y = partB["Class"]

X = partB.drop("Class", axis=1)

X = scaler.transform(X)  # Aplica a mesma normalização para partB
 
# Faz previsões com o modelo treinado
y_pred = (model.predict(X) > 0.5).astype("int32").flatten()
 ![df4](https://github.com/user-attachments/assets/5db0b2cd-ede5-47dc-bccf-f29c346b27b0)

# Calcula a matriz de confusão e métricas de avaliação
conf_clf = confusion_matrix(y, y_pred)

tn, fp, fn, tp = conf_clf.ravel()
 
print("TN:", tn)

print("TP:", tp)

print("FP:", fp)

print("FN:", fn)

print()

print("Accuracy:", accuracy_score(y, y_pred) * 100)

print("Precision:", precision_score(y, y_pred) * 100)

print("Recall:", recall_score(y, y_pred) * 100)

![ac96](https://github.com/user-attachments/assets/f33a0cde-77dd-4394-aaee-581cb0332d96)

