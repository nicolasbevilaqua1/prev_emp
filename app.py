from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Carregar o modelo e o scaler ajustado
model, scaler = joblib.load('model_and_scaler.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    experience = int(request.form['experience'])
    employed = int(request.form['employed'])
    age = int(request.form['age'])
    education = int(request.form['education'])
    
    # Criar um array com os dados
    input_data = np.array([[experience, employed, age, education]])

    # Padronizar os dados de entrada usando o scaler ajustado
    input_data_scaled = scaler.transform(input_data)

    # Fazer a previsão
    prediction = model.predict(input_data_scaled)
    prediction_proba = model.predict_proba(input_data_scaled)[0][1] * 100
    
    return f'A probabilidade de estar exercendo atividade remunerada na área de formação é de {prediction_proba:.2f}%'

if __name__ == '__main__':
    app.run(debug=True)
