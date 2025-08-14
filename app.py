from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Untuk render grafik tanpa display
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import timedelta

import os
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# ============================
# Fungsi pra-proses data
# ============================
def preprocess_data(df):
    def clean_number(val):
        return float(str(val).replace(',', ''))
    for col in ['Price', 'Open', 'High', 'Low']:
        df[col] = df[col].apply(clean_number)

    def convert_vol(val):
        val = str(val).replace(',', '').replace('M', 'e6').replace('K', 'e3')
        return float(eval(val))
    df['Vol.'] = df['Vol.'].apply(convert_vol)

    df['Change %'] = df['Change %'].str.replace('%', '').astype(float)
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
    df = df.sort_values('Date').reset_index(drop=True)
    return df

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Upload file
        if 'file' not in request.files:
            return 'Tidak ada file di form!'
        file = request.files['file']
        if file.filename == '':
            return 'Tidak ada file yang dipilih!'
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Load dan preprocess
            df = pd.read_csv(filepath)
            df = preprocess_data(df)

            # Tentukan min dan max tanggal target
            min_date = (df['Date'].max() + timedelta(days=1)).date()
            max_date = (df['Date'].max() + timedelta(days=30)).date()

            return render_template('index.html',
                                   min_date=min_date,
                                   max_date=max_date,
                                   filename=filename)
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    filename = request.form['filename']
    target_date = request.form['target_date']
    actual_price = request.form.get('actual_price', '')

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    df = pd.read_csv(filepath)
    df = preprocess_data(df)

    # Model
    X = df[['Open', 'High', 'Low', 'Vol.', 'Change %']]
    y = df['Price']

    model = LinearRegression()
    model.fit(X, y)

    last_features = df.iloc[-1][['Open', 'High', 'Low', 'Vol.', 'Change %']].values.reshape(1, -1)
    predicted_price = model.predict(last_features)[0]

    mape = None
    if actual_price and float(actual_price) > 0:
        actual_price = float(actual_price)
        mape = np.mean(np.abs((actual_price - predicted_price) / actual_price)) * 100

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(df['Date'], df['Price'], color='blue', marker='o', label='Data Historis')
    plt.scatter(pd.to_datetime(target_date), predicted_price, color='red', label='Prediksi', zorder=5, s=100)
    if mape is not None:
        plt.scatter(pd.to_datetime(target_date), actual_price, color='green', label='Harga Aktual', zorder=5, s=100)
    plt.xlabel("Tanggal")
    plt.ylabel("Harga")
    plt.title("Prediksi Harga Saham dengan Regresi Linear")
    plt.legend()
    plt.grid(True)

    plot_path = os.path.join('static', 'plot.png')
    plt.savefig(plot_path)
    plt.close()

    return render_template('result.html',
                           target_date=target_date,
                           predicted_price=predicted_price,
                           actual_price=actual_price if actual_price else None,
                           mape=mape,
                           plot_url=plot_path)

if __name__ == '__main__':
    app.run(debug=True)
