import numpy as np
import pandas as pd
from scipy import stats
from pickle import dump
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, Flatten, Conv1D, MaxPooling1D
from tensorflow.keras.utils import plot_model

from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sktime.performance_metrics.forecasting import mean_absolute_scaled_error

from statsmodels.tsa.arima.model import ARIMA
from math import sqrt

import plotly.express as px
import plotly.graph_objects as go
from keras.models import load_model
from pickle import load
import seaborn as sns
import matplotlib
matplotlib.use('agg')

def load_csv(csv_file):
    df = pd.read_csv(csv_file,
        sep=',',
        names=['datetime', 'voltage', 'current', 'power', 'frequency', 'power_factor', 'current_ch1', 'current_ch2', 'current_ch3'],
        parse_dates=['datetime'],
        # infer_datetime_format=True,
        index_col='datetime',
        skiprows=0,
        na_values=['nan'])
    
    df_day = df.resample('D').mean().fillna(0)

    df_hour = df.resample('h').mean().fillna(0)
    df_hour['day'] = df_hour.index.dayofweek # 0 Senin, 6 Minggu
    df_hour['hour'] = df_hour.index.hour
    df_hour['date'] = df_hour.index.date

    df_day = df.resample('D').mean().fillna(0)
    df_day['day'] = df_day.index.dayofweek # 0 Senin, 6 Minggu
    df_day['date'] = df_day.index.date

    return df, df_day, df_hour

def predict(csv_file, window_size, epochs, batch_size, id):
    df, df_day, df_hour = load_csv(csv_file)
    select ='power'
    header_scale = [select] # univariate
    df_use = df_hour
    scaled = pd.DataFrame()
    scaler = MinMaxScaler(feature_range=(0, 1))
    # scaler = StandardScaler()
    scaler.fit(df_use[header_scale])
    dump(scaler, open('media/'+str(id)+'/scaler_'+ select +'.pkl', 'wb'))
    scaled[header_scale] = scaler.transform(df_use[header_scale])
    # window_size = 24
    col_size = len(header_scale)
    segment, label = segment_signal(scaled, scaled[select], window_size, col_size)
    new_segment = segment
    n_train_time = int(len(new_segment)*0.8)
    # train_test_split = np.random.rand(len(new_segment)) < 0.8
    train_x = new_segment[:n_train_time]
    train_y = label[:n_train_time]
    test_x = new_segment[n_train_time:]
    test_y = label[n_train_time:]

    model = Sequential()
    model.add(LSTM(200, input_shape=(train_x.shape[1], train_x.shape[2])))
    model.add(Dropout(0.1))
    model.add(Flatten())
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    # plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True, rankdir="LR")
    history = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, validation_data=(test_x, test_y), verbose=2, shuffle=True) # Di sini, aku mengatur shuffle=True agar data diacak sebelum setiap epoch.
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper right')
    plt.savefig("media/"+str(id)+"/"+select+'.png')

    # plt.show()

    scaler.fit(df_use[header_scale])
    yhat = model.predict(test_x)
    inv_yhat = scaler.inverse_transform(yhat)
    # invert scaling for actual
    inv_y = test_y.reshape((len(test_y), 1))
    inv_y = scaler.inverse_transform(inv_y)
    # invert scaling for actual
    inv_train_y = train_y.reshape((len(train_y), 1))
    inv_train_y = scaler.inverse_transform(inv_train_y)
    mse = mean_squared_error(inv_y[-200:], inv_yhat[-200:])
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(inv_y, inv_yhat)
    mape = mean_absolute_percentage_error(inv_y, inv_yhat)
    mase = mean_absolute_scaled_error(inv_y, inv_yhat, y_train=inv_train_y)
    model.save('media/'+str(id)+'/LSTM_'+ select +'_model.h5')

    aa=[x for x in range(window_size+len(train_y),window_size+len(train_y)+len(test_y))]
    bb=[x for x in range(len(df_use))]
    plt.figure(figsize=(25,5))
    plt.plot(bb, df_use[select].to_numpy(), marker='.',
    label="Data Histori")
    plt.plot(aa, inv_yhat, 'r', label="Prediksi")
    plt.ylabel(df.columns[0], size=15)
    plt.xlabel('Time step for first 500 hours', size=15)
    plt.legend(fontsize=15)
    plt.savefig('media/'+str(id)+'/Data_Histori.png')
    # plt.show()

    header_scale = ['power']  # Sesuaikan sesuai dengan nama kolom saat fitting scaler
    pred_fut = 100
    hasil = multi_step(df_use[window_size+len(train_y):window_size+len(train_y)+len(test_y)],
                   pred_fut, scaler, model, header_scale, window_size, col_size)
    
    total = np.vstack((inv_yhat, hasil[-pred_fut:]))

    with open('media/'+str(id)+'/LSTM_200_n.npy', 'wb') as f:
        np.save(f, inv_yhat[-200:])
    with open('media/'+str(id)+'/actual.npy', 'wb') as f:
        np.save(f, df_use['power'][window_size+len(train_y):window_size+len(train_y)+len(test_y)].values)
    with open('media/'+str(id)+'/CNN_LSTM_200.npy', 'wb') as f:
        np.save(f, inv_yhat[-200:])
    with open('media/'+str(id)+'/multi.npy', 'wb') as f:
        np.save(f, total)

    mse_arima, rmse_arima, mae_arima, mape_arima, mase_arima = arima(df_hour, id)

    arima(df_hour, id)
    data_analyst(df_day, id)
    pola_harian(df, id)
    prediksi_penggunaan(df_hour, id)
    penggunaan_harian(df_day, id)
    resume(df, id)
    resampling(df, id)
    freq(df, id)

    return rmse, mae, mape, mase, mse_arima, rmse_arima, mae_arima, mape_arima, mase_arima


def arima(df_hour, id):
    # Asumsikan 'select' adalah nama kolom yang ingin Anda gunakan untuk prediksi
    select = 'power'
    # Pastikan hanya menggunakan kolom 'select' dari df_hour
    X = df_hour[select].values

    size = int(len(X) * 0.8)
    train, test = X[0:size], X[size:]
    history = [x for x in train]  # Sekarang 'history' adalah list univariat
    predictions = list()

    # walk-forward validation
    for t in range(len(test)):
        model = ARIMA(history, order=(3, 1, 0))
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)  # Menambahkan observasi ke 'history'
        # print('predicted=%f, expected=%f' % (yhat, obs))

    # evaluate forecasts
    rmse = sqrt(mean_squared_error(test, predictions))
    # print('Test RMSE: %.3f' % rmse)

    # plot forecasts against actual outcomes
    plt.figure(figsize=(25,5))
    plt.plot(test, label='Actual')
    plt.plot(predictions, color='red', label='Predictions')
    plt.legend()
    plt.savefig('media/'+str(id)+'/arima.png')
    # plt.show()

    num_predictions = len(predictions)

    # Perbaiki array x untuk plot
    aa = [x for x in range(700)]  # Untuk data train
    bb = [x for x in range(700, 700 + num_predictions)]  # Untuk prediksi

    plt.figure(figsize=(25,5))
    plt.plot(aa, train[:700], label='Train')  # Batasi train ke 700 poin pertama
    plt.plot(bb, predictions, color='red', label='Predictions')  # Plot semua prediksi
    plt.legend()
    plt.savefig('media/'+str(id)+'/arima_predict.png')
    # plt.offline.plot(plt, auto_open = False, output_type="div")
    # plt.show()
    mse = mean_squared_error(test[-200:], predictions[-200:])
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(test, predictions)
    mape = mean_absolute_percentage_error(test, predictions)
    mase = mean_absolute_scaled_error(test, predictions, y_train=train)

    with open('media/'+str(id)+'/ARIMA_200.npy', 'wb') as f:
        np.save(f, predictions[-200:])

    return mse, rmse, mae, mape, mase

def slope(x1, y1, x2, y2):
    s = (y2-y1)/(x2-x1)
    return s

def data_analyst(df_day, id):
    Y = df_day['power']
    X = np.arange(1,len(df_day)+1)
    reg = LinearRegression().fit(np.vstack(X), Y)
    df_day['regression'] = reg.predict(np.vstack(X))
    print(slope(X[0],df_day['regression'][0],X[-1],df_day['regression'][-1]))

    fig = px.histogram(df_day, x=df_day.index, y="power",
    histfunc="avg", title="Histogram on Date Axes")
    fig.update_traces(xbins_size="M1")
    fig.update_xaxes(showgrid=True, ticklabelmode="period",
    dtick="M1", tickformat="%b\n%Y")
    fig.update_layout(bargap=0.1)
    fig.add_trace(go.Scatter(mode="markers", x=df_day.index,
    y=df_day["power"], name="Penggunaan Harian"))
    fig.add_trace(go.Scatter(mode="lines", x=df_day.index,
    y=df_day["regression"], name="Garis Trend"))
    # fig.update_xaxes(
    # dtick="M1",
    # tickformat="%d %B %Y")
    fig.write_image("media/"+str(id)+"/data_analyst.png") 
    # fig.show()

def pola_harian(df, id):
    df_hour = df.resample('H').mean().fillna(0)
    df_hour['day'] = df_hour.index.dayofweek # 0 Senin, 6 Minggu
    df_hour['hour'] = df_hour.index.hour
    df_hour['date'] = df_hour.index.date

    hari = 6 # 0 Senin, 6 Minggu
    jml_minggu = 10
    i = 0

    # Melakukan iterasi untuk menemukan index terakhir dimana 'hour' == 23
    for index, row in df_hour[:-24:-1].iterrows():
        if row['hour'] == 23:
            break
        i += 1  # Memperbaiki indentasi untuk 'i += 1'

    if i == 0:
        stepback = None
    else:
        stepback = -i

    # Menghitung segmen dari df_hour berdasarkan jumlah minggu dan penyesuaian 'i'
    df_hour_seg = df_hour[-(168*jml_minggu)-i:stepback]

    x_hour = ['00.00', '01.00', '02.00', '03.00', '04.00',
    '05.00', '06.00', '07.00', '08.00', '09.00', '10.00',
    '11.00', '12.00', '13.00', '14.00', '15.00', '16.00',
    '17.00', '18.00', '19.00', '20.00', '21.00', '22.00',
    '23.00']
    # Membuat figure
    fig = go.Figure()

    # Melakukan iterasi untuk setiap tanggal unik dimana 'day' sama dengan hari tertentu
    for date in df_hour_seg.loc[df_hour_seg['day'] == hari]['date'].unique():
        y_minggu = df_hour_seg.loc[df_hour_seg['date'] == date, 'power']
        fig.add_trace(go.Scatter(x=x_hour, y=y_minggu, mode='markers', name=str(date)))

    # Menghitung rata-rata power per jam untuk hari tertentu
    y_mean = df_hour_seg.loc[df_hour_seg['day'] == hari].groupby('hour')['power'].mean()
    fig.add_trace(go.Scatter(x=x_hour, y=y_mean, mode='lines', name='Rata-rata'))
    fig.update_xaxes(title_text="Time", title_standoff=8)
    fig.update_yaxes(title_text="Power (W)", title_standoff=5)
    fig.update_layout(autosize=False, width=800, height=300, margin=dict(l=10, r=10, b=10, t=10, pad=4))
    fig.write_image("media/"+str(id)+"/pola_harian.png") 

def prediksi_power(df_hour, id):
    date_now = '2022-11-11'
    window_size = 24

    # Konversi 'power (W)' ke 'power_use' dalam kWh, asumsikan 'df_hour' sudah ada
    df_hour['power_use'] = df_hour['power']/1000

    # Persiapan data untuk plot
    y_data = df_hour['power_use'][date_now:'2022-11-11 18:00:00']
    y_data = list(y_data) + [None for i in range(24 - len(y_data))]

    # Load model dan scaler
    model = load_model('media/'+str(id)+'/LSTM_power_model.h5')
    scaler = load(open('media/'+str(id)+'/scaler_power.pkl', 'rb'))

    # Scaling 'power (W)'
    scaled = pd.DataFrame(scaler.transform(df_hour[['power']]), columns=['power'])

    # Gunakan fungsi segment_multi (pastikan sudah didefinisikan)
    segment_temp = segment_multi(scaled, window_size, 1)

    # Lakukan prediksi
    pred = model.predict(segment_temp)
    pred = scaler.inverse_transform(pred)[0][0]/1000

    # Penyusunan data prediksi
    first_none = next((enum for enum, i in enumerate(y_data) if i is None), len(y_data))
    y_pred_data = [None for i in range(first_none)] + [pred] + [None for i in range(24 - first_none - 1)]
    x_hour = ['00.00', '01.00', '02.00', '03.00', '04.00',
    '05.00', '06.00', '07.00', '08.00', '09.00', '10.00',
    '11.00', '12.00', '13.00', '14.00', '15.00', '16.00',
    '17.00', '18.00', '19.00', '20.00', '21.00', '22.00',
    '23.00']

    # Membuat figure dan traces
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_hour, y=y_pred_data, mode='lines+markers', line_color="#0cc70c", name='Prediksi Penggunaan', opacity=1.0))
    fig.add_trace(go.Scatter(x=x_hour, y=y_data, mode='lines+markers', line_color="#315BBC", name='Konsumsi Daya'))

    # Update layout dan axes
    fig.update_layout(autosize=False, width=800, height=300, margin=dict(l=10, r=10, b=10, t=10, pad=4), legend=dict(orientation="h", yanchor="top", y=0.99, xanchor="left", x=0.01))
    fig.update_xaxes(tickmode='linear', title_text="Konsumsi Daya (kWh) terhadap Waktu", title_standoff=10)
    fig.update_yaxes(rangemode="tozero")

    # Menampilkan plot
    fig.write_image("media/"+str(id)+"/prediksi_power.png")
    # fig.show()

def prediksi_penggunaan(df_hour, id):
    date_now = '2022-11-11'
    select = 'current_ch1'

    # Define the hours to be displayed on the x-axis
    x_hour = [
        '00.00', '01.00', '02.00', '03.00', '04.00',
        '05.00', '06.00', '07.00', '08.00', '09.00', '10.00',
        '11.00', '12.00', '13.00', '14.00', '15.00', '16.00',
        '17.00', '18.00', '19.00', '20.00', '21.00', '22.00',
        '23.00'
    ]

    # Convert power from Watts to kWh
    df_hour['power_use'] = df_hour['power'] / 1000

    # Prepare your y_data for plotting
    y_data = df_hour['power_use'][date_now:'2022-11-11 17:00:00']
    y_data = list(y_data) + [None for i in range(24 - len(y_data))]
    # Load your scaler
    # Note: There seems to be a mistake in 'wb' mode. It should be 'rb' for reading.
    scaler = load(open('media/'+str(id)+'/scaler_power.pkl', 'rb'))
    # Assuming 'df_use' and 'header_scale' are defined and relevant to your scenario
    # Scale your data
    # df_use[header_scale] = scaler.transform(df_use[header_scale])

    # Prediction placeholder
    pred = 0.123

    # Prepare your prediction data for plotting
    first_none = [enum for enum, i in enumerate(y_data) if i is None][0]
    y_pred_data = [None for i in range(first_none-1)] + [y_data[first_none-1]] + [pred] + [None for i in range(24 - first_none)]

    # Create and display the figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_hour, y=y_pred_data, mode='lines+markers', line_color="#0cc70c", name='Prediksi Penggunaan', opacity=1.0))
    fig.add_trace(go.Scatter(x=x_hour, y=y_data, mode='lines+markers', line_color="#315BBC", name='Konsumsi Daya'))
    fig.update_xaxes(tickmode='linear', title_text="Konsumsi Daya (kWh) terhadap Waktu", title_standoff=10)
    fig.update_yaxes(rangemode="tozero")
    fig.update_layout(autosize=False, width=800, height=300, margin=dict(l=10, r=10, b=10, t=10, pad=4), legend=dict(orientation="h", yanchor="top", y=0.99, xanchor="left", x=0.01))
    # fig.show()
    fig.write_image('media/'+str(id)+'/prediksi_penggunaan.png')

def penggunaan_harian(df_day, id):
    jml_minggu = 12  # Number of weeks
    i = 0

    # Find the index of the last Saturday in the dataset to make full weeks
    for index, row in df_day[:-8:-1].iterrows():
        if row['day'] == 6:  # Assuming 'day' column with Sunday=0 through Saturday=6
            break
        i += 1

    if i == 0:
        stepback = None
    else:
        stepback = -i

    # Select the relevant segment of the dataset
    df_day_seg = df_day[-(7*jml_minggu)-i:stepback]

    # Assign week numbers to each day
    list_minggu = []
    for i in range(1, jml_minggu + 1):
        for j in range(7):
            list_minggu.append(i)

    df_day_seg['minggu'] = list_minggu

    # English day names for the x-axis
    x_day = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', "Friday", 'Saturday', 'Sunday']

    # Create and populate the figure with data
    fig = go.Figure()

    # Plot each week's daily power usage
    for minggu in df_day_seg['minggu'].unique():
        y_minggu = df_day_seg['power'].loc[df_day_seg['minggu'] == minggu]
        fig.add_trace(go.Scatter(x=x_day, y=y_minggu, mode='markers', name='Week ' + str(minggu)))

    # Plot the average daily power usage
    y_mean = df_day_seg.groupby('day')['power'].mean()
    fig.add_trace(go.Scatter(x=x_day, y=y_mean, mode='lines', name='Average'))

    # Update axes labels
    fig.update_xaxes(title_text="Day", title_standoff=8)
    fig.update_yaxes(title_text="Power", title_standoff=5)

    # Update layout
    fig.update_layout(autosize=False, width=800, height=320, margin=dict(l=10, r=10, b=10, t=10, pad=4))

    # Display the plot
    # fig.show()
    fig.write_image('media/'+str(id)+'/penggunaan_harian.png')

def resume(df, id):
    i = 1
    cols = [0, 1, 2, 3, 4, 5, 6, 7]
    plt.figure(figsize=(20, 10))
    for col in cols:
        plt.subplot(len(cols), 1, i)
        plt.plot(df.values[:, col])
        plt.title(df.columns[col], y=0.75, loc='left')
        if col < 7:
            plt.xticks([])
        i += 1
    # plt.show()
    plt.savefig('media/'+str(id)+'/resume.png')

def resampling(df, id):
    plt.figure(figsize=(10,8))
    sns.set_theme(font_scale=1.4)
    sns.heatmap(df.corr(), vmin=-1, vmax=1, annot=True)
    plt.title('Monthly resampling', size=12)
    plt.savefig('media/'+str(id)+'/resampling.png')

def freq(df, id):
    l = df.columns.values
    number_of_columns=8
    number_of_rows = int(len(l)-1/number_of_columns)
    plt.figure(figsize=(2.5*number_of_columns,5*number_of_rows
    ))
    sns.set_theme(font_scale=1.5)
    for i in range(0,len(l)):
        plt.subplot(number_of_rows + 1,number_of_columns,i+1)
        sns.set_style('whitegrid')
        sns.boxplot(y=df[l[i]] ,color='green',orient='v')
        plt.tight_layout()
    plt.savefig('media/'+str(id)+'/freq.png')
    

    plt.figure(figsize=(6*number_of_columns,5*number_of_rows))
    sns.set_theme(font_scale=1.5)
    for i in range(0,len(l)):
        plt.subplot(number_of_rows + 1,number_of_columns,i+1)
        sns.distplot(df[l[i]],kde=True)
    plt.savefig('media/'+str(id)+'/chart.png')

# def last(df):
    

def multi_step(data_in, steps, scaler, model, header_scale, window_size, col_size):
    scaled_temp = pd.DataFrame()
    # Pastikan bahwa data_in memiliki nama kolom yang sesuai
    scaled_temp[header_scale] = scaler.transform(data_in[header_scale])

    for i in range(steps):
        segment_temp = segment_multi(scaled_temp[-window_size:], window_size, col_size)
        pred = model.predict(segment_temp)
        df_append = pd.DataFrame(pred, columns=header_scale)  # Pastikan kolom sesuai dengan fit scaler
        scaled_temp = pd.concat([scaled_temp, df_append], ignore_index=True)

    # Gunakan tail untuk mengambil 'steps' terakhir untuk inverse_transform
    forecast = scaler.inverse_transform(scaled_temp[header_scale].tail(steps))
    return forecast

def segment_multi(data, window_size, col_size):
    segments = np.empty((0, window_size, col_size))
    labels = np.empty((0))
    list_data = []
    for col in data.columns:
        list_data.append(data[col][-window_size:])
    segments = np.vstack([segments, np.dstack(list_data)])
    return segments


def windows(data, size):
    start = 0
    while start < len(data):
        yield int(start), int(start + size)
        start += size / size

def segment_signal(data, data_label, window_size, col_size):
    segments = np.empty((0, window_size, col_size))
    labels = np.empty((0))
    for (start, end) in windows(data.index.tolist(), window_size):
        list_data = []
        for col in data.columns:
            list_data.append(data[col][start:end])
        if len(data.index.tolist()[start:end]) == window_size:
            try:
                # labels = np.append(labels, stats.mode(data_label[start:end])[0][0])
                # label dengan modus
                labels = np.append(labels, data_label.iloc[end-1])  # Fix end index
                segments = np.vstack([segments, np.dstack(list_data)])
            except:
                pass
    return segments, labels

# predict('/Users/user/kerja/mumtaz/core/core/static/09-07-2022_00.00_sampai_11-11-2022_23.59.csv')