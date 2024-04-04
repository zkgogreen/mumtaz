from django.shortcuts import render, redirect
from .models import predict as Predict
from django.core.files.storage import default_storage
from django.core.files.storage import FileSystemStorage
from core.utils import predict,  load_csv, arima

# Create your views here.
def new(request):
    context = {}
    if request.method == 'POST':
        windows_size = request.POST['windows_size'] #24
        epochs = request.POST['epochs']             #50
        batch_size = request.POST['batch_size']      #400
        file = request.FILES['file']
        predict_result = Predict.objects.create(windows_size=windows_size, epochs=epochs, batch_size=batch_size)
        
        fs = FileSystemStorage()
        fs.save(str(predict_result.id)+'/main.csv', file)
        rmse, mae, mape, mase, mse_arima, rmse_arima, mae_arima, mape_arima, mase_arima = predict(file, int(windows_size), int(epochs), int(batch_size), predict_result.id)

        predict_result.rmse_predict = rmse
        predict_result.mae_predict = mae
        predict_result.mape_predict = mape
        predict_result.mase_predict = mase
        predict_result.mse_arima = mse_arima
        predict_result.rmse_arima = rmse_arima
        predict_result.mae_arima = mae_arima
        predict_result.mape_arima = mape_arima
        predict_result.mase_arima = mase_arima
        predict_result.save()
        return redirect('predict:result', id=predict_result.id)
    return render(request, 'new.html', context)

def result(request, id):
    context = {
        'predict':Predict.objects.get(id=id)
    }
    return render(request, 'result.html', context)

def arima(request, id):
    df, df_day, df_hour = load_csv('media/'+ str(id) +'/main.csv')
    mse, rmse, mae, mape, mase = arima(df_hour, id=id)
    Predict.objects.filter(id=id).update(mse_arima=mse , rmse_arima= rmse, mae_arima= mae, mape_arima= mape, mase_arima=mase)
    return redirect('predict:result', id=id)


