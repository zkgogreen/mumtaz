from django.shortcuts import render, redirect
from predict.models import predict as Predict

from .utils import predict,  load_csv, arima


def index(request):
    context = {
        'history':Predict.objects.all()
    }

    return render(request, 'index.html', context)

def arima_views(request, id):
    # if request.method == 'POST':
    df, df_day, df_hour = load_csv('media/main.csv')
    mse, rmse, mae, mape, mase = arima(df_hour)
    context = {
        'mse':mse,
        'rmse':rmse,
        'mae':mae,
        'mape':mape,
        'mase':mase
    }
    return render(request, 'arima.html', context)