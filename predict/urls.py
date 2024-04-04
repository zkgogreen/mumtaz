from django.urls import path
from . import views

urlpatterns = [
    path('', views.new, name="new"),
    path('result/<str:id>', views.result, name='result'),
    path('arima/<str:id>', views.arima, name='arima'),
]