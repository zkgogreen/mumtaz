from django.db import models

# Create your models here.
class predict(models.Model):
    file            = models.CharField(max_length=225, blank=True, null=True)
    created         = models.DateField(auto_now_add=True)
    windows_size    = models.IntegerField(default=24)
    epochs          = models.IntegerField(default=50)
    batch_size      = models.IntegerField(default=400)
    rmse_predict    = models.IntegerField(default=0)
    mae_predict     = models.IntegerField(default=0)
    mape_predict    = models.IntegerField(default=0)
    mase_predict    = models.IntegerField(default=0)
    mse_arima       = models.IntegerField(default=0)
    rmse_arima      = models.IntegerField(default=0)
    mae_arima       = models.IntegerField(default=0)
    mape_arima      = models.IntegerField(default=0)
    mase_arima      = models.IntegerField(default=0)