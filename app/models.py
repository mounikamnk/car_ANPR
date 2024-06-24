from django.db import models

# Create your models here.

class CarNumberPlate(models.Model):
    number_plate = models.CharField(max_length=20)
    detected_at = models.DateTimeField(auto_now_add=True)
    location = models.CharField(max_length=255, blank=True, null=True)

    def __str__(self):
        return self.number_plate