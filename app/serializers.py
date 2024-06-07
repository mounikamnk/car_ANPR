from rest_framework import serializers

class NumberPlateSerializer(serializers.Serializer):
    plate = serializers.CharField(max_length=25)
