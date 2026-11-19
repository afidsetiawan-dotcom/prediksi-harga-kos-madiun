from rest_framework import serializers

class PredictionSerializer(serializers.Serializer):
    jarak_km = serializers.FloatField(min_value=0)
    rating = serializers.FloatField(min_value=0, max_value=5)
    ac = serializers.ChoiceField(choices=["ya", "tidak"])
