import math
from django.utils import timezone
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView
from .serializers import PredictionSerializer
from .services import dataset_summary, get_artifact, predict_price

class HealthView(APIView):
    def get(self, request):
        try:
            artifact = get_artifact()
            return Response({"status": "ok", "model_available": True, "algorithm": artifact["algorithm"]})
        except FileNotFoundError as exc:
            return Response({"status": "degraded", "model_available": False, "detail": str(exc)}, status=503)

class PredictView(APIView):
    def post(self, request):
        serializer = PredictionSerializer(data=request.data)
        if not serializer.is_valid():
            return Response({"success": False, "errors": serializer.errors}, status=status.HTTP_400_BAD_REQUEST)
        try:
            price = predict_price(**serializer.validated_data)
        except FileNotFoundError as exc:
            return Response({"success": False, "detail": str(exc)}, status=503)
        if not math.isfinite(price) or price < 0:
            return Response(
                {"success": False, "detail": "Model menghasilkan harga yang tidak valid."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
        rounded_price = round(price, 2)
        price_rupiah = round(price * 1000)
        algorithm = get_artifact()["algorithm"]
        data = {
            **serializer.validated_data,
            "harga_prediksi_ribu": rounded_price,
            "harga_prediksi_rupiah": price_rupiah,
            "algoritma": algorithm,
            "created_at": timezone.now().isoformat(),
        }
        return Response({
            "success": True,
            "message": "Prediksi berhasil",
            "data": data,
            # Keep the original top-level fields for existing clients.
            "harga_prediksi": rounded_price,
            "harga_rupiah": price_rupiah,
            "satuan": "ribu rupiah per bulan",
        })

class SummaryView(APIView):
    def get(self, request):
        return Response(dataset_summary())
