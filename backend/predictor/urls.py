from django.urls import path
from .views import HealthView, PredictView, SummaryView

urlpatterns = [
    path("health/", HealthView.as_view()),
    path("predict/", PredictView.as_view()),
    path("summary/", SummaryView.as_view()),
]
