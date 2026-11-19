import os
import unittest

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

import django
from django.apps import apps

if not apps.ready:
    django.setup()

from rest_framework.test import APIClient
from predictor.services import dataset_summary, get_artifact


class PredictorApiTests(unittest.TestCase):
    def setUp(self):
        get_artifact.cache_clear()
        self.client = APIClient(HTTP_HOST="127.0.0.1")

    def test_health(self):
        response = self.client.get("/api/health/")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.data["model_available"])

    def test_prediction(self):
        response = self.client.post("/api/predict/", {"jarak_km": 1.5, "rating": 4.2, "ac": "ya"}, format="json")
        self.assertEqual(response.status_code, 200)
        self.assertIsInstance(response.data["data"]["harga_prediksi_ribu"], float)
        self.assertEqual(response.data["harga_prediksi"], response.data["data"]["harga_prediksi_ribu"])
        self.assertEqual(dataset_summary()["model"]["features"], ["jarak_km", "rating", "ac_num"])

    def test_invalid_input(self):
        response = self.client.post("/api/predict/", {"jarak_km": "salah", "rating": 4.2, "ac": "mungkin"}, format="json")
        self.assertEqual(response.status_code, 400)

    def test_summary(self):
        response = self.client.get("/api/summary/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["jumlah_data"], 30)
        self.assertIn("model", response.data)
        self.assertTrue(response.data["evaluation"]["available"])
        self.assertAlmostEqual(
            response.data["evaluation"]["holdout"]["linear_regression"]["mae"],
            246.05575320952633,
        )
        self.assertAlmostEqual(
            response.data["evaluation"]["leave_one_out"]["dummy_regressor_mean"]["mae"],
            240.0,
        )

    def test_api_does_not_require_a_database(self):
        self.assertEqual(
            django.conf.settings.DATABASES["default"]["ENGINE"],
            "django.db.backends.dummy",
        )
