import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import unittest

import pandas as pd
import requests
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "frontend" / "app.py"


class StreamlitAppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._previous_api_url = os.environ.get("DJANGO_API_URL")
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            cls._port = sock.getsockname()[1]
        cls._api_url = f"http://127.0.0.1:{cls._port}/api/"
        cls._server = subprocess.Popen(
            [
                sys.executable,
                str(ROOT / "backend" / "manage.py"),
                "runserver",
                f"127.0.0.1:{cls._port}",
                "--noreload",
            ],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        os.environ["DJANGO_API_URL"] = cls._api_url
        for _ in range(100):
            if cls._server.poll() is not None:
                raise RuntimeError("Django test server exited before becoming ready.")
            try:
                response = requests.get(cls._api_url + "health/", timeout=0.5)
                if response.ok:
                    return
            except requests.RequestException:
                time.sleep(0.1)
        cls._server.terminate()
        cls._server.wait(timeout=5)
        raise RuntimeError("Django test server did not become ready.")

    @classmethod
    def tearDownClass(cls):
        if cls._server.poll() is None:
            cls._server.terminate()
            cls._server.wait(timeout=5)
        if cls._previous_api_url is None:
            os.environ.pop("DJANGO_API_URL", None)
        else:
            os.environ["DJANGO_API_URL"] = cls._previous_api_url

    def run_page(self, page: str) -> AppTest:
        app = AppTest.from_file(str(APP_PATH), default_timeout=20).run()
        app.radio(key="navigation").set_value(page).run()
        self.assertEqual(app.exception, [])
        return app

    def test_dashboard_shows_dataset_metrics_and_charts(self):
        app = self.run_page(":material/dashboard:  Dashboard")
        self.assertEqual(len(app.metric), 7)
        self.assertEqual(len(app.get("vega_lite_chart")), 4)
        self.assertTrue(
            any("Prediksi Harga Kos Madiun" in item.value for item in app.title)
        )

    def test_navigation_contains_only_the_five_retained_pages(self):
        app = AppTest.from_file(str(APP_PATH), default_timeout=20).run()
        self.assertEqual(
            list(app.radio(key="navigation").options),
            [
                ":material/dashboard:  Dashboard",
                ":material/auto_awesome:  Prediksi Harga",
                ":material/dataset:  Dataset",
                ":material/analytics:  Evaluasi Model",
                ":material/info:  Tentang Aplikasi",
            ],
        )
        self.assertFalse(
            any(
                "Riwayat Prediksi" in item
                for item in app.radio(key="navigation").options
            )
        )

    def test_prediction_calls_api_and_displays_result(self):
        app = self.run_page(":material/auto_awesome:  Prediksi Harga")
        next(button for button in app.button if button.label == "Prediksi Harga Kos").click().run()
        self.assertEqual(app.exception, [])
        prediction = app.session_state["latest_prediction"]
        self.assertGreater(prediction["harga_prediksi_ribu"], 0)
        self.assertEqual(prediction["algoritma"], "Linear Regression")
        self.assertTrue(any("Hasil prediksi" in item.value for item in app.subheader))
        self.assertTrue(any("tidak menjamin harga pasar aktual" in item.value for item in app.info))
        self.assertNotIn("prediction_history", app.session_state)

    def test_dataset_search_and_filters(self):
        app = self.run_page(":material/dataset:  Dataset")
        app.text_input(key="dataset_search").set_value("Kos Mawar").run()
        table = app.dataframe[0].value
        self.assertIsInstance(table, pd.DataFrame)
        self.assertEqual(table["nama"].tolist(), ["Kos Mawar"])

        app.text_input(key="dataset_search").set_value("")
        app.selectbox(key="dataset_ac_filter").set_value("ya")
        app.slider(key="dataset_price_filter").set_value((1000, 1500))
        app.run()
        table = app.dataframe[0].value
        self.assertTrue((table["ac"] == "ya").all())
        self.assertTrue(table["harga"].between(1000, 1500).all())

    def test_evaluation_page_uses_recorded_metrics_and_comparison_chart(self):
        app = self.run_page(":material/analytics:  Evaluasi Model")
        labels = {metric.label for metric in app.metric}
        self.assertIn("MAE (ribu Rp)", labels)
        self.assertIn("R² training evaluator", labels)
        self.assertIn("LOO Dummy MAE", labels)
        r2_test = next(metric for metric in app.metric if metric.label == "R² test")
        self.assertLess(float(r2_test.value), 0)
        self.assertEqual(len(app.get("vega_lite_chart")), 1)
        self.assertTrue(
            any("belum mengungguli baseline" in item.value for item in app.markdown)
        )

    def test_about_page_renders(self):
        app = self.run_page(":material/info:  Tentang Aplikasi")
        self.assertTrue(any("Tentang Aplikasi" in item.value for item in app.title))
        section_titles = {item.value for item in app.subheader}
        self.assertTrue(
            {"Tujuan aplikasi", "Teknologi", "Dataset", "Cara kerja"}.issubset(
                section_titles
            )
        )
        self.assertTrue(
            any("tidak menggunakan database" in item.value for item in app.warning)
        )


if __name__ == "__main__":
    unittest.main()
