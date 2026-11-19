import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import pandas as pd
import requests
from streamlit.testing.v1 import AppTest

from ml.evaluate_model import build_evaluation_report
from ml.train_model import (
    canonical_dataset_sha256,
    fit_production_model,
)

ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "frontend" / "app.py"


class StreamlitAppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._previous_api_url = os.environ.get("DJANGO_API_URL")
        cls._previous_prediction_mode = os.environ.get("PREDICTION_MODE")
        cls._previous_groq_api_key = os.environ.get("GROQ_API_KEY")
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
        os.environ["PREDICTION_MODE"] = "django"
        os.environ["GROQ_API_KEY"] = ""
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
        if cls._previous_prediction_mode is None:
            os.environ.pop("PREDICTION_MODE", None)
        else:
            os.environ["PREDICTION_MODE"] = cls._previous_prediction_mode
        if cls._previous_groq_api_key is None:
            os.environ.pop("GROQ_API_KEY", None)
        else:
            os.environ["GROQ_API_KEY"] = cls._previous_groq_api_key

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


class CommunityCloudDirectModeTests(unittest.TestCase):
    def test_evaluation_snapshot_uses_the_canonical_dataset_and_holdout(self):
        report = build_evaluation_report()
        production = fit_production_model()

        self.assertEqual(report["dataset"]["sha256"], canonical_dataset_sha256())
        self.assertEqual(report["dataset"]["rows"], 30)
        self.assertEqual(report["holdout"]["training_rows"], 24)
        self.assertEqual(report["holdout"]["test_rows"], 6)
        self.assertEqual(
            report["holdout"]["linear_regression"], production["metrics"]
        )

    def test_dataset_fingerprint_ignores_newline_convention(self):
        original = (ROOT / "data" / "kos_madiun.csv").read_bytes()
        lf_contents = original.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        crlf_contents = lf_contents.replace(b"\n", b"\r\n")

        with tempfile.TemporaryDirectory() as temp_dir:
            lf_path = Path(temp_dir) / "lf.csv"
            crlf_path = Path(temp_dir) / "crlf.csv"
            lf_path.write_bytes(lf_contents)
            crlf_path.write_bytes(crlf_contents)

            self.assertEqual(
                canonical_dataset_sha256(lf_path),
                canonical_dataset_sha256(crlf_path),
            )

    def test_direct_mode_pages_and_diagram_fallbacks_do_not_call_django(self):
        with patch.dict(
            os.environ,
            {"PREDICTION_MODE": "direct", "GROQ_API_KEY": ""},
        ):
            with patch("requests.get") as get, patch("requests.post") as post:
                app = AppTest.from_file(str(APP_PATH), default_timeout=20).run()
                self.assertEqual(app.exception, [])
                self.assertEqual(len(app.get("vega_lite_chart")), 4)
                self.assertTrue(
                    any(
                        "Mode prediksi" == metric.label and metric.value == "Langsung"
                        for metric in app.metric
                    )
                )
                self.assertEqual(
                    sum("Penjelasan AI belum diaktifkan" in item.value for item in app.info),
                    4,
                )

                app.radio(key="navigation").set_value(
                    ":material/dataset:  Dataset"
                ).run()
                self.assertEqual(app.exception, [])
                self.assertEqual(len(app.get("vega_lite_chart")), 2)
                self.assertEqual(
                    sum("Penjelasan AI belum diaktifkan" in item.value for item in app.info),
                    2,
                )

                app.radio(key="navigation").set_value(
                    ":material/analytics:  Evaluasi Model"
                ).run()
                self.assertEqual(app.exception, [])
                self.assertEqual(len(app.get("vega_lite_chart")), 1)
                self.assertEqual(
                    sum("Penjelasan AI belum diaktifkan" in item.value for item in app.info),
                    1,
                )

                app.radio(key="navigation").set_value(
                    ":material/info:  Tentang Aplikasi"
                ).run()
                self.assertEqual(app.exception, [])
                get.assert_not_called()
                post.assert_not_called()

    def test_direct_prediction_supports_both_ac_values_and_rupiah_conversion(self):
        from ml.train_model import fit_production_model

        artifact = fit_production_model()
        with patch.dict(
            os.environ,
            {"PREDICTION_MODE": "direct", "GROQ_API_KEY": ""},
        ):
            with patch("requests.get") as get, patch("requests.post") as post:
                app = AppTest.from_file(str(APP_PATH), default_timeout=20).run()
                app.radio(key="navigation").set_value(
                    ":material/auto_awesome:  Prediksi Harga"
                ).run()
                for ac in ("ya", "tidak"):
                    app.selectbox(key="prediction_ac").set_value(ac).run()
                    next(
                        button
                        for button in app.button
                        if button.label == "Prediksi Harga Kos"
                    ).click().run()
                    self.assertEqual(app.exception, [])
                    prediction = app.session_state["latest_prediction"]
                    self.assertEqual(prediction["ac"], ac)
                    model_input = pd.DataFrame(
                        [[
                            prediction["jarak_km"],
                            prediction["rating"],
                            artifact["ac_mapping"][ac],
                        ]],
                        columns=artifact["features"],
                    )
                    expected_price = float(artifact["model"].predict(model_input)[0])
                    self.assertEqual(
                        prediction["harga_prediksi_rupiah"],
                        round(expected_price * 1000),
                    )
                    self.assertEqual(prediction["algoritma"], "Linear Regression")
                get.assert_not_called()
                post.assert_not_called()

    def test_production_model_uses_all_rows_and_preserves_metrics(self):
        from ml.train_model import fit_production_model

        artifact = fit_production_model()
        report = json.loads(
            (ROOT / "ml" / "artifacts" / "evaluation_metrics.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(artifact["training_rows"], 30)
        self.assertEqual(artifact["features"], ["jarak_km", "rating", "ac_num"])
        self.assertEqual(artifact["ac_mapping"], {"tidak": 0, "ya": 1})
        self.assertEqual(
            artifact["metrics"], report["holdout"]["linear_regression"]
        )


if __name__ == "__main__":
    unittest.main()
