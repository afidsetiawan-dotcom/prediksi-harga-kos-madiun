import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "frontend"))

from groq_explainer import (  # noqa: E402
    GroqApiError,
    generate_diagram_explanation,
)


class GroqExplainerTests(unittest.TestCase):
    def successful_response(self):
        response = Mock()
        response.status_code = 200
        response.ok = True
        response.json.return_value = {
            "choices": [{"message": {"content": "Diagram menunjukkan data aktual secara ringkas."}}]
        }
        return response

    def test_successful_groq_response(self):
        with patch("groq_explainer.requests.post", return_value=self.successful_response()) as post:
            text = generate_diagram_explanation(
                api_key="test-key",
                model="openai/gpt-oss-120b",
                title="Distribusi harga",
                description="Jumlah kos per rentang harga",
                data=[{"rentang": "800-900", "jumlah": 5}],
                x_label="Rentang harga",
                y_label="Jumlah kos",
            )
        self.assertIn("aktual", text)
        body = post.call_args.kwargs["json"]
        self.assertEqual(body["temperature"], 0.2)
        self.assertEqual(body["max_completion_tokens"], 350)
        self.assertEqual(body["reasoning_effort"], "low")

    def test_missing_api_key(self):
        with self.assertRaises(GroqApiError) as context:
            generate_diagram_explanation(
                api_key="", model="openai/gpt-oss-120b", title="Test",
                description="Test", data=[],
            )
        self.assertEqual(context.exception.kind, "missing_key")

    def test_empty_response(self):
        response = self.successful_response()
        response.json.return_value = {"choices": [{"message": {"content": "   "}}]}
        with patch("groq_explainer.requests.post", return_value=response):
            with self.assertRaises(GroqApiError) as context:
                generate_diagram_explanation(
                    api_key="test-empty-key", model="openai/gpt-oss-120b",
                    title="Test", description="Test", data=[],
                )
        self.assertEqual(context.exception.kind, "empty")

    def test_rate_limit_response(self):
        response = Mock(status_code=429, ok=False)
        with patch("groq_explainer.requests.post", return_value=response):
            with self.assertRaises(GroqApiError) as context:
                generate_diagram_explanation(
                    api_key="test-rate-key", model="openai/gpt-oss-120b",
                    title="Test", description="Test", data=[],
                )
        self.assertEqual(context.exception.kind, "rate_limit")

    def test_timeout_response(self):
        with patch("groq_explainer.requests.post", side_effect=requests.Timeout):
            with self.assertRaises(GroqApiError) as context:
                generate_diagram_explanation(
                    api_key="test-timeout-key", model="openai/gpt-oss-120b",
                    title="Test", description="Test", data=[],
                )
        self.assertEqual(context.exception.kind, "timeout")


if __name__ == "__main__":
    unittest.main()
