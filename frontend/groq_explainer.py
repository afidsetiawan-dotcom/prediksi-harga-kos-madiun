import os
from typing import Any

import requests
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError


GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"


class GroqApiError(Exception):
    def __init__(self, message: str, kind: str = "connection"):
        super().__init__(message)
        self.kind = kind


def get_groq_api_key() -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if api_key:
        return api_key.strip()
    try:
        return str(st.secrets["GROQ_API_KEY"]).strip()
    except (KeyError, FileNotFoundError, StreamlitSecretNotFoundError):
        return ""


def get_groq_model() -> str:
    model = os.getenv("GROQ_MODEL")
    if model:
        return model.strip()
    try:
        return str(st.secrets.get("GROQ_MODEL", DEFAULT_GROQ_MODEL)).strip()
    except (KeyError, FileNotFoundError, StreamlitSecretNotFoundError):
        return DEFAULT_GROQ_MODEL


@st.cache_data(ttl=24 * 60 * 60)
def _generate_cached(
    api_key: str,
    model: str,
    title: str,
    description: str,
    x_label: str,
    y_label: str,
    x_unit: str,
    y_unit: str,
    categories: list[str],
    data: list[Any],
    sample_count: int | None,
    additional_metrics: dict[str, Any],
) -> str:
    payload = {
        "title": title,
        "description": description,
        "x_axis": {"label": x_label, "unit": x_unit},
        "y_axis": {"label": y_label, "unit": y_unit},
        "categories": categories,
        "values": data,
        "sample_count": sample_count,
        "additional_metrics": additional_metrics,
    }
    system_prompt = (
        "Anda adalah analis data akademis berbahasa Indonesia. Gunakan hanya data "
        "diagram yang diberikan. Jangan mengarang nilai atau kategori. Jangan menganggap "
        "nilai null sebagai nol; null berarti kategori tidak memiliki data. "
        "korelasi sebagai sebab-akibat. Jangan menyebut R2 sebagai akurasi. Jelaskan MAE, "
        "MSE, RMSE, dan R2 sesuai nilai sebenarnya. R2 negatif harus dijelaskan benar. "
        "Bedakan performa training dan testing. Dataset relatif kecil, hindari kesimpulan "
        "umum yang terlalu luas. Jika data tidak cukup, sebut keterbatasannya. Berikan "
        "2-4 kalimat singkat, informatif, dan cocok untuk presentasi akademik."
    )
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    "Buat penjelasan diagram dari informasi terstruktur berikut:\n"
                    + __import__("json").dumps(payload, ensure_ascii=False)
                ),
            },
        ],
        "temperature": 0.2,
        "max_completion_tokens": 350,
        "reasoning_effort": "low",
    }

    response = requests.post(
        GROQ_ENDPOINT,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=body,
        timeout=(5, 30),
    )
    if response.status_code == 429:
        raise GroqApiError("Permintaan AI sedang mencapai batas penggunaan. Silakan coba kembali nanti.", "rate_limit")
    if not response.ok:
        raise GroqApiError("Penjelasan AI sementara tidak tersedia. Grafik dan fitur lainnya tetap dapat digunakan.", "http")

    try:
        payload_response = response.json()
        explanation = payload_response["choices"][0]["message"]["content"].strip()
    except (ValueError, KeyError, IndexError, TypeError):
        raise GroqApiError("Penjelasan AI sementara tidak tersedia. Grafik dan fitur lainnya tetap dapat digunakan.", "empty")

    if not explanation:
        raise GroqApiError("Penjelasan AI sementara tidak tersedia. Grafik dan fitur lainnya tetap dapat digunakan.", "empty")
    return explanation


def generate_diagram_explanation(
    api_key: str,
    model: str,
    title: str,
    description: str,
    data: list[Any],
    x_label: str = "",
    y_label: str = "",
    x_unit: str = "",
    y_unit: str = "",
    categories: list[str] | None = None,
    sample_count: int | None = None,
    additional_metrics: dict[str, Any] | None = None,
) -> str:
    if not api_key:
        raise GroqApiError(
            "Penjelasan AI belum diaktifkan. Silakan konfigurasi Groq API Key untuk menggunakan fitur ini.",
            "missing_key",
        )
    try:
        return _generate_cached(
            api_key,
            model,
            title,
            description,
            x_label,
            y_label,
            x_unit,
            y_unit,
            categories or [],
            data,
            sample_count,
            additional_metrics or {},
        )
    except requests.Timeout:
        raise GroqApiError("Penjelasan AI sementara tidak tersedia. Grafik dan fitur lainnya tetap dapat digunakan.", "timeout")
    except requests.RequestException:
        raise GroqApiError("Penjelasan AI sementara tidak tersedia. Grafik dan fitur lainnya tetap dapat digunakan.", "connection")


def show_diagram_explanation(
    title: str,
    description: str,
    data: list[Any],
    x_label: str = "",
    y_label: str = "",
    x_unit: str = "",
    y_unit: str = "",
    categories: list[str] | None = None,
    sample_count: int | None = None,
    additional_metrics: dict[str, Any] | None = None,
) -> None:
    api_key = get_groq_api_key()
    model = get_groq_model()

    if not api_key:
        with st.container(border=True, key=f"ai-{title[:30]}"):
            st.info("Penjelasan AI belum diaktifkan. Silakan konfigurasi Groq API Key untuk menggunakan fitur ini.")
        return

    try:
        with st.spinner("Membuat penjelasan diagram...", show_time=False):
            explanation = generate_diagram_explanation(
                api_key=api_key,
                model=model,
                title=title,
                description=description,
                data=data,
                x_label=x_label,
                y_label=y_label,
                x_unit=x_unit,
                y_unit=y_unit,
                categories=categories,
                sample_count=sample_count,
                additional_metrics=additional_metrics,
            )
        with st.container(border=True, key=f"ai-{title[:30]}"):
            st.markdown("**✨ GENERATED BY AI**")
            st.markdown("**Penjelasan Diagram**")
            st.write(explanation)
            st.caption(f"Generated using Groq API · {model}")
    except GroqApiError as exc:
        with st.container(border=True, key=f"ai-{title[:30]}"):
            st.info(str(exc))
