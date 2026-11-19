import hashlib
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
import altair as alt
from dotenv import load_dotenv
from groq_explainer import show_diagram_explanation
from streamlit.errors import StreamlitSecretNotFoundError

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
from ml.train_model import fit_production_model

load_dotenv(ROOT_DIR / ".env")


def get_app_setting(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value and value.strip():
        return value.strip()
    try:
        return str(st.secrets.get(name, default)).strip()
    except StreamlitSecretNotFoundError:
        return default


PREDICTION_MODE = get_app_setting("PREDICTION_MODE", "django").lower()
if PREDICTION_MODE not in {"django", "direct"}:
    raise ValueError("PREDICTION_MODE harus bernilai 'django' atau 'direct'.")
API_URL = (
    os.getenv("DJANGO_API_URL", "http://127.0.0.1:8000/api/").rstrip("/") + "/"
    if PREDICTION_MODE == "django"
    else None
)
DATA_PATH = ROOT_DIR / "data" / "kos_madiun.csv"
EVALUATION_PATH = ROOT_DIR / "ml" / "artifacts" / "evaluation_metrics.json"

PAGES = [
    ":material/dashboard:  Dashboard",
    ":material/auto_awesome:  Prediksi Harga",
    ":material/dataset:  Dataset",
    ":material/analytics:  Evaluasi Model",
    ":material/info:  Tentang Aplikasi",
]
PREDICTION_CAVEAT = (
    "Hasil prediksi merupakan estimasi berdasarkan model Machine Learning yang dilatih "
    "menggunakan dataset terbatas. Hasil tidak menjamin harga pasar aktual."
)
MADIUN_TIMEZONE = timezone(timedelta(hours=7))
st.set_page_config(
    page_title="Prediksi Harga Kos Madiun",
    page_icon=":material/home:",
    layout="wide",
)

st.html(
    """
    <style>
      .st-key-page-header,
      .st-key-prediction-form,
      .st-key-prediction-results {
        border-radius: 18px;
        box-shadow: 0 8px 24px rgba(52, 64, 84, 0.055);
      }
      .st-key-page-header {
        background: linear-gradient(135deg, #FFFFFF 0%, #FCEEF2 100%);
      }
      .st-key-prediction-form {
        background: #FFFFFF;
      }
      .st-key-prediction-results {
        background: linear-gradient(135deg, #FFFFFF 0%, #F8F5FC 100%);
      }
      .st-key-navigation {
        margin-top: 0.25rem;
      }
      .st-key-run_prediction button {
        min-height: 48px;
        font-size: 14px;
        font-weight: 600;
      }
      .st-key-navigation label {
        min-height: 44px;
      }
    </style>
    """
)


@st.cache_data(ttl=60)
def fetch_summary():
    if API_URL is None:
        raise RuntimeError("Summary API hanya tersedia dalam mode Django.")
    response = requests.get(API_URL + "summary/", timeout=5)
    response.raise_for_status()
    return response.json()


@st.cache_data
def load_dataset():
    frame = pd.read_csv(DATA_PATH)
    frame["ac"] = frame["ac"].astype(str).str.strip().str.lower()
    return frame


@st.cache_resource
def load_direct_model() -> dict:
    return fit_production_model()


def get_direct_summary() -> dict:
    dataset = load_dataset()
    artifact = load_direct_model()
    try:
        report = json.loads(EVALUATION_PATH.read_text(encoding="utf-8"))
        dataset_hash = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest().upper()
        if report["dataset"]["sha256"].upper() != dataset_hash:
            evaluation = {
                "available": False,
                "status": "dataset_mismatch",
                "detail": "Dataset berubah sejak evaluasi; metrik perlu diperbarui.",
            }
        elif report["holdout"]["linear_regression"] != artifact["metrics"]:
            evaluation = {
                "available": False,
                "status": "model_metrics_mismatch",
                "detail": "Metrik evaluasi tidak cocok dengan model yang dilatih.",
            }
        else:
            evaluation = {"available": True, "status": "current", **report}
    except FileNotFoundError:
        evaluation = {
            "available": False,
            "status": "missing",
            "detail": "File evaluasi belum tersedia.",
        }
    except (json.JSONDecodeError, KeyError, TypeError):
        evaluation = {
            "available": False,
            "status": "invalid",
            "detail": "File evaluasi tidak valid.",
        }

    return {
        "jumlah_data": int(len(dataset)),
        "rata_rata_harga": float(dataset["harga"].mean()),
        "harga_minimum": float(dataset["harga"].min()),
        "harga_maksimum": float(dataset["harga"].max()),
        "rata_rata_harga_ac": float(dataset.loc[dataset["ac"] == "ya", "harga"].mean()),
        "rata_rata_harga_tanpa_ac": float(
            dataset.loc[dataset["ac"] == "tidak", "harga"].mean()
        ),
        "model": {
            "algorithm": artifact["algorithm"],
            "features": artifact["features"],
            "training_rows": artifact["training_rows"],
            "metrics": artifact["metrics"],
        },
        "evaluation": evaluation,
    }


def get_summary() -> dict:
    if PREDICTION_MODE == "direct":
        return get_direct_summary()
    return fetch_summary()


def predict_direct(jarak_km: float, rating: float, ac: str) -> dict:
    artifact = load_direct_model()
    row = pd.DataFrame(
        [[jarak_km, rating, artifact["ac_mapping"][ac]]],
        columns=artifact["features"],
    )
    price = float(artifact["model"].predict(row)[0])
    if not np.isfinite(price) or price < 0:
        raise ValueError("Model menghasilkan harga yang tidak valid.")
    return {
        "jarak_km": jarak_km,
        "rating": rating,
        "ac": ac,
        "harga_prediksi_ribu": round(price, 2),
        "harga_prediksi_rupiah": round(price * 1000),
        "algoritma": artifact["algorithm"],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def backend_status() -> dict:
    if API_URL is None:
        raise RuntimeError("Health API hanya tersedia dalam mode Django.")
    try:
        response = requests.get(API_URL + "health/", timeout=3)
        payload = response.json()
        if not isinstance(payload, dict):
            return {
                "reachable": True,
                "model_available": False,
                "detail": "Backend mengirim respons health yang tidak valid.",
            }
        return {
            "reachable": True,
            "model_available": bool(payload.get("model_available")),
            "algorithm": payload.get("algorithm"),
            "detail": payload.get("detail"),
        }
    except requests.RequestException as exc:
        return {"reachable": False, "model_available": False, "detail": str(exc)}
    except ValueError as exc:
        return {
            "reachable": True,
            "model_available": False,
            "detail": f"Respons health bukan JSON yang valid: {exc}",
        }


def format_id_number(value: float, decimals: int = 2) -> str:
    formatted = f"{value:,.{decimals}f}"
    return formatted.replace(",", "_").replace(".", ",").replace("_", ".")


def format_rupiah(value: float) -> str:
    return f"Rp {format_id_number(value, 0)}"


def format_timestamp(value: str) -> str:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.astimezone(MADIUN_TIMEZONE).strftime("%d %b %Y, %H:%M:%S WIB")
    except (AttributeError, ValueError):
        return value


def price_distribution(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["Jumlah kos"])
    minimum = float(frame["harga"].min())
    maximum = float(frame["harga"].max())
    if minimum == maximum:
        return pd.DataFrame(
            {"Jumlah kos": [len(frame)]},
            index=[f"{format_id_number(minimum, 0)}"],
        )
    edges = np.linspace(minimum, maximum, 9)
    labels = [
        f"{format_id_number(start, 0)}–{format_id_number(end, 0)}"
        for start, end in zip(edges[:-1], edges[1:])
    ]
    bins = pd.cut(frame["harga"], bins=edges, labels=labels, include_lowest=True)
    counts = bins.value_counts(sort=False)
    counts.index.name = "Rentang harga (ribu)"
    return counts.to_frame("Jumlah kos")


def average_price_by_ac(frame: pd.DataFrame) -> pd.DataFrame:
    averages = frame.groupby("ac")["harga"].mean()
    return pd.DataFrame(
        [
            {
                "Dengan AC": averages.get("ya"),
                "Tanpa AC": averages.get("tidak"),
            }
        ],
        index=["Harga rata-rata (ribu)"],
    )


def build_relationship_chart(
    frame: pd.DataFrame,
    x_field: str,
    x_title: str,
    y_field: str,
    y_title: str,
) -> alt.Chart:
    chart_data = frame.assign(
        fasilitas=frame["ac"].map({"ya": "Dengan AC", "tidak": "Tanpa AC"})
    )
    return (
        alt.Chart(chart_data)
        .mark_circle(size=82, opacity=0.82)
        .encode(
            x=alt.X(f"{x_field}:Q", title=x_title),
            y=alt.Y(f"{y_field}:Q", title=y_title, scale=alt.Scale(zero=False)),
            color=alt.Color(
                "fasilitas:N",
                title="Fasilitas",
            ),
            tooltip=[
                alt.Tooltip("nama:N", title="Nama kos"),
                alt.Tooltip("jarak_km:Q", title="Jarak (km)", format=".1f"),
                alt.Tooltip("rating:Q", title="Rating", format=".1f"),
                alt.Tooltip("fasilitas:N", title="Fasilitas"),
                alt.Tooltip("harga:Q", title="Harga (ribu)", format=",.0f"),
            ],
        )
        .properties(height=300)
        .configure_axis(
            gridColor="#F0E7EA",
            labelColor="#667085",
            titleColor="#344054",
            tickColor="#F0E7EA",
            domainColor="#F0E7EA",
            labelFontSize=12,
            titleFontSize=13,
            titleFontWeight=600,
        )
        .configure_legend(
            labelColor="#667085",
            titleColor="#344054",
            labelFontSize=13,
            titleFontSize=13,
        )
        .configure_view(stroke=None)
    )


def show_backend_status(status: dict) -> None:
    if status.get("mode") == "direct":
        st.sidebar.success("Prediksi langsung aktif")
        return
    if status["reachable"] and status["model_available"]:
        st.sidebar.success("Backend dan model aktif")
    elif status["reachable"]:
        st.sidebar.warning("Backend aktif, model belum tersedia")
        if status.get("detail"):
            st.sidebar.caption(status["detail"])
    else:
        st.sidebar.error("Backend tidak terhubung")
        st.sidebar.caption("Periksa server Django dan DJANGO_API_URL.")


def show_prediction(entry: dict, price_rupiah: int | None = None) -> None:
    price_rupiah = price_rupiah or round(entry["harga_prediksi_ribu"] * 1000)
    with st.container(border=True, key="prediction-results"):
        st.subheader("Hasil prediksi", icon=":material/auto_awesome:")
        price_col, rupiah_col = st.columns([1.2, 1])
        price_col.metric(
            "Estimasi harga per bulan",
            f"{format_id_number(entry['harga_prediksi_ribu'], 2)} ribu",
            border=True,
            height="stretch",
        )
        rupiah_col.metric(
            "Setara dalam rupiah",
            format_rupiah(price_rupiah),
            border=True,
            height="stretch",
        )
        st.markdown("**Input yang digunakan**")
        input_cols = st.columns(4)
        input_cols[0].write(f"Jarak: **{format_id_number(entry['jarak_km'], 1)} km**")
        input_cols[1].write(f"Rating: **{format_id_number(entry['rating'], 1)}**")
        input_cols[2].write(f"Fasilitas AC: **{entry['ac']}**")
        input_cols[3].write(f"Algoritma: **{entry['algoritma']}**")
        st.caption(f"Waktu prediksi: {format_timestamp(entry['created_at'])}")


def render_dashboard(status: dict) -> None:
    with st.container(border=True, key="page-header"):
        st.caption("Selamat datang • Madiun")
        st.title("Prediksi Harga Kos Madiun", icon=":material/home:")
        st.write("Temukan estimasi harga kos berdasarkan jarak, rating, dan fasilitas.")
    dataset = load_dataset()
    try:
        summary = get_summary()
    except requests.RequestException as exc:
        summary = None
        st.warning(f"Ringkasan dari backend belum tersedia: {exc}")

    with st.grid(4, min_column_width=190, gap="small"):
        st.metric("Jumlah data kos", len(dataset), border=True, height="stretch")
        st.metric(
            "Rata-rata harga",
            format_id_number(dataset["harga"].mean(), 2),
            border=True,
            height="stretch",
        )
        st.metric(
            "Harga minimum",
            format_id_number(dataset["harga"].min(), 0),
            border=True,
            height="stretch",
        )
        st.metric(
            "Harga maksimum",
            format_id_number(dataset["harga"].max(), 0),
            border=True,
            height="stretch",
        )
    st.caption("Angka harga dalam ribu rupiah per bulan.")

    model_info = summary.get("model") if summary else None
    with st.grid(3, min_column_width=190, gap="small"):
        st.metric(
            "Algoritma",
            model_info["algorithm"] if model_info else "Belum tersedia",
            border=True,
            height="stretch",
        )
        if status.get("mode") == "direct":
            status_label, status_value = "Mode prediksi", "Langsung"
        elif status["reachable"]:
            status_label, status_value = "Status backend", "Terhubung"
        else:
            status_label, status_value = "Status backend", "Tidak terhubung"
        st.metric(
            status_label,
            status_value,
            border=True,
            height="stretch",
        )
        st.metric(
            "Status model",
            "Aktif" if status["model_available"] else "Belum tersedia",
            border=True,
            height="stretch",
        )

    st.subheader("Gambaran data aktual")
    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            st.markdown("#### Distribusi harga kos")
            price_distribution_data = price_distribution(dataset)
            st.bar_chart(
                price_distribution_data,
                width="stretch",
                height=300,
                alt="Distribusi jumlah kos pada setiap rentang harga",
            )
            show_diagram_explanation(
                title="Distribusi harga kos",
                description="Menunjukkan jumlah kos pada setiap rentang harga aktual.",
                x_label="Rentang harga",
                y_label="Jumlah kos",
                x_unit="ribu rupiah per bulan",
                y_unit="jumlah data",
                categories=[str(index) for index in price_distribution_data.index],
                data=[
                    {"rentang_harga": str(index), "jumlah_kos": int(row["Jumlah kos"])}
                    for index, row in price_distribution_data.iterrows()
                ],
                sample_count=len(dataset),
            )
    with right:
        with st.container(border=True):
            st.markdown("#### Rata-rata harga berdasarkan fasilitas AC")
            ac_price_data = average_price_by_ac(dataset)
            st.bar_chart(
                ac_price_data,
                width="stretch",
                height=300,
                alt="Perbandingan rata-rata harga kos dengan dan tanpa AC",
            )
            show_diagram_explanation(
                title="Rata-rata harga berdasarkan fasilitas AC",
                description="Membandingkan rata-rata harga kos dengan AC dan tanpa AC.",
                x_label="Fasilitas",
                y_label="Harga rata-rata",
                x_unit="kategori",
                y_unit="ribu rupiah per bulan",
                categories=["Dengan AC", "Tanpa AC"],
                data=[
                    {
                        "kategori": "Dengan AC",
                        "rata_rata_harga": (
                            round(float(ac_price_data.iloc[0]["Dengan AC"]), 2)
                            if pd.notna(ac_price_data.iloc[0]["Dengan AC"])
                            else None
                        ),
                    },
                    {
                        "kategori": "Tanpa AC",
                        "rata_rata_harga": (
                            round(float(ac_price_data.iloc[0]["Tanpa AC"]), 2)
                            if pd.notna(ac_price_data.iloc[0]["Tanpa AC"])
                            else None
                        ),
                    },
                ],
                sample_count=len(dataset),
            )

    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            st.markdown("#### Jarak ke kampus dan harga")
            st.altair_chart(
                build_relationship_chart(dataset, "jarak_km", "Jarak (km)", "harga", "Harga (ribu)"),
                width="stretch",
                alt="Hubungan jarak kos ke kampus dengan harga aktual",
            )
            show_diagram_explanation(
                title="Jarak ke kampus dan harga",
                description="Menampilkan titik data hubungan jarak kos ke kampus dengan harga aktual.",
                x_label="Jarak",
                y_label="Harga",
                x_unit="km",
                y_unit="ribu rupiah per bulan",
                categories=["Dengan AC", "Tanpa AC"],
                data=[
                    {
                        "jarak_km": round(float(row["jarak_km"]), 1),
                        "harga_ribu": round(float(row["harga"]), 2),
                        "fasilitas": "Dengan AC" if row["ac"] == "ya" else "Tanpa AC",
                    }
                    for _, row in dataset.iterrows()
                ],
                sample_count=len(dataset),
            )
    with right:
        with st.container(border=True):
            st.markdown("#### Rating dan harga")
            st.altair_chart(
                build_relationship_chart(dataset, "rating", "Rating", "harga", "Harga (ribu)"),
                width="stretch",
                alt="Hubungan rating kos dengan harga aktual",
            )
            show_diagram_explanation(
                title="Rating dan harga",
                description="Menampilkan titik data hubungan rating kos dengan harga aktual.",
                x_label="Rating",
                y_label="Harga",
                x_unit="skala rating",
                y_unit="ribu rupiah per bulan",
                categories=["Dengan AC", "Tanpa AC"],
                data=[
                    {
                        "rating": round(float(row["rating"]), 1),
                        "harga_ribu": round(float(row["harga"]), 2),
                        "fasilitas": "Dengan AC" if row["ac"] == "ya" else "Tanpa AC",
                    }
                    for _, row in dataset.iterrows()
                ],
                sample_count=len(dataset),
            )
def render_prediction() -> None:
    st.title("Prediksi Harga Kos", icon=":material/auto_awesome:")
    prediction_source = (
        "model Linear Regression langsung di Streamlit."
        if PREDICTION_MODE == "direct"
        else "model Django."
    )
    st.write(
        f"Masukkan karakteristik kos untuk mendapatkan estimasi dari "
        f"{prediction_source}"
    )
    with st.container(border=True, key="prediction-form"):
        st.subheader("Karakteristik kos", icon=":material/tune:")
        input_cols = st.columns(2)
        jarak = input_cols[0].slider(
            "Jarak ke kampus (km)", 0.6, 5.0, 1.5, 0.1, key="prediction_jarak"
        )
        rating = input_cols[1].slider(
            "Rating kos", 3.1, 4.9, 4.0, 0.1, key="prediction_rating"
        )
        ac = st.selectbox("Fasilitas AC", ["ya", "tidak"], key="prediction_ac")
        submitted = st.button(
            "Prediksi Harga Kos",
            type="primary",
            key="run_prediction",
            width="stretch",
        )

    if submitted:
        if PREDICTION_MODE == "direct":
            try:
                prediction = predict_direct(jarak, rating, ac)
                st.session_state.latest_prediction = prediction
                st.session_state.latest_prediction_rupiah = prediction[
                    "harga_prediksi_rupiah"
                ]
                st.success("Prediksi berhasil.")
            except (KeyError, TypeError, ValueError) as exc:
                st.error(f"Prediksi tidak berhasil: {exc}")
        else:
            try:
                if API_URL is None:
                    raise RuntimeError("Django API tidak tersedia dalam mode ini.")
                response = requests.post(
                    API_URL + "predict/",
                    json={"jarak_km": jarak, "rating": rating, "ac": ac},
                    timeout=5,
                )
                result = response.json()
                if response.ok and result.get("success"):
                    prediction = result.get("data", result)
                    st.session_state.latest_prediction = prediction
                    st.session_state.latest_prediction_rupiah = int(
                        prediction.get(
                            "harga_prediksi_rupiah",
                            round(prediction["harga_prediksi_ribu"] * 1000),
                        )
                    )
                    st.success("Prediksi berhasil.")
                else:
                    errors = result.get(
                        "errors", result.get("detail", "Input atau model tidak valid.")
                    )
                    st.error(f"Prediksi tidak berhasil: {errors}")
            except requests.RequestException as exc:
                st.error(f"Tidak dapat menghubungi backend Django: {exc}")
            except (KeyError, TypeError, ValueError) as exc:
                st.error(f"Respons backend tidak sesuai format: {exc}")

    latest = st.session_state.get("latest_prediction")
    if latest:
        show_prediction(
            latest,
            st.session_state.get("latest_prediction_rupiah"),
        )
    st.info(PREDICTION_CAVEAT)


def render_dataset() -> None:
    st.title("Dataset Kos Madiun", icon=":material/dataset:")
    st.write("Data asli tidak diubah; filter hanya berlaku untuk tampilan saat ini.")
    dataset = load_dataset()
    with st.grid(2, min_column_width=240, gap="small"):
        search = st.text_input(
            "Cari nama kos",
            key="dataset_search",
            type="search",
            placeholder="Masukkan nama kos",
        ).strip().casefold()
        ac_filter = st.selectbox(
            "Filter fasilitas AC",
            ["Semua", "ya", "tidak"],
            key="dataset_ac_filter",
        )
    with st.grid(3, min_column_width=190, gap="small"):
        price_range = st.slider(
            "Rentang harga (ribu)",
            int(dataset["harga"].min()),
            int(dataset["harga"].max()),
            (int(dataset["harga"].min()), int(dataset["harga"].max())),
            step=50,
            key="dataset_price_filter",
        )
        distance_range = st.slider(
            "Rentang jarak (km)",
            float(dataset["jarak_km"].min()),
            float(dataset["jarak_km"].max()),
            (float(dataset["jarak_km"].min()), float(dataset["jarak_km"].max())),
            step=0.1,
            key="dataset_distance_filter",
        )
        rating_range = st.slider(
            "Rentang rating",
            float(dataset["rating"].min()),
            float(dataset["rating"].max()),
            (float(dataset["rating"].min()), float(dataset["rating"].max())),
            step=0.1,
            key="dataset_rating_filter",
        )

    filtered = dataset[
        dataset["nama"].str.casefold().str.contains(search, regex=False)
        & dataset["harga"].between(*price_range)
        & dataset["jarak_km"].between(*distance_range)
        & dataset["rating"].between(*rating_range)
    ]
    if ac_filter != "Semua":
        filtered = filtered[filtered["ac"] == ac_filter]

    with st.grid(4, min_column_width=190, gap="small"):
        st.metric("Kos sesuai filter", len(filtered), border=True, height="stretch")
        st.metric(
            "Rata-rata harga",
            format_id_number(filtered["harga"].mean(), 2)
            if not filtered.empty
            else "—",
            border=True,
            height="stretch",
        )
        st.metric(
            "Rata-rata jarak",
            f"{format_id_number(filtered['jarak_km'].mean(), 2)} km"
            if not filtered.empty
            else "—",
            border=True,
            height="stretch",
        )
        st.metric(
            "Rata-rata rating",
            format_id_number(filtered["rating"].mean(), 2) if not filtered.empty else "—",
            border=True,
            height="stretch",
        )
    st.caption("Angka harga dalam ribu rupiah per bulan.")

    if filtered.empty:
        st.info("Tidak ada data yang cocok dengan pencarian dan filter.")
        return

    st.dataframe(
        filtered[["nama", "jarak_km", "harga", "ac", "rating"]],
        column_config={
            "nama": st.column_config.TextColumn("Nama kos", width="large"),
            "jarak_km": st.column_config.NumberColumn(
                "Jarak ke kampus (km)", format="%.1f", width="small"
            ),
            "harga": st.column_config.NumberColumn(
                "Harga (ribu Rp/bulan)", format="%,.0f", width="medium"
            ),
            "ac": st.column_config.TextColumn("AC", width="small"),
            "rating": st.column_config.NumberColumn(
                "Rating", format="%.1f", width="small"
            ),
        },
        width="stretch",
        height=420,
        row_height=42,
        hide_index=True,
        alt="Tabel dataset kos yang sudah difilter",
    )
    chart_col, comparison_col = st.columns(2)
    with chart_col:
        with st.container(border=True):
            st.markdown("#### Distribusi harga pada hasil filter")
            filtered_distribution = price_distribution(filtered)
            st.bar_chart(
                filtered_distribution,
                width="stretch",
                height=320,
                alt="Distribusi harga aktual pada data yang difilter",
            )
            show_diagram_explanation(
                title="Distribusi harga pada hasil filter",
                description="Menunjukkan jumlah kos pada setiap rentang harga setelah filter diterapkan.",
                x_label="Rentang harga",
                y_label="Jumlah kos",
                x_unit="ribu rupiah per bulan",
                y_unit="jumlah data",
                categories=[str(index) for index in filtered_distribution.index],
                data=[
                    {"rentang_harga": str(index), "jumlah_kos": int(row["Jumlah kos"])}
                    for index, row in filtered_distribution.iterrows()
                ],
                sample_count=len(filtered),
            )
    with comparison_col:
        with st.container(border=True):
            st.markdown("#### Perbandingan harga berdasarkan AC")
            filtered_ac_price = average_price_by_ac(filtered)
            st.bar_chart(
                filtered_ac_price,
                width="stretch",
                height=320,
                alt="Rata-rata harga pada hasil filter berdasarkan fasilitas AC",
            )
            show_diagram_explanation(
                title="Perbandingan harga berdasarkan AC pada hasil filter",
                description="Membandingkan rata-rata harga kos dengan dan tanpa AC pada data yang difilter.",
                x_label="Fasilitas",
                y_label="Harga rata-rata",
                x_unit="kategori",
                y_unit="ribu rupiah per bulan",
                categories=["Dengan AC", "Tanpa AC"],
                data=[
                    {
                        "kategori": "Dengan AC",
                        "rata_rata_harga": (
                            round(float(filtered_ac_price.iloc[0]["Dengan AC"]), 2)
                            if pd.notna(filtered_ac_price.iloc[0]["Dengan AC"])
                            else None
                        ),
                    },
                    {
                        "kategori": "Tanpa AC",
                        "rata_rata_harga": (
                            round(float(filtered_ac_price.iloc[0]["Tanpa AC"]), 2)
                            if pd.notna(filtered_ac_price.iloc[0]["Tanpa AC"])
                            else None
                        ),
                    },
                ],
                sample_count=len(filtered),
            )
    st.markdown("**Ringkasan statistik hasil filter**")
    st.dataframe(
        filtered[["jarak_km", "harga", "rating"]].describe().round(2),
        width="stretch",
        alt="Statistik deskriptif dataset yang difilter",
    )


def render_evaluation() -> None:
    st.title("Evaluasi Model", icon=":material/analytics:")
    st.write(
        "Metrik regresi dihitung dari hasil evaluasi tercatat, bukan dari prediksi yang dibuat saat ini."
    )
    try:
        summary = get_summary()
    except requests.RequestException as exc:
        st.error(f"Evaluasi tidak dapat dimuat dari backend: {exc}")
        return

    model_info = summary.get("model")
    evaluation = summary.get("evaluation")
    if not model_info:
        st.warning("Model Joblib belum tersedia.")
        return
    if not evaluation or not evaluation.get("available"):
        detail = evaluation.get("detail") if evaluation else "Laporan evaluasi tidak tersedia."
        st.warning(f"Metrik evaluasi tidak ditampilkan: {detail}")
        return

    holdout = evaluation["holdout"]
    linear = holdout["linear_regression"]
    dummy = holdout["dummy_regressor_mean"]
    loo_linear = evaluation["leave_one_out"]["linear_regression"]
    loo_dummy = evaluation["leave_one_out"]["dummy_regressor_mean"]

    st.success("Algoritma utama: Linear Regression. Semua angka harga menggunakan ribu rupiah per bulan.")
    st.subheader("Holdout 80:20 — model evaluator")
    st.caption(
        f"Evaluator dilatih dengan {holdout['training_rows']} baris dan diuji pada "
        f"{holdout['test_rows']} baris (random_state={holdout['random_state']})."
    )
    with st.grid(4, min_column_width=190, gap="small"):
        st.metric("MAE (ribu Rp)", f"{linear['mae']:,.3f}", border=True, height="stretch")
        st.metric("MSE (ribu Rp)²", f"{linear['mse']:,.3f}", border=True, height="stretch")
        st.metric("RMSE (ribu Rp)", f"{linear['rmse']:,.3f}", border=True, height="stretch")
        st.metric("R² test", f"{linear['r2']:.3f}", border=True, height="stretch")
        st.metric(
            "R² training evaluator",
            f"{holdout['training_r2']:.3f}",
            border=True,
            height="stretch",
        )
    st.warning(
        "Holdout hanya berisi enam data uji, sehingga hasil sangat sensitif terhadap pembagian data."
    )

    st.subheader("Perbandingan dengan DummyRegressor")
    st.write(
        "Pada pembagian yang sama, Linear Regression belum mengungguli baseline yang hanya "
        "memprediksi rata-rata harga training."
    )
    with st.grid(4, min_column_width=190, gap="small"):
        st.metric("Dummy MAE", f"{dummy['mae']:,.2f}", border=True, height="stretch")
        st.metric("Dummy RMSE", f"{dummy['rmse']:,.2f}", border=True, height="stretch")
        st.metric("Dummy R²", f"{dummy['r2']:.3f}", border=True, height="stretch")
        st.metric(
            "Training evaluator",
            f"{holdout['training_rows']} data",
            border=True,
            height="stretch",
        )

    st.subheader("Cross-validation tambahan — Leave-One-Out")
    st.caption(
        f"{evaluation['leave_one_out']['folds']} fold; R² dihitung dari seluruh prediksi "
        "out-of-fold yang digabungkan. LOO tidak menggantikan holdout dan tidak digunakan "
        "untuk tuning model."
    )
    with st.grid(4, min_column_width=190, gap="small"):
        st.metric("LOO Linear MAE", f"{loo_linear['mae']:,.2f}", border=True, height="stretch")
        st.metric("LOO Dummy MAE", f"{loo_dummy['mae']:,.2f}", border=True, height="stretch")
        st.metric("LOO Linear RMSE", f"{loo_linear['rmse']:,.2f}", border=True, height="stretch")
        st.metric("LOO Dummy RMSE", f"{loo_dummy['rmse']:,.2f}", border=True, height="stretch")

    comparison = pd.DataFrame(
        {
            "Linear Regression": [
                linear["mae"],
                linear["rmse"],
                loo_linear["mae"],
                loo_linear["rmse"],
            ],
            "DummyRegressor": [
                dummy["mae"],
                dummy["rmse"],
                loo_dummy["mae"],
                loo_dummy["rmse"],
            ],
        },
        index=["Holdout MAE", "Holdout RMSE", "LOO MAE", "LOO RMSE"],
    )
    st.subheader("Grafik MAE dan RMSE")
    st.bar_chart(
        comparison,
        horizontal=True,
        stack=None,
        width="stretch",
        height=360,
        alt="Perbandingan MAE dan RMSE Linear Regression dengan DummyRegressor",
    )
    show_diagram_explanation(
        title="Grafik MAE dan RMSE",
        description="Membandingkan kesalahan MAE dan RMSE Linear Regression dengan DummyRegressor pada holdout dan leave-one-out.",
        x_label="Nilai metrik",
        y_label="Skenario evaluasi",
        x_unit="ribu rupiah per bulan",
        y_unit="metrik",
        categories=["Holdout MAE", "Holdout RMSE", "LOO MAE", "LOO RMSE"],
        data=[
            {
                "skenario": index,
                "linear_regression": round(float(row["Linear Regression"]), 3),
                "dummy_regressor": round(float(row["DummyRegressor"]), 3),
            }
            for index, row in comparison.iterrows()
        ],
        sample_count=holdout["training_rows"] + holdout["test_rows"],
        additional_metrics={
            "holdout_training_rows": holdout["training_rows"],
            "holdout_test_rows": holdout["test_rows"],
            "linear_mae": linear["mae"],
            "linear_mse": linear["mse"],
            "linear_rmse": linear["rmse"],
            "linear_r2_test": linear["r2"],
            "linear_r2_training": holdout["training_r2"],
            "dummy_mae": dummy["mae"],
            "dummy_rmse": dummy["rmse"],
            "dummy_r2": dummy["r2"],
            "loo_folds": evaluation["leave_one_out"]["folds"],
            "loo_linear_mae": loo_linear["mae"],
            "loo_linear_rmse": loo_linear["rmse"],
            "loo_dummy_mae": loo_dummy["mae"],
            "loo_dummy_rmse": loo_dummy["rmse"],
        },
    )

    st.subheader("Model evaluator dan model produksi")
    st.write(
        f"Model evaluator di atas hanya dilatih dengan {holdout['training_rows']} data training. "
        f"Model produksi pada Joblib dilatih kembali menggunakan seluruh {model_info['training_rows']} "
        "data agar dapat melayani prediksi. Keduanya adalah fit berbeda; metrik holdout bukan "
        "skor training model produksi."
    )
    st.caption(f"Sumber evaluasi: {evaluation['source']}.")
    st.caption(
        "Metrik mengukur besar kesalahan regresi; angka ini bukan persentase akurasi."
    )


def render_about() -> None:
    st.title("Tentang Aplikasi", icon=":material/info:")
    st.write("Ringkasan tujuan, teknologi, dan alur estimasi harga kos Madiun.")
    workflow_description = (
        "Dalam mode direct, Streamlit melatih Linear Regression dari dataset dan "
        "memprediksi tanpa server Django."
        if PREDICTION_MODE == "direct"
        else "Jarak, rating, dan ketersediaan AC dikirim ke Django REST API. "
        "Model memproses masukan tersebut dan mengembalikan estimasi harga."
    )
    with st.grid(2, min_column_width=260, gap="small"):
        with st.container(border=True, height="stretch"):
            st.subheader("Tujuan aplikasi", icon=":material/home_work:")
            st.write(
                "Aplikasi memperkirakan harga kos di Madiun menggunakan "
                "Linear Regression."
            )
        with st.container(border=True, height="stretch"):
            st.subheader("Teknologi", icon=":material/developer_board:")
            st.markdown(
                "- **Frontend:** Streamlit\n"
                "- **Backend dan API:** Django REST Framework\n"
                "- **Machine Learning:** Scikit-learn Linear Regression\n"
                "- **Penyimpanan model:** Joblib"
            )
        with st.container(border=True, height="stretch"):
            st.subheader("Dataset", icon=":material/dataset:")
            st.write("Dataset kos Madiun berisi 30 baris.")
            st.caption("Harga dan hasil estimasi menggunakan ribu rupiah per bulan.")
        with st.container(border=True, height="stretch"):
            st.subheader("Cara kerja", icon=":material/route:")
            st.write(workflow_description)
    st.info(
        "Hasil prediksi merupakan estimasi dan tidak menjamin harga pasar aktual."
    )
    st.warning("Aplikasi tidak menggunakan database.")


if PREDICTION_MODE == "direct":
    direct_artifact = load_direct_model()
    status = {
        "mode": "direct",
        "reachable": True,
        "model_available": True,
        "algorithm": direct_artifact["algorithm"],
        "training_rows": direct_artifact["training_rows"],
    }
else:
    status = {"mode": "django", **backend_status()}
with st.sidebar:
    st.subheader("Prediksi Harga Kos Madiun", icon=":material/home:")
    st.caption("Dashboard estimasi harga hunian")
    st.caption("MENU UTAMA")
    page = st.radio(
        "Navigasi",
        PAGES,
        label_visibility="collapsed",
        key="navigation",
    )
    show_backend_status(status)

if page == PAGES[0]:
    render_dashboard(status)
elif page == PAGES[1]:
    render_prediction()
elif page == PAGES[2]:
    render_dataset()
elif page == PAGES[3]:
    render_evaluation()
else:
    render_about()
