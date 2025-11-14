# app_streamlit.py
# RYE Analyzer - rich, mobile-friendly Streamlit app with:
# Multi-format ingest • Presets & tooltips • Auto column detect • Auto rolling window
# Single / Compare • Multi-domain • Reports tabs • Diagnostics & PDF reporting

from __future__ import annotations
import io, json, os, sys, traceback, importlib.util, importlib.machinery
from typing import Optional, Dict, Any

import gzip
import zipfile

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ------------- Make sure we can see core.py regardless of folder layout -------------
ROOT = os.path.dirname(os.path.abspath(__file__))
CANDIDATE_PATHS = [
    ROOT,
    os.path.join(ROOT, "src"),
    os.path.join(ROOT, "app"),
    os.path.join(ROOT, "rye_analyzer"),
    os.path.join(ROOT, "rye-analyzer"),
]
for p in CANDIDATE_PATHS:
    if p not in sys.path and os.path.isdir(p):
        sys.path.insert(0, p)


def _try_import_core():
    # 1) normal import
    try:
        import core  # type: ignore
        return core
    except Exception:
        pass
    # 2) package-style (if running inside a pkg)
    try:
        from . import core  # type: ignore
        return core
    except Exception:
        pass
    # 3) direct load from file if it exists nearby
    for base in CANDIDATE_PATHS:
        fp = os.path.join(base, "core.py")
        if os.path.isfile(fp):
            try:
                loader = importlib.machinery.SourceFileLoader("core", fp)
                mod = importlib.util.module_from_spec(
                    importlib.util.spec_from_loader("core", loader)
                )
                loader.exec_module(mod)  # type: ignore
                sys.modules["core"] = mod
                return mod
            except Exception:
                pass
    return None


_core_mod = _try_import_core()
if _core_mod is None:
    st.set_page_config(page_title="RYE Analyzer", page_icon="📈", layout="wide")
    st.error(
        "Could not import core.py. Ensure a file named core.py is in the same folder "
        "as this app or in one of: src/, app/, rye_analyzer/, rye-analyzer/."
    )
    st.stop()

# ---------------- Try to import PRESETS ----------------
PRESETS = None
preset_import_error: Optional[str] = None
try:
    from presets import PRESETS as _PRESETS  # preferred location
    PRESETS = _PRESETS
except Exception as e1:
    try:
        PRESETS = _core_mod.PRESETS  # fallback if kept in core.py
    except Exception as e2:
        preset_import_error = (
            "Could not import PRESETS (presets.py/core.py). Using a tiny default. "
            f"{e1.__class__.__name__}: {e1} / {e2.__class__.__name__}: {e2}"
        )
        PRESETS = {
            "Generic": type(
                "Preset",
                (),
                {
                    "name": "Generic",
                    "time": ["time"],
                    "performance": ["performance"],
                    "energy": ["energy"],
                    "domain": "domain",
                    "default_rolling": 10,
                    "tooltips": {
                        "Generic": "Basic preset used when presets are not available."
                    },
                },
            )()
        }

# Column alias map (if available) so we can be smart about domain fallbacks
COLUMN_ALIASES = getattr(_core_mod, "COLUMN_ALIASES", {})
DOMAIN_ALIASES = getattr(_core_mod, "DOMAIN_ALIASES", [])

# ---------------- Import core helpers (with graceful fallbacks) ----------------
load_table = _core_mod.load_table
normalize_columns = _core_mod.normalize_columns
safe_float = _core_mod.safe_float

# compute_rye_from_df may be named differently in older cores; alias whichever exists
if hasattr(_core_mod, "compute_rye_from_df"):
    _compute_rye_from_df = _core_mod.compute_rye_from_df
elif hasattr(_core_mod, "compute_rye"):
    _compute_rye_from_df = _core_mod.compute_rye
else:
    st.set_page_config(page_title="RYE Analyzer", page_icon="📈", layout="wide")
    st.error("Neither compute_rye_from_df nor compute_rye found in core.py; add one of them.")
    st.stop()

# summarize function may be named summarize_series or summarize
if hasattr(_core_mod, "summarize_series"):
    _summarize_series = _core_mod.summarize_series
elif hasattr(_core_mod, "summarize"):
    _summarize_series = _core_mod.summarize
else:
    st.set_page_config(page_title="RYE Analyzer", page_icon="📈", layout="wide")
    st.error("Neither summarize_series nor summarize found in core.py; add one of them.")
    st.stop()

# rolling_series is optional; provide a pandas fallback if missing
if hasattr(_core_mod, "rolling_series"):
    _rolling_series = _core_mod.rolling_series
else:

    def _rolling_series(arr, window: int):
        s = pd.Series(arr, dtype=float)
        if window <= 1:
            return s.fillna(0.0).values
        return s.rolling(window=window, min_periods=1).mean().values


# optional column inference
_infer_columns = getattr(_core_mod, "infer_columns", None)

# Optional advanced analytics (each may be absent)
detect_regimes = getattr(_core_mod, "detect_regimes", None)
energy_delta_performance_correlation = getattr(
    _core_mod, "energy_delta_performance_correlation", None
)
estimate_noise_floor = getattr(_core_mod, "estimate_noise_floor", None)
bootstrap_rolling_mean = getattr(_core_mod, "bootstrap_rolling_mean", None)

# ---------------- Translation helpers ----------------
language = "English"  # default; will be updated from sidebar later


def tr(en: str, es: str) -> str:
    """Simple inline translator for English / Spanish UI text."""
    return es if language == "Español" else en


# ---------------- Local helpers (work even if core lacks them) ----------------
def ema_series(x, span: int) -> np.ndarray:
    if span is None or span <= 1:
        return np.asarray(x, dtype=float)
    s = pd.Series(x, dtype=float)
    return s.ewm(span=span, adjust=False).mean().values


def cumulative_series(x) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    a[~np.isfinite(a)] = 0.0
    return np.cumsum(a)


def smart_window(n_rows: int, preset_default: Optional[int]) -> int:
    if preset_default and preset_default > 0:
        return int(preset_default)
    if n_rows <= 0:
        return 10
    # about 5 percent of series length, clipped [3, 200]
    guess = max(3, min(200, int(round(max(3, n_rows * 0.05)))))
    return guess


# ---------------- PDF builder (optional) + diagnostics ----------------
def _probe_fpdf_version() -> str:
    try:
        import fpdf

        ver = getattr(fpdf, "__version__", "installed (version unknown)")
        return f"fpdf module found: {ver}"
    except Exception as e:
        return f"fpdf import failed: {e.__class__.__name__}: {e}"


build_pdf = None
_pdf_import_error = None
try:
    spec = importlib.util.find_spec("report")
    if spec is None:
        _pdf_import_error = "report.py not found in the working directory."
    else:
        from report import build_pdf  # noqa
except Exception:
    _pdf_import_error = traceback.format_exc()

# ---------------- UI helpers ----------------
def add_stability_bands(fig: go.Figure, y_max_hint: Optional[float] = None):
    top = y_max_hint if y_max_hint is not None else 1.0
    fig.add_hrect(y0=0.6, y1=max(0.6, top), fillcolor="green", opacity=0.08, line_width=0)
    fig.add_hrect(y0=0.3, y1=0.6, fillcolor="yellow", opacity=0.08, line_width=0)
    fig.add_hrect(
        y0=min(-0.5, 0.3 - 1.0), y1=0.3, fillcolor="red", opacity=0.08, line_width=0
    )


def _delta_performance(series):
    arr = pd.Series(series, dtype=float)
    return arr.diff().fillna(0.0).values


def _first_or(default, lst):
    return lst[0] if isinstance(lst, list) and lst else default


_HAS_POPOVER = hasattr(st, "popover")

# ---------------- Page config ----------------
st.set_page_config(page_title="RYE Analyzer", page_icon="📈", layout="wide")
st.title(tr("RYE Analyzer", "Analizador RYE"))

with st.expander(tr("What is RYE?", "¿Qué es RYE?")):
    st.write(
        tr(
            "Repair Yield per Energy (RYE) measures how efficiently a system converts effort or energy "
            "into successful repair or performance gains. Higher RYE means better efficiency.",
            "El Rendimiento de Reparación por Energía (RYE) mide qué tan eficientemente un sistema "
            "convierte esfuerzo o energía en reparación exitosa o mejora de desempeño. Un RYE más alto "
            "significa mejor eficiencia.",
        )
    )

# ---------------- Seed session_state BEFORE widgets ----------------
_default_preset_name = list(PRESETS.keys())[0]
_default_preset = PRESETS.get(_default_preset_name, next(iter(PRESETS.values())))

if "defaults_initialized" not in st.session_state:
    st.session_state["defaults_initialized"] = True
    st.session_state["default_col_time"] = _first_or(
        "time", getattr(_default_preset, "time", ["time"])
    )
    st.session_state["default_col_domain"] = getattr(
        _default_preset, "domain", "domain"
    ) or "domain"
    st.session_state["default_col_repair"] = _first_or(
        "performance", getattr(_default_preset, "performance", ["performance"])
    )
    st.session_state["default_col_energy"] = _first_or(
        "energy", getattr(_default_preset, "energy", ["energy"])
    )

# ---------------- Sidebar (inputs) ----------------
with st.sidebar:
    # Language selector first so everything else can react
    language = st.selectbox(
        "Language / Idioma",
        ["English", "Español"],
        index=0,
    )

    st.header(tr("Inputs", "Entradas"))

    preset_name = st.selectbox(tr("Preset", "Preajuste"), list(PRESETS.keys()), index=0)
    preset = PRESETS.get(preset_name, next(iter(PRESETS.values())))
    marketing_mode = preset_name.lower().startswith("marketing")

    ttips = getattr(preset, "tooltips", None) or {}
    if isinstance(ttips, dict) and ttips:
        if _HAS_POPOVER:
            with st.popover(tr("Preset tips", "Notas del preajuste"), use_container_width=True):
                for k, v in ttips.items():
                    st.markdown(f"**{k}**: {v}")
        else:
            with st.expander(tr("Preset tips", "Notas del preajuste")):
                for k, v in ttips.items():
                    st.markdown(f"**{k}**: {v}")

    if marketing_mode:
        st.info(
            tr(
                "Marketing preset: treat your performance column as outcomes "
                "(conversions, revenue, ROAS, retention) and your energy column as cost "
                "or budget (spend, impressions, touches). RYE then estimates outcome per "
                "unit of spend.",
                "Preajuste de marketing: la columna de desempeño se interpreta como resultados "
                "(conversiones, ingresos, ROAS, retención) y la columna de energía como costo o "
                "presupuesto (gasto, impresiones, contactos). RYE estima el resultado por unidad "
                "de gasto.",
            )
        )

    st.write(
        tr(
            "Upload one file to analyze. Optionally upload a second file to compare.",
            "Sube un archivo para analizar. Opcionalmente puedes subir un segundo archivo para comparar.",
        )
    )

    # Accept any file type; load_any will decide how to parse or fail gracefully
    file1 = st.file_uploader(tr("Primary file", "Archivo principal"), type=None, key="file1")
    file2 = st.file_uploader(
        tr("Comparison file (optional)", "Archivo de comparación (opcional)"),
        type=None,
        key="file2",
    )

    # Preview dataframe used only for column inference (read once here)
    df_preview: Optional[pd.DataFrame] = None
    if file1 is not None:
        try:
            df_preview = load_table(file1)
            df_preview = normalize_columns(df_preview)
        except Exception:
            df_preview = None
        try:
            file1.seek(0)
        except Exception:
            pass

    # Automatic one-time column inference after upload
    if (
        df_preview is not None
        and _infer_columns is not None
        and not st.session_state.get("auto_columns_applied", False)
    ):
        try:
            guess = _infer_columns(df_preview, preset_name=preset_name)
            if guess.get("time"):
                st.session_state["col_time"] = guess["time"]
            if guess.get("domain"):
                st.session_state["col_domain"] = guess["domain"]
            if guess.get("performance"):
                st.session_state["col_repair"] = guess["performance"]
            if guess.get("energy"):
                st.session_state["col_energy"] = guess["energy"]
            st.session_state["auto_columns_applied"] = True
        except Exception:
            pass

    st.divider()
    st.write(tr("Column names in your data", "Nombres de columnas en tus datos"))

    # Manual auto-detect button (safe, runs before text_input widgets)
    if st.button(tr("Auto-detect columns from data", "Detectar columnas automáticamente")):
        if _infer_columns is None:
            st.warning(
                tr(
                    "Column inference not available (core.infer_columns missing).",
                    "La inferencia de columnas no está disponible (falta core.infer_columns).",
                )
            )
        elif df_preview is None:
            st.warning(
                tr(
                    "Upload a primary file first.",
                    "Primero sube un archivo principal.",
                )
            )
        else:
            try:
                guess = _infer_columns(df_preview, preset_name=preset_name)
                st.success(tr(f"Detected: {guess}", f"Detectado: {guess}"))
                if guess.get("time"):
                    st.session_state["col_time"] = guess["time"]
                if guess.get("domain"):
                    st.session_state["col_domain"] = guess["domain"]
                if guess.get("performance"):
                    st.session_state["col_repair"] = guess["performance"]
                if guess.get("energy"):
                    st.session_state["col_energy"] = guess["energy"]
            except Exception as e:
                st.error(tr(f"Auto-detect failed: {e}", f"La detección falló: {e}"))
                st.code(traceback.format_exc(), language="text")

    # Now create the widgets that read from session_state
    col_time = st.text_input(
        tr("Time column (optional)", "Columna de tiempo (opcional)"),
        value=st.session_state.get(
            "col_time", st.session_state.get("default_col_time", "time")
        ),
        key="col_time",
    )
    col_domain = st.text_input(
        tr("Domain column (optional)", "Columna de dominio (opcional)"),
        value=st.session_state.get(
            "col_domain", st.session_state.get("default_col_domain", "domain")
        ),
        key="col_domain",
    )
    col_repair = st.text_input(
        tr("Performance/Repair column", "Columna de desempeño/reparación"),
        value=st.session_state.get(
            "col_repair", st.session_state.get("default_col_repair", "performance")
        ),
        key="col_repair",
    )
    col_energy = st.text_input(
        tr("Energy/Effort column", "Columna de energía/esfuerzo"),
        value=st.session_state.get(
            "col_energy", st.session_state.get("default_col_energy", "energy")
        ),
        key="col_energy",
    )

    st.divider()
    default_window = int(getattr(preset, "default_rolling", 10) or 10)
    auto_roll = st.checkbox(
        tr("Auto rolling window", "Ventana móvil automática"),
        value=True,
        help=tr(
            "Use preset default or smart guess by series length.",
            "Usa el valor por defecto del preajuste o una estimación según la longitud de la serie.",
        ),
    )
    window = st.number_input(
        tr("Rolling window", "Ventana móvil"),
        min_value=1,
        max_value=1000,
        value=default_window,
        step=1,
        help=tr(
            "Moving average length applied to the RYE series.",
            "Longitud del promedio móvil aplicado a la serie de RYE.",
        ),
        disabled=auto_roll,
    )

    ema_span = st.number_input(
        tr("EMA smoothing (optional)", "Suavizado EMA (opcional)"),
        min_value=0,
        max_value=1000,
        value=0,
        step=1,
        help=tr("Extra smoothing; 0 disables EMA.", "Suavizado adicional; 0 desactiva la EMA."),
    )
    sim_factor = st.slider(
        tr("Multiply energy by", "Multiplicar energía por"),
        min_value=0.10,
        max_value=3.0,
        value=1.0,
        step=0.05,
        help=tr(
            "What-if: scale energy before computing RYE.",
            "Escenario hipotético: escala la energía antes de calcular RYE.",
        ),
    )

    st.divider()
    doi_or_link = st.text_input(
        tr("Zenodo DOI or dataset link (optional)", "DOI de Zenodo o enlace al conjunto de datos (opcional)"),
        value="",
        help="Example: 10.5281/zenodo.123456 or a dataset URL",
    )

    st.divider()
    if preset_import_error:
        st.info(preset_import_error)
    if st.button(tr("Download example CSV", "Descargar CSV de ejemplo")):
        example = pd.DataFrame(
            {
                "time": np.arange(0, 15),
                "domain": ["AI"] * 5 + ["Bio"] * 5 + ["Robotics"] * 5,
                "performance": [
                    0,
                    0,
                    0.1,
                    0.2,
                    0.35,
                    0.38,
                    0.5,
                    0.46,
                    0.52,
                    0.6,
                    0.6,
                    0.6,
                    0.62,
                    0.62,
                    0.65,
                ],
                "energy": [
                    1,
                    1,
                    1,
                    1,
                    1,
                    1.1,
                    1.0,
                    1.02,
                    1.05,
                    1.1,
                    1.1,
                    1.12,
                    1.09,
                    1.1,
                    1.1,
                ],
            }
        )
        b = example.to_csv(index=False).encode("utf-8")
        st.download_button(
            tr("Save example.csv", "Guardar example.csv"),
            b,
            file_name="example.csv",
            mime="text/csv",
        )

# ---------------- Core workers ----------------
def _decompress_gzip_to_bytes_io(file, original_name: str) -> io.BytesIO:
    """Decompress a gzip UploadedFile or buffer into a BytesIO and preserve an inner name."""
    try:
        with gzip.GzipFile(fileobj=file) as gz:
            data = gz.read()
        inner = io.BytesIO(data)
        # strip .gz if present so downstream logic sees the real extension
        if original_name.lower().endswith(".gz"):
            inner.name = original_name[:-3]  # type: ignore[attr-defined]
        else:
            inner.name = original_name  # type: ignore[attr-defined]
        inner.seek(0)
        return inner
    except Exception as e:
        st.error(
            tr(
                f"Could not decompress gzip file: {e}",
                f"No se pudo descomprimir el archivo gzip: {e}",
            )
        )
        raise


def _extract_from_zip_to_bytes_io(file, original_name: str) -> Optional[io.BytesIO]:
    """
    Extract a useful inner file from a zip.
    Priority:
      1) occurrence.txt (DwC-A)
      2) event.txt, emof.txt
      3) any csv, tsv, txt
      4) any nc, netcdf, json
    Returns BytesIO or None.
    """
    try:
        with zipfile.ZipFile(file) as zf:
            names = zf.namelist()
            target = None

            preferred = [
                "occurrence.txt",
                "event.txt",
                "emof.txt",
            ]
            lower_map = {n.lower(): n for n in names}

            for cand in preferred:
                if cand in lower_map:
                    target = lower_map[cand]
                    break

            if target is None:
                for ext in (".csv", ".tsv", ".txt"):
                    matches = [n for n in names if n.lower().endswith(ext)]
                    if matches:
                        target = matches[0]
                        break

            if target is None:
                for ext in (".nc", ".netcdf", ".json"):
                    matches = [n for n in names if n.lower().endswith(ext)]
                    if matches:
                        target = matches[0]
                        break

            if target is None:
                st.error(
                    tr(
                        "Zip archive found but no tabular or NetCDF file could be identified.",
                        "Se encontró un archivo zip pero no se identificó ningún archivo tabular o NetCDF.",
                    )
                )
                return None

            with zf.open(target) as inner_file:
                data = inner_file.read()

        inner = io.BytesIO(data)
        inner.name = target  # type: ignore[attr-defined]
        inner.seek(0)
        return inner
    except Exception as e:
        st.error(
            tr(
                f"Could not read zip archive: {e}",
                f"No se pudo leer el archivo zip: {e}",
            )
        )
        st.code(traceback.format_exc(), language="text")
        return None


def load_any(file) -> Optional[pd.DataFrame]:
    """
    Safe loader for user files:
    - Handles Excel, CSV, TSV, TXT
    - Handles gzip compressed files (.gz) by decompressing then delegating
    - Handles zip archives (.zip), with DwC-A or a single CSV/TSV/TXT/NetCDF inside
    - Attempts XML to table when possible
    - Delegates complex or remaining formats to core.load_table
    - Rejects very large uploads (> 200 MB) to avoid exhausting memory
    - Caps extremely large tables to 1,000,000 rows to avoid crashes
    """
    if file is None:
        return None

    # Basic size guard (works with Streamlit UploadedFile objects)
    max_size_mb = 200
    try:
        if hasattr(file, "size") and file.size and file.size > max_size_mb * 1024 * 1024:
            st.error(
                tr(
                    f"File too large (>{max_size_mb} MB). Please upload a smaller file.",
                    f"Archivo demasiado grande (>{max_size_mb} MB). Sube un archivo más pequeño.",
                )
            )
            return None
    except Exception:
        pass

    filename = getattr(file, "name", "")
    name_lower = filename.lower() if filename else ""

    # Handle gzip compressed files by decompressing then recursing
    if name_lower.endswith(".gz"):
        try:
            inner = _decompress_gzip_to_bytes_io(file, filename)
            return load_any(inner)
        except Exception:
            return None

    # Handle zip archives by extracting a useful inner file then recursing
    if name_lower.endswith(".zip"):
        inner = _extract_from_zip_to_bytes_io(file, filename)
        if inner is None:
            return None
        return load_any(inner)

    # At this point we deal with the actual data file
    try:
        # Excel
        if name_lower.endswith(".xlsx") or name_lower.endswith(".xls"):
            try:
                df = pd.read_excel(file, engine="openpyxl")
            except Exception:
                df = pd.read_excel(file)

        # CSV and TSV
        elif name_lower.endswith(".csv"):
            df = pd.read_csv(file)
        elif name_lower.endswith(".tsv"):
            df = pd.read_csv(file, sep="\t")

        # Generic text: try auto separator, then fall back to tab
        elif name_lower.endswith(".txt"):
            try:
                df = pd.read_csv(file, sep=None, engine="python")
            except Exception:
                file.seek(0)
                df = pd.read_csv(file, sep="\t")

        # XML: try pandas read_xml which works for many simple structures
        elif name_lower.endswith(".xml"):
            try:
                df = pd.read_xml(file)
            except Exception as e:
                st.error(
                    tr(
                        f"Could not parse XML as a table: {e}",
                        f"No se pudo convertir el XML en una tabla: {e}",
                    )
                )
                st.code(traceback.format_exc(), language="text")
                return None

        # GeoJSON and GML: prefer geopandas if installed
        elif name_lower.endswith(".geojson") or name_lower.endswith(".json") and "geojson" in name_lower:
            try:
                import geopandas as gpd  # type: ignore

                gdf = gpd.read_file(file)
                df = pd.DataFrame(gdf.drop(columns=gdf.geometry.name, errors="ignore"))
            except Exception:
                try:
                    file.seek(0)
                except Exception:
                    pass
                try:
                    df = pd.read_json(file)
                except Exception as e:
                    st.error(
                        tr(
                            f"Could not read GeoJSON/JSON: {e}",
                            f"No se pudo leer el GeoJSON/JSON: {e}",
                        )
                    )
                    st.code(traceback.format_exc(), language="text")
                    return None

        elif name_lower.endswith(".gml"):
            try:
                import geopandas as gpd  # type: ignore

                gdf = gpd.read_file(file)
                df = pd.DataFrame(gdf.drop(columns=gdf.geometry.name, errors="ignore"))
            except Exception as e:
                st.error(
                    tr(
                        f"Could not read GML. Install geopandas to support this format. Error: {e}",
                        f"No se pudo leer GML. Instala geopandas para soportar este formato. Error: {e}",
                    )
                )
                st.code(traceback.format_exc(), language="text")
                return None

        # NetCDF and similar, or anything else: delegate to core.load_table
        else:
            # load_table in core.py is expected to handle:
            # parquet, feather, json, ndjson, h5, hdf5, nc, netcdf, arrow, etc
            df = load_table(file)

        df = normalize_columns(df)

        if df.empty:
            st.error(
                tr(
                    "The file was read successfully, but it contains no rows.",
                    "El archivo se leyó correctamente, pero no contiene filas.",
                )
            )
            return None

        # Row guard for extremely large datasets
        max_rows = 1_000_000
        if len(df) > max_rows:
            st.warning(
                tr(
                    f"File has {len(df)} rows; only the first {max_rows} rows will be used to keep the app stable.",
                    f"El archivo tiene {len(df)} filas; solo se usarán las primeras {max_rows} para mantener la app estable.",
                )
            )
            df = df.head(max_rows)

        return df

    except Exception as e:
        st.error(tr(f"Could not read file. {e}", f"No se pudo leer el archivo. {e}"))
        st.code(traceback.format_exc(), language="text")
        return None


def ensure_columns(df: pd.DataFrame, repair: str, energy: str) -> bool:
    """
    Ensure the requested repair/energy columns exist.

    We keep this strict (no silent remapping) so that compute_rye_from_df,
    which already has its own safety logic, receives exactly what the user
    expects.
    """
    miss = [c for c in [repair, energy] if c not in df.columns]
    if miss:
        st.error(tr(f"Missing columns: {', '.join(miss)}", f"Faltan columnas: {', '.join(miss)}"))
        st.write(tr("Found columns:", "Columnas encontradas:"), list(df.columns))
        return False
    return True


def compute_block(df: pd.DataFrame, label: str, sim_mult: float, auto_roll_flag: bool) -> Dict[str, Any]:
    df_sim = df.copy()
    if col_energy in df_sim.columns:
        df_sim[col_energy] = pd.to_numeric(df_sim[col_energy], errors="coerce").apply(
            lambda x: safe_float(x) * sim_mult
        )

    rye = _compute_rye_from_df(df_sim, repair_col=col_repair, energy_col=col_energy)

    # window selection
    w = window
    if auto_roll_flag:
        w = smart_window(len(df_sim), getattr(preset, "default_rolling", None))

    rye_roll = _rolling_series(rye, w)
    rye_ema = ema_series(rye, span=int(ema_span)) if ema_span and ema_span > 1 else None
    rye_cum = cumulative_series(rye)

    summary = _summarize_series(rye)
    summary_roll = _summarize_series(rye_roll)

    out: Dict[str, Any] = {
        "label": label,
        "df": df_sim,
        "w": w,
        "rye": rye,
        "rye_roll": rye_roll,
        "rye_ema": rye_ema,
        "rye_cum": rye_cum,
        "summary": summary,
        "summary_roll": summary_roll,
    }

    # Optional advanced analytics
    try:
        if detect_regimes is not None:
            out["regimes"] = detect_regimes(
                rye_roll if len(rye_roll) >= max(3, w) else rye
            )
    except Exception:
        out["regimes"] = None

    try:
        if (
            energy_delta_performance_correlation is not None
            and col_repair in df_sim.columns
            and col_energy in df_sim.columns
        ):
            out["correlation"] = energy_delta_performance_correlation(
                df_sim, perf_col=col_repair, energy_col=col_energy
            )
    except Exception:
        out["correlation"] = None

    try:
        if estimate_noise_floor is not None:
            out["noise_floor"] = estimate_noise_floor(rye)
    except Exception:
        out["noise_floor"] = None

    try:
        if bootstrap_rolling_mean is not None:
            out["bands"] = bootstrap_rolling_mean(rye, window=w, n_boot=100)
    except Exception:
        out["bands"] = None

    return out


def make_interpretation(summary: dict, w: int, sim_mult: float, preset_name: str) -> str:
    """
    Turn summary stats into a richer narrative for humans and teams.

    Works in both languages and in both system and marketing presets.
    Does not assume any particular summary keys beyond a few safe ones.
    """
    mean_v = float(summary.get("mean", 0) or 0)
    max_v = float(summary.get("max", 0) or 0)
    min_v = float(summary.get("min", 0) or 0)
    std_v = float(summary.get("std", 0) or 0)
    iqr_v = float(summary.get("iqr", 0) or 0)
    count_v = float(summary.get("count", 0) or 0)
    p10 = summary.get("p10", None)
    p90 = summary.get("p90", None)
    resil = float(summary.get("resilience", 0) or 0) if "resilience" in summary else None

    # optional extra hints if summarize() provides them
    nonzero_frac = summary.get("nonzero_fraction", None)
    positive_frac = summary.get("positive_fraction", None)

    marketing_mode_local = preset_name.lower().startswith("marketing")

    # variation label
    if std_v < 0.05:
        var_label_en = "very low"
        var_label_es = "muy baja"
    elif std_v < 0.1:
        var_label_en = "low"
        var_label_es = "baja"
    elif std_v < 0.25:
        var_label_en = "moderate"
        var_label_es = "moderada"
    else:
        var_label_en = "high"
        var_label_es = "alta"

    lines: list[str] = []

    # 1. Data volume and basic stats
    if language == "Español":
        if count_v <= 0:
            lines.append(
                "No se encontraron observaciones válidas de RYE. Revisa el archivo, las columnas seleccionadas "
                "o cualquier filtrado previo."
            )
            return " ".join(lines)
        lines.append(
            f"La eficiencia promedio (RYE medio) es {mean_v:.3f}, con un rango aproximado de "
            f"[{min_v:.3f}, {max_v:.3f}] en los datos analizados."
        )
        lines.append(f"Se analizaron {int(count_v)} observaciones.")
        if std_v > 0 or iqr_v > 0:
            lines.append(
                f"La variación en la eficiencia es {var_label_es} "
                f"(desviación estándar {std_v:.3f}, IQR {iqr_v:.3f})."
            )
        if p10 is not None and p90 is not None:
            lines.append(
                f"La mayoría de los ciclos de reparación se encuentran entre {p10:.3f} y {p90:.3f} de RYE "
                "según el rango del 10 al 90 por ciento."
            )
    else:
        if count_v <= 0:
            lines.append(
                "No valid RYE observations were found. Check the file, the selected columns, "
                "or any filtering applied before analysis."
            )
            return " ".join(lines)
        lines.append(
            f"Average efficiency (RYE mean) is {mean_v:.3f}, with an approximate range "
            f"of [{min_v:.3f}, {max_v:.3f}] across the observed data."
        )
        lines.append(f"Based on {int(count_v)} observations.")
        if std_v > 0 or iqr_v > 0:
            lines.append(
                f"Variation in efficiency is {var_label_en} "
                f"(std {std_v:.3f}, IQR {iqr_v:.3f})."
            )
        if p10 is not None and p90 is not None:
            lines.append(
                f"Most repair cycles sit between {p10:.3f} and {p90:.3f} RYE, "
                "based on the 10th to 90th percentile range."
            )

    # 2. Flat or suspicious patterns
    span = max_v - min_v
    near_zero_band = abs(mean_v) < 0.02 and span < 0.05 and count_v >= 10

    if language == "Español":
        if near_zero_band:
            lines.append(
                "El RYE parece casi plano cerca de cero. Esto puede significar que el sistema apenas está "
                "respondiendo a la energía, que el cálculo de RYE está mal configurado o que el conjunto de "
                "datos no contiene suficiente variación en reparación o desempeño."
            )
        elif span < 0.05 and abs(mean_v) > 0.1:
            lines.append(
                "El RYE es casi constante pero distinto de cero. El sistema opera en una banda de eficiencia "
                "muy estable; para mejorar más, habría que cambiar la estrategia o las condiciones de entrada."
            )
    else:
        if near_zero_band:
            lines.append(
                "RYE is almost perfectly flat around zero. This can mean the system barely reacts to energy, "
                "that the RYE calculation is misconfigured, or that the dataset lacks meaningful variation in "
                "repair or performance."
            )
        elif span < 0.05 and abs(mean_v) > 0.1:
            lines.append(
                "RYE is almost constant but clearly above or below zero. The system is operating in a very stable "
                "efficiency band; meaningful improvement would likely require changing the strategy or inputs."
            )

    # 3. Resilience-focused text
    if resil is not None:
        if language == "Español":
            if resil < 0.1:
                lines.append(
                    "La resiliencia es prácticamente cero. Esto sugiere que no hay regulación estable de la "
                    "reparación: la eficiencia puede saltar de picos a caídas sin una zona de control clara."
                )
            elif resil < 0.4:
                lines.append(
                    "La resiliencia es intermedia. El sistema mantiene cierta estabilidad pero aparecen periodos "
                    "claros en los que la eficiencia de reparación se degrada."
                )
            else:
                lines.append(
                    "La resiliencia es alta. La eficiencia de reparación se mantiene estable aunque cambie la "
                    "energía o las condiciones, lo que indica bucles de control efectivos."
                )
        else:
            if resil < 0.1:
                lines.append(
                    "Resilience is effectively zero. There is no stable repair regulation; efficiency can jump "
                    "from spikes to crashes without a clear control zone."
                )
            elif resil < 0.4:
                lines.append(
                    "Resilience is moderate. The system holds some stability, but there are distinct periods "
                    "where repair efficiency degrades."
                )
            else:
                lines.append(
                    "Resilience is high. Repair efficiency remains stable even as energy or conditions change, "
                    "which points to effective control loops."
                )

    # 4. Efficiency strength and band interpretation
    if marketing_mode_local:
        if language == "Español":
            if mean_v > 1.0:
                lines.append(
                    "Cada unidad de presupuesto generó más de una unidad de resultado en promedio; la campaña "
                    "muestra una eficiencia sobresaliente."
                )
            elif mean_v > 0.5:
                lines.append(
                    "La eficiencia es fuerte. Tiene sentido recortar segmentos de alto costo y bajo resultado "
                    "para empujar el RYE hacia arriba."
                )
            else:
                lines.append(
                    "La eficiencia es modesta. Conviene localizar canales o campañas en los que el gasto es alto "
                    "pero los resultados son débiles para repararlos o reasignar presupuesto."
                )
        else:
            if mean_v > 1.0:
                lines.append(
                    "Each unit of budget returned more than one unit of outcome on average; campaign efficiency is outstanding."
                )
            elif mean_v > 0.5:
                lines.append(
                    "Efficiency is strong. It is worth trimming high cost, low outcome segments to push RYE even higher."
                )
            else:
                lines.append(
                    "Efficiency is modest. Focus on channels or campaigns where spend is high but outcomes are weak, then repair or reallocate."
                )
    else:
        if language == "Español":
            if mean_v > 1.0:
                lines.append(
                    "Cada unidad de energía produjo más de una unidad de reparación en promedio. El sistema "
                    "opera con una eficiencia de reparación muy alta."
                )
            elif mean_v > 0.5:
                lines.append(
                    "La eficiencia es sólida. Reducir gastos de energía innecesarios y concentrarse en regiones "
                    "de alto rendimiento puede elevar aún más el promedio."
                )
            else:
                lines.append(
                    "La eficiencia es modesta. Busca zonas donde la energía invertida es alta pero el retorno "
                    "de reparación es bajo para podarlas o rediseñar las intervenciones."
                )
        else:
            if mean_v > 1.0:
                lines.append(
                    "Each unit of energy returned more than one unit of repair on average. The system is operating with very high repair efficiency."
                )
            elif mean_v > 0.5:
                lines.append(
                    "Efficiency is solid. Reducing unnecessary energy use and focusing on high yield regions can lift the mean further."
                )
            else:
                lines.append(
                    "Efficiency is modest. Look for regions where energy is high but repair return is weak, then prune or redesign interventions."
                )

    # 5. Early warning on instability around thresholds
    if p10 is not None and p90 is not None:
        crosses_low = p10 < 0.3 < p90
        crosses_high = p10 < 0.6 < p90
        if language == "Español":
            if crosses_low and not crosses_high:
                lines.append(
                    "La serie cruza la banda de RYE 0.3, lo que indica periodos donde la eficiencia cae por "
                    "debajo de lo deseable. Es una señal temprana para reforzar los bucles de corrección."
                )
            elif crosses_high:
                lines.append(
                    "El RYE recorre tanto zonas débiles como zonas estables por encima de 0.6. Esto sugiere que "
                    "hay condiciones bajo las cuales el sistema ya opera en modo de alta eficiencia; vale la pena "
                    "identificarlas y expandirlas."
                )
        else:
            if crosses_low and not crosses_high:
                lines.append(
                    "The series crosses the 0.3 RYE band, which means there are periods where efficiency falls "
                    "below a healthy level. This is an early warning to strengthen correction loops."
                )
            elif crosses_high:
                lines.append(
                    "RYE visits both weak and stable zones above 0.6. This suggests that under some conditions the "
                    "system already operates in a high efficiency mode; those conditions are worth identifying and scaling."
                )

    # 6. Optional nonzero and positive fraction hints
    if nonzero_frac is not None:
        try:
            nz = float(nonzero_frac)
        except Exception:
            nz = None
        if nz is not None:
            if language == "Español":
                if nz < 0.2:
                    lines.append(
                        "Solo una fracción pequeña de los ciclos muestra RYE distinto de cero, lo que indica que la "
                        "mayoría de los pasos no cambian el estado de reparación de manera medible."
                    )
                elif nz < 0.6:
                    lines.append(
                        "Una parte moderada de los ciclos aporta reparación efectiva; hay margen para reducir ciclos "
                        "innecesarios que consumen energía sin mejorar el sistema."
                    )
                else:
                    lines.append(
                        "La mayoría de los ciclos contribuyen con alguna reparación; la optimización puede enfocarse "
                        "en elevar la eficiencia de los peores segmentos."
                    )
            else:
                if nz < 0.2:
                    lines.append(
                        "Only a small fraction of cycles show nonzero RYE, which means most steps do not measurably "
                        "change the repair state."
                    )
                elif nz < 0.6:
                    lines.append(
                        "A moderate share of cycles provide effective repair; there is room to reduce cycles that burn "
                        "energy without improving the system."
                    )
                else:
                    lines.append(
                        "Most cycles contribute some repair; optimization can focus on lifting the worst performing segments."
                    )

    if positive_frac is not None:
        try:
            pf = float(positive_frac)
        except Exception:
            pf = None
        if pf is not None:
            if language == "Español":
                if pf < 0.5:
                    lines.append(
                        "Menos de la mitad de los ciclos terminan con RYE positivo. El sistema pasa mucho tiempo "
                        "corrigiendo fallos o revirtiendo ineficiencias."
                    )
                elif pf < 0.8:
                    lines.append(
                        "Una mayoría de los ciclos es positiva, pero los periodos negativos siguen siendo importantes. "
                        "Es clave localizar esos tramos y rediseñar la estrategia de reparación."
                    )
                else:
                    lines.append(
                        "Casi todos los ciclos aportan reparación positiva. El reto principal es mejorar la eficiencia "
                        "de los ciclos menos rentables, no eliminar ciclos fallidos."
                    )
            else:
                if pf < 0.5:
                    lines.append(
                        "Fewer than half of the cycles end with positive RYE. The system spends a lot of time "
                        "correcting failures or undoing inefficiencies."
                    )
                elif pf < 0.8:
                    lines.append(
                        "Most cycles are positive, but negative periods are still significant. It is crucial to "
                        "locate those stretches and redesign the repair strategy."
                    )
                else:
                    lines.append(
                        "Almost all cycles deliver positive repair. The main challenge is to lift efficiency among "
                        "the least productive cycles rather than eliminating failed ones."
                    )

    # 6b. Ecological and marine specific hint if preset name suggests that domain
    preset_lower = preset_name.lower()
    if any(key in preset_lower for key in ["marine", "ocean", "ecology", "limnology"]):
        if language == "Español":
            lines.append(
                "Las oscilaciones de RYE pueden reflejar acoplamiento metabólico entre la producción primaria "
                "y la respiración del ecosistema, así como cambios estacionales en la estabilidad del sistema."
            )
        else:
            lines.append(
                "Oscillations in RYE may reflect metabolic coupling between primary production and ecosystem "
                "respiration, as well as seasonal shifts in system stability."
            )

    # 7. Rolling window and simulation factor
    if language == "Español":
        lines.append(f"La ventana móvil de {w} puntos ayuda a suavizar el ruido de corto plazo.")
        if sim_mult != 1.0:
            if sim_mult < 1.0:
                lines.append(
                    f"Se aplicó un factor de escala de energía de {sim_mult:.2f}. Si los resultados se mantienen, "
                    "un menor gasto de energía debería elevar el RYE observado."
                )
            else:
                lines.append(
                    f"Se aplicó un factor de escala de energía de {sim_mult:.2f}. A menos que la reparación o el "
                    "desempeño mejoren en la misma proporción, el RYE tenderá a disminuir."
                )
    else:
        lines.append(f"A rolling window of {w} points smooths short term noise.")
        if sim_mult != 1.0:
            if sim_mult < 1.0:
                lines.append(
                    f"An energy scaling factor of {sim_mult:.2f} was applied. If outcomes stay constant, "
                    "using less energy should increase observed RYE."
                )
            else:
                lines.append(
                    f"An energy scaling factor of {sim_mult:.2f} was applied. Unless repair or performance improves "
                    "at a similar rate, RYE will tend to fall."
                )

    # 8. Next steps guidance
    if marketing_mode_local:
        if language == "Español":
            lines.append(
                "Siguiente paso para equipos de marketing: vincula picos y caídas de RYE con canales, creativos y "
                "audiencias específicos, y utiliza esa señal para mover presupuesto, ajustar frecuencia y diseñar pruebas A/B."
            )
        else:
            lines.append(
                "Next steps for marketing teams: map RYE spikes and dips to specific channels, creatives, and "
                "audiences, and use that signal to guide budget shifts, frequency caps, and A/B tests."
            )
    else:
        if language == "Español":
            lines.append(
                "Siguiente paso: vincula los picos y caídas de RYE con intervenciones concretas y repite ciclos TGRM "
                "(detectar, corregir con el mínimo cambio, verificar). Usa las zonas de alto RYE como huellas de "
                "configuraciones sanas y las zonas de bajo RYE como candidatos para reparación cuantificada."
            )
        else:
            lines.append(
                "Next: map spikes and dips in RYE to concrete interventions and iterate TGRM loops "
                "(detect, minimal fix, verify). Treat high RYE zones as fingerprints of healthy configurations "
                "and low RYE zones as candidates for quantified repair."
            )

    return " ".join(lines)


def make_quick_summary(summary: dict, w: int, preset_name: str) -> str:
    """
    Short 1 to 2 sentence summary for the top of the report.
    This is what Erika or any scientist can read in 3 seconds.
    """
    mean_v = float(summary.get("mean", 0) or 0)
    resil = float(summary.get("resilience", 0) or 0)
    preset_lower = preset_name.lower()
    marketing_mode_local = preset_lower.startswith("marketing")

    if language == "Español":
        base = f"Eficiencia media RYE {mean_v:.2f}, resiliencia {resil:.2f}."
        if resil < 0.1:
            tail = " Los ciclos muestran inestabilidad clara entre periodos."
        elif resil < 0.4:
            tail = " La estabilidad es mixta; hay tramos donde la eficiencia cae de forma notable."
        else:
            tail = " La eficiencia se mantiene estable incluso cuando cambian las condiciones."

        if any(key in preset_lower for key in ["marine", "ocean", "ecology", "limnology"]):
            tail += " En contexto marino, esto puede reflejar diferencias entre estaciones o temporadas."

        if marketing_mode_local:
            tail += " Interpreta RYE como resultado por unidad de presupuesto."
        return base + tail
    else:
        base = f"Average RYE efficiency {mean_v:.2f}, resilience {resil:.2f}."
        if resil < 0.1:
            tail = " Cycles show clear instability between periods."
        elif resil < 0.4:
            tail = " Stability is mixed, with stretches where efficiency drops sharply."
        else:
            tail = " Repair efficiency stays stable even as conditions change."

        if any(key in preset_lower for key in ["marine", "ocean", "ecology", "limnology"]):
            tail += " In a marine context this likely reflects differences between stations or seasons."

        if marketing_mode_local:
            tail += " Interpret RYE as outcome per unit of spend."
        return base + tail


# ---------------- Main UI ----------------
tab1, tab2, tab3, tab4 = st.tabs(
    [
        tr("Single analysis", "Análisis único"),
        tr("Compare datasets", "Comparar conjuntos"),
        tr("Multi domain", "Multi dominio"),
        tr("Reports", "Reportes"),
    ]
)

df1 = load_any(file1)
df2 = load_any(file2)

# ---------- Tab 1 ----------
with tab1:
    if df1 is None:
        st.info(tr("Upload a file in the sidebar to begin.", "Sube un archivo en la barra lateral para comenzar."))
    else:
        if ensure_columns(df1, col_repair, col_energy):
            block = compute_block(df1, "primary", sim_factor, auto_roll)
            rye = block["rye"]
            rye_roll = block["rye_roll"]
            rye_ema = block["rye_ema"]
            rye_cum = block["rye_cum"]
            w = block["w"]
            summary = block["summary"]

            colA, colB, colC = st.columns(3)
            colA.metric("RYE mean", f"{summary.get('mean', 0):.4f}", help=tr("Average RYE across rows", "Promedio de RYE en todas las filas"))
            colB.metric("Median", f"{summary.get('median', 0):.4f}")
            colC.metric(
                "Resilience",
                f"{summary.get('resilience', 0):.3f}" if "resilience" in summary else "-",
                help=tr("Stability of efficiency under fluctuation (if computed)", "Estabilidad de la eficiencia frente a fluctuaciones (si se calcula)"),
            )

            if marketing_mode:
                st.caption(
                    tr(
                        "For marketing: higher RYE means more outcome per dollar or per unit of effort, "
                        "holding everything else constant.",
                        "En marketing: un RYE más alto significa más resultado por dólar o por unidad de esfuerzo, "
                        "manteniendo todo lo demás constante.",
                    )
                )

            st.write(tr("Columns:", "Columnas:"))
            st.json(list(df1.columns))

            # Simple marketing helpers
            if marketing_mode:
                with st.expander(tr("Marketing helpers", "Ayudas de marketing"), expanded=False):
                    cols = list(df1.columns)

                    # choose defaults
                    default_outcome = col_repair if col_repair in cols else (cols[0] if cols else None)
                    if col_energy in cols:
                        default_cost = col_energy
                    elif len(cols) > 1:
                        default_cost = cols[1]
                    else:
                        default_cost = default_outcome

                    idx_outcome = cols.index(default_outcome) if default_outcome in cols else 0
                    idx_cost = cols.index(default_cost) if default_cost in cols else 0

                    outcome_col = st.selectbox(
                        tr("Outcome column", "Columna de resultado"),
                        cols,
                        index=idx_outcome,
                        key="mk_outcome_col",
                        help=tr(
                            "Revenue, conversions or main result.",
                            "Ingresos, conversiones u otro resultado principal.",
                        ),
                    )
                    cost_col = st.selectbox(
                        tr("Cost column", "Columna de costo"),
                        cols,
                        index=idx_cost,
                        key="mk_cost_col",
                        help=tr(
                            "Spend, budget, impressions or similar cost measure.",
                            "Gasto, presupuesto, impresiones u otra medida de costo.",
                        ),
                    )

                    outcome_series = pd.to_numeric(df1[outcome_col], errors="coerce")
                    cost_series = pd.to_numeric(df1[cost_col], errors="coerce")
                    total_outcome = float(outcome_series.sum())
                    total_cost = float(cost_series.sum())
                    roas = total_outcome / total_cost if total_cost != 0 else np.nan

                    mcol1, mcol2, mcol3 = st.columns(3)
                    mcol1.metric(tr("Total outcome", "Resultado total"), f"{total_outcome:,.2f}")
                    mcol2.metric(tr("Total cost", "Costo total"), f"{total_cost:,.2f}")
                    mcol3.metric(
                        tr("ROAS (outcome per cost)", "ROAS (resultado por costo)"),
                        f"{roas:,.3f}" if np.isfinite(roas) else "NA",
                    )

                    domain_col_for_marketing = col_domain if col_domain in df1.columns else None
                    if domain_col_for_marketing is not None:
                        seg_df = pd.DataFrame(
                            {
                                "segment": df1[domain_col_for_marketing].astype(str),
                                "RYE": rye,
                            }
                        )
                        seg_mean = seg_df.groupby("segment", dropna=False)["RYE"].mean().sort_values(ascending=False)

                        st.markdown(tr("Top segments by RYE:", "Segmentos con mejor RYE:"))
                        st.dataframe(
                            seg_mean.head(5)
                            .reset_index()
                            .rename(columns={"segment": "segment", "RYE": "mean_RYE"})
                        )

                        if len(seg_mean) > 5:
                            st.markdown(tr("Segments to repair first (lowest RYE):", "Segmentos a reparar primero (RYE más bajo):"))
                            st.dataframe(
                                seg_mean.tail(5)
                                .reset_index()
                                .rename(columns={"segment": "segment", "RYE": "mean_RYE"})
                            )

                    st.caption(
                        tr(
                            "Use high RYE segments as templates and reduce spend in low RYE segments until they are repaired.",
                            "Usa los segmentos con RYE alto como plantillas y reduce el gasto en segmentos con RYE bajo hasta repararlos.",
                        )
                    )

            # RYE line(s)
            if col_time in df1.columns:
                idx = df1[col_time]
                fig = px.line(
                    x=idx, y=rye, labels={"x": col_time, "y": "RYE"}, title="RYE"
                )
                add_stability_bands(fig)
                st.plotly_chart(fig, use_container_width=True)

                fig2 = px.line(
                    x=idx,
                    y=rye_roll,
                    labels={"x": col_time, "y": f"RYE rolling {w}"},
                    title=f"RYE rolling window {w}",
                )
                if rye_ema is not None:
                    fig2.add_scatter(x=idx, y=rye_ema, mode="lines", name=f"EMA {ema_span}")
                add_stability_bands(fig2)
                st.plotly_chart(fig2, use_container_width=True)

                fig3 = px.line(
                    x=idx,
                    y=rye_cum,
                    labels={"x": col_time, "y": "Cumulative RYE"},
                    title="Cumulative RYE (sum of step yields)",
                )
                st.plotly_chart(fig3, use_container_width=True)
            else:
                fig = px.line(
                    pd.DataFrame({"RYE": rye}), y="RYE", title="RYE"
                )
                add_stability_bands(fig)
                st.plotly_chart(fig, use_container_width=True)

                ycols = {f"RYE rolling {w}": rye_roll}
                if rye_ema is not None:
                    ycols[f"EMA {ema_span}"] = rye_ema
                fig2 = px.line(
                    pd.DataFrame(ycols), title=f"Smoothed RYE (rolling {w} and EMA)"
                )
                add_stability_bands(fig2)
                st.plotly_chart(fig2, use_container_width=True)

                fig3 = px.line(
                    pd.DataFrame({"Cumulative RYE": rye_cum}),
                    y="Cumulative RYE",
                    title="Cumulative RYE",
                )
                st.plotly_chart(fig3, use_container_width=True)

            # Extra charts
            with st.expander(tr("More visuals", "Más visualizaciones")):
                hist = px.histogram(
                    pd.DataFrame({"RYE": rye}),
                    x="RYE",
                    nbins=30,
                    title="RYE distribution (histogram)",
                )
                st.plotly_chart(hist, use_container_width=True)

                if col_repair in df1.columns and col_energy in df1.columns:
                    dperf = _delta_performance(df1[col_repair])
                    scatter = px.scatter(
                        x=df1[col_energy],
                        y=dperf,
                        labels={"x": col_energy, "y": "Δ" + col_repair},
                        title="Energy vs ΔPerformance",
                    )
                    st.plotly_chart(scatter, use_container_width=True)

            # Diagnostics (optional analytics)
            with st.expander(tr("Diagnostics", "Diagnósticos")):
                if energy_delta_performance_correlation is not None:
                    try:
                        corr = energy_delta_performance_correlation(
                            df1, perf_col=col_repair, energy_col=col_energy
                        )
                        st.write(tr("Energy and ΔPerformance correlation:", "Correlación entre energía y Δdesempeño:"), corr)
                    except Exception:
                        pass
                if estimate_noise_floor is not None:
                    try:
                        noise = estimate_noise_floor(rye_roll)
                        st.write(tr("Noise floor:", "Nivel de ruido:"), noise)
                    except Exception:
                        pass
                if detect_regimes is not None:
                    try:
                        regimes = detect_regimes(rye_roll)
                        if regimes:
                            st.write(tr("Detected regimes:", "Regímenes detectados:"))
                            st.json(regimes)
                    except Exception:
                        pass

            st.divider()
            st.subheader(tr("Summary", "Resumen"))
            st.code(json.dumps(summary, indent=2))

            enriched = df1.copy()
            enriched["RYE"] = rye
            enriched[f"RYE_rolling_{w}"] = rye_roll
            if rye_ema is not None:
                enriched[f"RYE_ema_{ema_span}"] = rye_ema
            enriched["RYE_cumulative"] = rye_cum
            st.download_button(
                tr("Download enriched CSV (with RYE)", "Descargar CSV enriquecido (con RYE)"),
                enriched.to_csv(index=False).encode("utf-8"),
                file_name="rye_enriched.csv",
                mime="text/csv",
            )

            st.download_button(
                tr("Download RYE series CSV", "Descargar serie de RYE en CSV"),
                pd.Series(rye, name="RYE").to_csv(index_label="index").encode("utf-8"),
                file_name="rye.csv",
                mime="text/csv",
            )
            st.download_button(
                tr("Download summary JSON", "Descargar JSON de resumen"),
                io.BytesIO(json.dumps(summary, indent=2).encode("utf-8")).getvalue(),
                file_name="summary.json",
                mime="application/json",
            )

# ---------- Tab 2 ----------
with tab2:
    if df1 is None or df2 is None:
        st.info(tr("Upload two files to compare.", "Sube dos archivos para comparar."))
    else:
        if ensure_columns(df1, col_repair, col_energy) and ensure_columns(
            df2, col_repair, col_energy
        ):
            b1 = compute_block(df1, "A", sim_factor, auto_roll)
            b2 = compute_block(df2, "B", sim_factor, auto_roll)

            s1 = b1["summary"].get("mean", 0.0)
            s2 = b2["summary"].get("mean", 0.0)
            r1 = b1["summary"].get("resilience", 0)
            r2 = b2["summary"].get("resilience", 0)
            delta = s2 - s1
            pct = (delta / s1) * 100 if s1 != 0 else float("inf")

            colA, colB, colC, colD = st.columns(4)
            colA.metric("Mean RYE A", f"{s1:.4f}")
            colB.metric("Mean RYE B", f"{s2:.4f}")
            colC.metric("Δ Mean", f"{delta:.4f}", f"{pct:.2f}%")
            colD.metric("Resilience A / B", f"{r1:.3f} / {r2:.3f}" if r1 or r2 else "-")

            if marketing_mode and np.isfinite(pct):
                if delta > 0:
                    msg_en = (
                        f"Dataset B is more efficient. For the same unit of spend, it delivers about "
                        f"{pct:.1f}% more outcome than dataset A based on mean RYE."
                    )
                    msg_es = (
                        f"El conjunto B es más eficiente. Para la misma unidad de gasto entrega alrededor de "
                        f"{pct:.1f}% más resultado que el conjunto A según el RYE medio."
                    )
                elif delta < 0:
                    msg_en = (
                        f"Dataset B is less efficient. RYE suggests you get about {abs(pct):.1f}% less "
                        "outcome per unit of spend compared with dataset A."
                    )
                    msg_es = (
                        f"El conjunto B es menos eficiente. El RYE sugiere que obtienes alrededor de "
                        f"{abs(pct):.1f}% menos resultado por unidad de gasto en comparación con el conjunto A."
                    )
                else:
                    msg_en = "Datasets A and B deliver essentially the same outcome per unit of spend."
                    msg_es = "Los conjuntos A y B entregan prácticamente el mismo resultado por unidad de gasto."
                st.info(tr(msg_en, msg_es))

            if col_time in df1.columns and col_time in df2.columns:
                x1 = df1[col_time]
                x2 = df2[col_time]
                fig = px.line(
                    x=x1,
                    y=b1["rye"],
                    labels={"x": col_time, "y": "RYE"},
                    title="RYE comparison",
                )
                fig.add_scatter(x=x2, y=b2["rye"], mode="lines", name="B")
                add_stability_bands(fig)
                st.plotly_chart(fig, use_container_width=True)

                fig2 = px.line(
                    x=x1,
                    y=b1["rye_roll"],
                    labels={"x": col_time, "y": f"RYE rolling {b1['w']}"},
                    title=f"RYE rolling comparison (A:{b1['w']} / B:{b2['w']})",
                )
                fig2.add_scatter(x=x2, y=b2["rye_roll"], mode="lines", name="B")
                add_stability_bands(fig2)
                st.plotly_chart(fig2, use_container_width=True)
            else:
                fig = px.line(
                    pd.DataFrame({"RYE_A": b1["rye"]}),
                    y="RYE_A",
                    title="RYE comparison",
                )
                fig.add_scatter(y=b2["rye"], mode="lines", name="RYE_B")
                add_stability_bands(fig)
                st.plotly_chart(fig, use_container_width=True)

                fig2 = px.line(
                    pd.DataFrame({f"RYE_A_rolling_{b1['w']}": b1["rye_roll"]}),
                    y=f"RYE_A_rolling_{b1['w']}",
                    title="RYE rolling comparison",
                )
                fig2.add_scatter(
                    y=b2["rye_roll"], mode="lines", name=f"RYE_B_rolling_{b2['w']}"
                )
                add_stability_bands(fig2)
                st.plotly_chart(fig2, use_container_width=True)

            combined = pd.DataFrame(
                {
                    "RYE_A": b1["rye"],
                    "RYE_B": b2["rye"],
                    f"RYE_A_rolling_{b1['w']}": b1["rye_roll"],
                    f"RYE_B_rolling_{b2['w']}": b2["rye_roll"],
                }
            )
            st.download_button(
                tr("Download combined CSV (A vs B)", "Descargar CSV combinado (A vs B)"),
                combined.to_csv(index_label="index").encode("utf-8"),
                file_name="rye_combined.csv",
                mime="text/csv",
            )
# ---------- Tab 3 ----------
with tab3:
    if df1 is None:
        st.info(tr("Upload a file to see domain splits.", "Sube un archivo para ver los dominios."))
    else:
        # robust domain column selection (case-insensitive + aliases)
        lower_to_actual = {c.lower(): c for c in df1.columns}
        effective_domain_col = None

        # 1) direct match (case-insensitive) for user-provided col_domain
        if col_domain:
            effective_domain_col = lower_to_actual.get(col_domain.lower())

        # 2) canonical 'domain'
        if effective_domain_col is None and "domain" in df1.columns:
            effective_domain_col = "domain"

        # 3) aliases from DOMAIN_ALIASES then COLUMN_ALIASES["domain"], if provided
        if effective_domain_col is None:
            alias_list: list[str] = []
            if isinstance(DOMAIN_ALIASES, (list, tuple)):
                alias_list.extend([str(a) for a in DOMAIN_ALIASES])
            alias_list.extend(COLUMN_ALIASES.get("domain", []))
            for cand in alias_list:
                cand_actual = lower_to_actual.get(str(cand).lower())
                if cand_actual is not None:
                    effective_domain_col = cand_actual
                    break

        if effective_domain_col is None:
            st.info(
                tr(
                    f"No suitable domain column found. Looked for '{col_domain}', 'domain', and any configured aliases.",
                    f"No se encontró una columna de dominio adecuada. Se buscó '{col_domain}', 'domain' y los alias configurados.",
                )
            )
            st.write(tr("Available columns:", "Columnas disponibles:"), list(df1.columns))
        elif ensure_columns(df1, col_repair, col_energy):
            # remember the effective domain col for use in reports (separate key, safe)
            st.session_state["effective_domain_col"] = effective_domain_col

            b = compute_block(df1, "primary", sim_factor, auto_roll)
            dfp = b["df"].copy()
            dfp["RYE"] = b["rye"]

            title = tr("RYE by domain", "RYE por dominio")
            if marketing_mode:
                title = tr("RYE by campaign or segment", "RYE por campaña o segmento")

            if col_time in dfp.columns:
                fig = px.line(
                    dfp,
                    x=col_time,
                    y="RYE",
                    color=effective_domain_col,
                    title=title,
                )
            else:
                fig = px.line(
                    dfp,
                    y="RYE",
                    color=effective_domain_col,
                    title=title,
                )
            add_stability_bands(fig)
            st.plotly_chart(fig, use_container_width=True)

# ---------- Tab 4 ----------
with tab4:
    if df1 is None:
        st.info(tr("Upload a file to generate a report.", "Sube un archivo para generar un reporte."))
    else:
        if ensure_columns(df1, col_repair, col_energy):
            b = compute_block(df1, "primary", sim_factor, auto_roll)
            rye = b["rye"]
            rye_roll = b["rye_roll"]
            w = b["w"]
            summary = b["summary"]

            st.write(tr("Build a portable report to share with teams.", "Genera un reporte portátil para compartir con tu equipo."))

            # new quick summary on screen for fast reading
            quick_summary = make_quick_summary(summary, w, preset_name)
            st.subheader(tr("Quick summary", "Resumen rápido"))
            st.write(quick_summary)

            domain_meta_col = ""
            effective_dom = st.session_state.get("effective_domain_col")
            if effective_dom and effective_dom in df1.columns:
                domain_meta_col = effective_dom
            elif col_domain in df1.columns:
                domain_meta_col = col_domain

            metadata = {
                "rows": len(df1),
                "preset": preset_name,
                "repair_col": col_repair,
                "energy_col": col_energy,
                "time_col": col_time if col_time in df1.columns else "",
                "domain_col": domain_meta_col,
                "rolling_window": w,
                "columns": list(df1.columns),
                "quick_summary": quick_summary,
            }
            if marketing_mode:
                metadata["use_case"] = "marketing_efficiency"
            if doi_or_link.strip():
                metadata["dataset_link"] = doi_or_link.strip()
            for k in ("regimes", "correlation", "noise_floor", "bands"):
                if b.get(k) is not None:
                    metadata[k] = b[k]

            interp = make_interpretation(summary, w, sim_mult=sim_factor, preset_name=preset_name)

            with st.expander(tr("PDF diagnostics", "Diagnóstico del PDF"), expanded=False):
                if build_pdf is None:
                    st.error(tr("PDF builder is not loaded.", "El generador de PDF no está cargado."))
                    st.write(_probe_fpdf_version())
                    if _pdf_import_error:
                        st.code(_pdf_import_error, language="text")
                    try:
                        st.write("Working directory:", os.getcwd())
                        st.write("Files:", os.listdir("."))
                        st.write("Python:", sys.version)
                        fonts_ok = os.path.exists("fonts")
                        st.write("fonts/ exists:", fonts_ok)
                        if fonts_ok:
                            st.write("fonts/ contents:", os.listdir("fonts"))
                    except Exception as e:
                        st.write("Diag error:", e)
                else:
                    st.success(tr("PDF builder loaded.", "Generador de PDF cargado."))
                    st.write(_probe_fpdf_version())

            if st.button(tr("Generate PDF report", "Generar reporte en PDF"), use_container_width=True):
                if build_pdf is None:
                    st.error(
                        tr(
                            "PDF generator not available. Ensure report.py exists and fpdf2 is in requirements.txt.",
                            "El generador de PDF no está disponible. Asegúrate de que report.py exista y que fpdf2 esté en requirements.txt.",
                        )
                    )
                else:
                    try:
                        pdf_bytes = build_pdf(
                            list(rye),
                            summary,
                            metadata=metadata,
                            plot_series={
                                "RYE": list(rye),
                                f"RYE rolling {w}": list(rye_roll),
                            },
                            interpretation=interp,
                        )
                        st.download_button(
                            tr("Download RYE report PDF", "Descargar reporte RYE en PDF"),
                            data=pdf_bytes,
                            file_name="rye_report.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                        )
                    except Exception as e:
                        st.error(tr(f"PDF generation failed: {e}", f"La generación del PDF falló: {e}"))
                        st.code(traceback.format_exc(), language="text")
