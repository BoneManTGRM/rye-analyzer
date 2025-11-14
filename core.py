# core.py
# Core utilities for RYE Analyzer:
# file loading (CSV/TSV/XLS/XLSX + optional Parquet/Feather/JSON/NDJSON/HDF5/Arrow/NetCDF)
# column normalization + auto inference
# numeric coercion
# RYE computation (delta performance divided by energy) with guards
# rolling helpers: SMA and EMA and recommended window size
# summaries with resilience and quantiles
# optional analytics: regimes, correlations, noise floor, bootstrap bands
# extras: cumulative RYE, per domain RYE, outlier flags, simple unit scaling

from __future__ import annotations
import io
import re
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Union, Any

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError  # added

# ------------------------------
# PRESETS
# ------------------------------
try:
    # If a full presets.py exists, use that (preferred).
    from presets import PRESETS  # type: ignore
except Exception:
    # Fallback minimal preset set, used only when presets.py is missing.
    @dataclass(frozen=True)
    class Preset:
        name: str
        time: List[str]
        performance: List[str]
        energy: List[str]
        domain: Optional[str] = None
        default_rolling: int = 10
        tooltips: Optional[Dict[str, str]] = None

    def _kw(*items) -> List[str]:
        return list(dict.fromkeys([s.strip() for s in items if s]))

    PRESETS: Dict[str, Preset] = {
        # Note: domain now refers to the COLUMN NAME "domain" so the sidebar default is correct.
        "AI": Preset(
            "AI",
            time=_kw("time", "step", "iteration", "epoch", "t"),
            performance=_kw(
                "performance",
                "accuracy",
                "acc",
                "f1",
                "reward",
                "score",
                "coherence",
                "loss_inv",
                "bleu",
                "rouge",
            ),
            energy=_kw(
                "energy",
                "tokens",
                "compute",
                "cost",
                "gradient_updates",
                "lr",
                "batch_tokens",
            ),
            domain="domain",
            default_rolling=10,
            tooltips={"loss_inv": "1/loss, higher is better"},
        ),
        "Biology": Preset(
            "Biology",
            time=_kw("time", "t", "hours", "days", "samples"),
            performance=_kw(
                "performance",
                "viability",
                "function",
                "yield",
                "recovery",
                "signal",
                "od",
                "growth",
                "fitness",
            ),
            energy=_kw(
                "energy",
                "dose",
                "stressor",
                "input",
                "treatment",
                "drug",
                "radiation",
            ),
            domain="domain",
            default_rolling=10,
        ),
        "Robotics": Preset(
            "Robotics",
            time=_kw("time", "t", "cycle", "episode"),
            performance=_kw(
                "performance",
                "task_success",
                "score",
                "stability",
                "tracking_inv",
                "uptime",
                "mean_reward",
            ),
            energy=_kw(
                "energy",
                "power",
                "torque_int",
                "battery_used",
                "effort",
                "cpu_load",
            ),
            domain="domain",
            default_rolling=10,
        ),
        "Marine Biology": Preset(
            "Marine Biology",
            time=_kw("time", "t", "date", "day", "doy", "timestamp", "sample_time", "year", "season"),
            performance=_kw(
                "performance",
                "survival",
                "growth",
                "calcification",
                "recruitment",
                "photosynthesis",
                "chlorophyll",
                "chl",
                "coverage",
                "abundance",
                "diversity",
                "shannon",
                "richness",
                # synthetic examples / common marine metrics
                "avg_live_coral_cover_pct",
                "net_primary_prod_mg_c_m2_day",
                "predator_biomass_kg",
                "herbivore_biomass_kg",
                "aragonite_saturation_state",
                "gross_primary_prod_g_c_m2_day",
            ),
            energy=_kw(
                "energy",
                "dose",
                "nutrients",
                "nitrate",
                "phosphate",
                "silicate",
                "light",
                "par",
                "effort",
                "treatment",
                "temperature",
                "temp",
                "pco2",
                "salinity",
                "stress",
                # synthetic marine-energy proxies
                "survey_effort_hours",
                "chlorophyll_mg_m3",
                "survey_minutes",
                "dissolved_inorganic_carbon_umol_kg",
                "ecosystem_respiration_g_c_m2_day",
            ),
            domain="domain",
            default_rolling=10,
            tooltips={
                "chl": "Chlorophyll a proxy",
                "par": "Photosynthetically active radiation",
            },
        ),
        "Fisheries": Preset(
            "Fisheries",
            time=_kw("time", "t", "date", "trip", "haul", "set", "tow", "year"),
            performance=_kw(
                "performance",
                "cpue",
                "yield",
                "biomass",
                "catch_rate",
                "survival",
                "recruitment",
                "predator_biomass_kg",
                "herbivore_biomass_kg",
            ),
            energy=_kw(
                "energy",
                "effort",
                "soak_time",
                "net_hours",
                "trawl_hours",
                "fuel",
                "cost",
                "survey_minutes",
            ),
            domain="domain",
            default_rolling=10,
        ),
        "Coral Reef Monitoring": Preset(
            "Coral Reef Monitoring",
            time=_kw("time", "t", "date", "survey", "dive", "year"),
            performance=_kw(
                "performance",
                "live_coral_cover",
                "juvenile_density",
                "calcification",
                "photosynthesis",
                "recovery",
                "avg_live_coral_cover_pct",
            ),
            energy=_kw(
                "energy",
                "intervention",
                "outplanting",
                "nursery_cost",
                "effort",
                "par",
                "dose",
                "survey_effort_hours",
            ),
            domain="domain",
            default_rolling=10,
        ),
        "Oceanography/CTD": Preset(
            "Oceanography/CTD",
            time=_kw("time", "t", "cast", "profile", "date"),
            performance=_kw(
                "performance",
                "signal",
                "stability",
                "coherence",
                "recovery",
                "oxygen",
                "chlorophyll",
                "fluorescence",
                "aragonite_saturation_state",
            ),
            energy=_kw(
                "energy",
                "pump_power",
                "ship_time",
                "fuel",
                "cost",
                "cast_depth",
                "niskin_trips",
                "dissolved_inorganic_carbon_umol_kg",
            ),
            domain="domain",
            default_rolling=10,
        ),
        "Aquaculture": Preset(
            "Aquaculture",
            time=_kw("time", "t", "day", "date", "batch"),
            performance=_kw(
                "performance",
                "growth_rate",
                "survival",
                "feed_conversion_inv",
                "yield",
                "biomass",
                "health_score",
            ),
            energy=_kw(
                "energy",
                "feed",
                "aeration_power",
                "oxygenation",
                "water_exchange",
                "temperature",
                "dose",
                "cost",
            ),
            domain="domain",
            default_rolling=10,
            tooltips={"feed_conversion_inv": "Inverse FCR, higher is better"},
        ),
        # Omics-friendly fallback preset (e.g., DESeq2-style tables)
        "Omics/DESeq2": Preset(
            "Omics/DESeq2",
            time=_kw(
                "time",
                "t",
                "sample_index",
                "run",
                "cycle",
            ),
            performance=_kw(
                "log2fc",
                "log2_fold_change",
                "c19_c_log2fc",
                "c19_p_log2fc",
                "effect_size",
                "performance",
            ),
            energy=_kw(
                "dose",
                "concentration",
                "treatment",
                "energy",
                "padj",
                "pvalue",
            ),
            domain="symbol",
            default_rolling=10,
            tooltips={
                "Time": "Use time or sample_index if available; if not, the row index can be treated as a run index.",
                "Performance": "Use log2 fold change or effect_size as your signal of change.",
                "Energy": "Use dose, concentration, or padj as a proxy cost or stress level.",
                "Domain": "Group by symbol or gene_id to see per-gene repair patterns.",
            },
        ),
        # Fallback Marketing preset (in case presets.py is not present)
        "Marketing": Preset(
            "Marketing",
            time=_kw("date", "time", "day", "week", "campaign_day", "cohort_day"),
            performance=_kw(
                "performance",
                "ctr",
                "click_through_rate",
                "conversion_rate",
                "conv_rate",
                "cvr",
                "roas",
                "return_on_ad_spend",
                "retention",
                "engagement",
                "open_rate",
                "click_rate",
                "reply_rate",
                "signup_rate",
                "trial_rate",
                "mql_rate",
                "sql_rate",
            ),
            energy=_kw(
                "energy",
                "spend",
                "ad_spend",
                "budget",
                "media_spend",
                "impressions",
                "views",
                "emails_sent",
                "sends",
                "touches",
                "clicks",
                "sessions",
                "visits",
                "ad_cost",
            ),
            domain="domain",
            default_rolling=14,
            tooltips={
                "ctr": "Click-through rate; clicks / impressions.",
                "roas": "Return on ad spend; revenue / ad_spend.",
                "retention": "Share of users who stay active over time.",
            },
        ),
    }

# ------------------------------
# Column aliases for inference
# ------------------------------
def _dedupe_str(seq: Sequence[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in seq:
        if x is None:
            continue
        s = str(x).strip()
        if not s:
            continue
        s_lower = s.lower()
        if s_lower in seen:
            continue
        seen.add(s_lower)
        out.append(s_lower)
    return out

# Base seeds that already work well with existing datasets
_BASE_COLUMN_ALIASES: Dict[str, List[str]] = {
    "time": [
        "time",
        "t",
        "timestamp",
        "date",
        "datetime",
        "step",
        "iteration",
        "epoch",
        "hours",
        "days",
        "sample_time",
        "survey",
        "cast",
        "profile",
        "dive",
        "trip",
        "haul",
        "set",
        "tow",
        "batch",
        "week",
        "campaign_day",
        "cohort_day",
        "year",
        "season",
        "month",
        "day_of_week",
        "mission_time",
        "cycle",
        "visit",
    ],
    "performance": [
        "performance",
        "repair",
        "accuracy",
        "acc",
        "f1",
        "reward",
        "score",
        "coherence",
        "loss_inv",
        "bleu",
        "rouge",
        "viability",
        "function",
        "yield",
        "recovery",
        "signal",
        "od",
        "growth",
        "fitness",
        "survival",
        "calcification",
        "recruitment",
        "photosynthesis",
        "chlorophyll",
        "chl",
        "coverage",
        "abundance",
        "diversity",
        "shannon",
        "richness",
        "cpue",
        "biomass",
        "catch_rate",
        "juvenile_density",
        "live_coral_cover",
        "oxygen",
        "fluorescence",
        "growth_rate",
        "feed_conversion_inv",
        "health_score",
        "soh",
        "capacity_retained",
        # marine CSV examples
        "avg_live_coral_cover_pct",
        "net_primary_prod_mg_c_m2_day",
        "predator_biomass_kg",
        "herbivore_biomass_kg",
        "aragonite_saturation_state",
        "gross_primary_prod_g_c_m2_day",
        # marketing and web
        "ctr",
        "click_through_rate",
        "conversion_rate",
        "conv_rate",
        "cvr",
        "roas",
        "return_on_ad_spend",
        "retention",
        "engagement",
        "open_rate",
        "click_rate",
        "reply_rate",
        "signup_rate",
        "trial_rate",
        "mql_rate",
        "sql_rate",
        # extra marketing outcomes
        "conversions",
        "revenue",
        "sales",
        "signups",
        "leads",
        "purchases",
        "orders",
        "subscriptions",
        "new_customers",
        "ltv",
        "clv",
        "engagement_rate",
        "bounce_rate_inv",
        "unsub_rate_inv",
        # omics and DESeq2
        "log2fc",
        "log2_fold_change",
        "c19_c_log2fc",
        "c19_p_log2fc",
        "effect_size",
        # robotics and control
        "task_success",
        "success_rate",
        "stability",
        "tracking_inv",
        "tracking_error_inv",
        "uptime",
        "mean_reward",
        "completion_rate",
        "episode_return",
        "collision_rate_inv",
        "fall_rate_inv",
        "path_accuracy",
    ],
    "energy": [
        "energy",
        "effort",
        "cost",
        "compute",
        "tokens",
        "gradient_updates",
        "lr",
        "batch_tokens",
        "dose",
        "stressor",
        "input",
        "treatment",
        "drug",
        "radiation",
        "par",
        "light",
        "temperature",
        "temp",
        "pco2",
        "salinity",
        "nutrients",
        "nitrate",
        "phosphate",
        "silicate",
        "feed",
        "aeration_power",
        "oxygenation",
        "water_exchange",
        "pump_power",
        "ship_time",
        "fuel",
        "cast_depth",
        "niskin_trips",
        "soak_time",
        "net_hours",
        "trawl_hours",
        "power",
        "cpu_load",
        "torque_int",
        "battery_used",
        # marine CSV examples
        "survey_effort_hours",
        "chlorophyll_mg_m3",
        "survey_minutes",
        "dissolved_inorganic_carbon_umol_kg",
        "ecosystem_respiration_g_c_m2_day",
        # marketing and business
        "spend",
        "ad_spend",
        "media_spend",
        "budget",
        "impressions",
        "views",
        "emails_sent",
        "sends",
        "touches",
        "clicks",
        "sessions",
        "visits",
        "ad_cost",
        "reach",
        "traffic",
        "sms_sent",
        "messages_sent",
        # omics and DESeq2 style cost or stress
        "padj",
        "pvalue",
        # robotics and hardware
        "motor_current",
        "joint_torque",
        "control_effort",
    ],
    "domain": [
        "domain",
        "group",
        "group_id",
        "condition",
        "condition_id",
        "treatment_group",
        "scenario",
        "scenario_id",
        "scenario_stress",
        "trial",
        "replicate",
        "rep",
        "block",
        "cluster",
        "species",
        "site",
        "site_id",
        "station",
        "station_id",
        "transect_id",
        "meadow_id",
        "reef",
        "habitat",
        "run_id",
        "run",
        "experiment",
        "experiment_id",
        "plate",
        "sample_id",
        "subject_id",
        "patient_id",
        "arm",
        "cohort",
        # omics and users
        "symbol",
        "ensembl_id",
        "gene",
        "gene_id",
        "user_id",
        # marketing and business
        "campaign",
        "channel",
        "segment",
        "creative",
        "ad_group",
        "placement",
        "strategy",
        "country",
        "region",
        "service",
        "property",
        "store",
        "lane",
        "line",
        # robotics and infra
        "robot_id",
        "airframe_id",
        "asset_id",
        "pack_id",
        "well_id",
        "plant",
    ],
}

# Start aliases from base seeds
_COLUMN_ALIASES_WORK: Dict[str, List[str]] = {
    "time": list(_BASE_COLUMN_ALIASES["time"]),
    "performance": list(_BASE_COLUMN_ALIASES["performance"]),
    "energy": list(_BASE_COLUMN_ALIASES["energy"]),
    "domain": list(_BASE_COLUMN_ALIASES["domain"]),
}

# Enrich aliases with everything defined in PRESETS (including large 100 column lists)
for _preset in PRESETS.values():
    try:
        _COLUMN_ALIASES_WORK["time"].extend(getattr(_preset, "time", []) or [])
        _COLUMN_ALIASES_WORK["performance"].extend(getattr(_preset, "performance", []) or [])
        _COLUMN_ALIASES_WORK["energy"].extend(getattr(_preset, "energy", []) or [])
        dom_name = getattr(_preset, "domain", None)
        if dom_name:
            _COLUMN_ALIASES_WORK["domain"].append(dom_name)
    except Exception:
        # If a preset is odd shaped, just skip enrichment for it
        continue

# Final deduped, lowercased alias map used by inference and app layers
COLUMN_ALIASES: Dict[str, List[str]] = {
    k: _dedupe_str(v) for k, v in _COLUMN_ALIASES_WORK.items()
}

# Separate domain alias list for multi domain logic (used by app layer)
DOMAIN_ALIASES: List[str] = list(COLUMN_ALIASES.get("domain", []))

# ------------------------------
# File IO
# ------------------------------
def _read_text_table(text: str) -> pd.DataFrame:
    """
    Generic CSV/TSV reader for raw text blobs.
    Uses a quick heuristic: if the first line has tabs and no commas,
    treat as TSV, otherwise treat as CSV.
    """
    if not text or not text.strip():
        raise EmptyDataError("Uploaded text file is empty.")
    lines = text.splitlines()
    first = lines[0] if lines else ""
    sep = "\t" if ("\t" in text and "," not in first) else ","
    return pd.read_csv(io.StringIO(text), sep=sep)


def _load_dwc_archive_from_filelike(file_like) -> pd.DataFrame:
    """
    Best-effort Darwin Core Archive (DwC-A) loader.

    Expects a zip-like object. If a meta.xml file is present, use it to locate
    the core table. Otherwise, fall back to the first .txt file in the archive.
    """
    file_like.seek(0)
    with zipfile.ZipFile(file_like) as z:
        names = z.namelist()
        core_name: Optional[str] = None

        # Try meta.xml if available
        if "meta.xml" in names:
            try:
                meta_bytes = z.read("meta.xml")
                root = ET.fromstring(meta_bytes)
                ns = {"dwc": "http://rs.tdwg.org/dwc/text/"}
                core = root.find("dwc:core", ns)
                if core is not None:
                    files_elem = core.find("dwc:files", ns)
                    if files_elem is not None:
                        loc = files_elem.find("dwc:location", ns)
                        if loc is not None and loc.text:
                            core_name = loc.text.strip()
            except Exception:
                core_name = None

        # Fallback: first .txt file
        if core_name is None:
            txt_names = [n for n in names if n.lower().endswith(".txt")]
            if txt_names:
                core_name = txt_names[0]

        if core_name is None:
            raise ValueError(
                "Zip archive does not look like a Darwin Core Archive "
                "(no meta.xml or .txt table found)."
            )

        with z.open(core_name) as f:
            text = f.read().decode("utf-8", errors="replace")
            return _read_text_table(text)


def _load_eml_from_bytes(data: bytes) -> pd.DataFrame:
    """
    Very simple EML handler.

    Many .eml files in ecology are metadata only. Here we try to interpret the
    file as a delimited text table; if that fails, raise a clear message so the
    user knows they should export a CSV or DwC-A instead.
    """
    text = data.decode("utf-8", errors="replace")
    try:
        return _read_text_table(text)
    except Exception as e:
        raise ValueError(
            "This EML file appears to be metadata, not a tabular dataset. "
            "Please export the underlying data table as CSV or DwC-A."
        ) from e


def _load_rtf_from_bytes(data: bytes) -> pd.DataFrame:
    """
    Best-effort RTF loader.

    Strip common RTF control sequences in a naive way and then attempt to parse
    as a delimited text table. This will only work if the RTF actually contains
    a simple table exported as text.
    """
    text = data.decode("utf-8", errors="replace")
    # Rough stripping of common RTF control sequences
    text = re.sub(r"\\'[0-9a-fA-F]{2}", " ", text)  # hex escapes
    text = re.sub(r"\\[a-zA-Z]+-?\d* ?", " ", text)  # control words
    text = re.sub(r"[{}]", " ", text)  # group braces
    try:
        return _read_text_table(text)
    except Exception as e:
        raise ValueError(
            "Could not interpret this RTF as a tabular dataset. "
            "If it contains a table, please export it as CSV."
        ) from e


def _load_special_from_bytes(data: bytes, name: str) -> Optional[pd.DataFrame]:
    """
    Handle special formats that are not plain CSV/TSV/Excel/etc:
      - Darwin Core Archive (.zip, .dwc, .dwca, .dwc-a)
      - EML (.eml)
      - RTF (.rtf)
    Returns a DataFrame on success, or None to signal "fall back to normal".
    """
    lower = name.lower()

    # DwC-A (zip-based Darwin Core Archive)
    if lower.endswith((".zip", ".dwc", ".dwca", ".dwc-a")):
        buf = io.BytesIO(data)
        try:
            return _load_dwc_archive_from_filelike(buf)
        except Exception:
            # If it's just a generic zip or an unexpected layout,
            # fall through to normal handling.
            return None

    # EML metadata / text
    if lower.endswith(".eml"):
        try:
            return _load_eml_from_bytes(data)
        except Exception:
            return None

    # RTF text
    if lower.endswith(".rtf"):
        try:
            return _load_rtf_from_bytes(data)
        except Exception:
            return None

    return None


def load_table(src) -> pd.DataFrame:
    """
    Read CSV/TSV/XLS/XLSX plus optional Parquet/Feather/JSON/NDJSON/HDF5/Arrow/NetCDF.

    New behavior:
      - If the upload looks like a Darwin Core Archive (.zip/.dwc/.dwca/.dwc-a),
        try to parse it via meta.xml or the first .txt file.
      - If the upload is .eml or .rtf, attempt a best-effort text-table parse
        and raise a clear error if it's metadata-only.
    """
    # Uploaded file like Streamlit's UploadedFile
    if hasattr(src, "read") and not isinstance(src, (str, bytes)):
        data = src.read()
        name = getattr(src, "name", "upload")
        lower = name.lower()

        # Special formats first (DwC-A zip, EML, RTF)
        special_df = _load_special_from_bytes(data, name)
        if special_df is not None:
            return special_df

        buf = io.BytesIO(data)

        if lower.endswith((".xls", ".xlsx")):
            return pd.read_excel(buf)

        try:
            if lower.endswith(".parquet"):
                return pd.read_parquet(buf)
            if lower.endswith(".feather"):
                return pd.read_feather(buf)
            if lower.endswith(".arrow"):
                import pyarrow.ipc as ipc  # type: ignore

                reader = ipc.RecordBatchFileReader(buf)
                table = reader.read_all()
                return table.to_pandas()  # type: ignore
            if lower.endswith((".h5", ".hdf5")):
                return pd.read_hdf(buf)
            if lower.endswith(".json"):
                try:
                    return pd.read_json(buf, lines=True)
                except Exception:
                    buf.seek(0)
                    return pd.read_json(buf)
            if lower.endswith((".nc", ".netcdf")):
                import xarray as xr  # type: ignore

                ds = xr.open_dataset(buf)
                return ds.to_dataframe().reset_index()
        except Exception:
            pass

        # Generic text fall-back (CSV or TSV)
        text = data.decode("utf-8", errors="replace")
        return _read_text_table(text)

    # Pathlike
    path = str(src)
    lower = path.lower()

    # Special formats from disk
    if lower.endswith((".zip", ".dwc", ".dwca", ".dwc-a")):
        with open(path, "rb") as f:
            return _load_dwc_archive_from_filelike(f)
    if lower.endswith(".eml"):
        with open(path, "rb") as f:
            return _load_eml_from_bytes(f.read())
    if lower.endswith(".rtf"):
        with open(path, "rb") as f:
            return _load_rtf_from_bytes(f.read())

    if lower.endswith((".xls", ".xlsx")):
        return pd.read_excel(path)

    try:
        if lower.endswith(".parquet"):
            return pd.read_parquet(path)
        if lower.endswith(".feather"):
            return pd.read_feather(path)
        if lower.endswith(".arrow"):
            import pyarrow.ipc as ipc  # type: ignore

            with open(path, "rb") as f:
                reader = ipc.RecordBatchFileReader(f)
                table = reader.read_all()
                return table.to_pandas()  # type: ignore
        if lower.endswith((".h5", ".hdf5")):
            return pd.read_hdf(path)
        if lower.endswith(".json"):
            try:
                return pd.read_json(path, lines=True)
            except Exception:
                return pd.read_json(path)
        if lower.endswith((".nc", ".netcdf")):
            import xarray as xr  # type: ignore

            ds = xr.open_dataset(path)
            return ds.to_dataframe().reset_index()
    except Exception:
        pass

    # Final CSV/TSV fallback
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, sep="\t")

# ------------------------------
# Column normalization and inference
# ------------------------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names to snake_case and add friendly aliases.

    Important behavior for compatibility:
      - If a dataframe has "performance" but not "accuracy",
        create an "accuracy" column equal to "performance".
      - If a dataframe has "energy" but not "tokens",
        create a "tokens" column equal to "energy".

    This lets older code or saved session_state that still refers
    to accuracy/tokens work even when the actual CSV uses
    performance/energy.
    """
    def norm(c: str) -> str:
        s = str(c).strip().lower()
        s = re.sub(r"[^\w]+", "_", s)
        s = re.sub(r"_+", "_", s).strip("_")
        return s or "col"

    out = df.copy()
    out.columns = [norm(c) for c in out.columns]

    cols = list(out.columns)

    # Add compatibility aliases
    if "performance" in cols and "accuracy" not in cols:
        out["accuracy"] = out["performance"]
        cols.append("accuracy")
    if "energy" in cols and "tokens" not in cols:
        out["tokens"] = out["energy"]
        cols.append("tokens")

    return out

def _best_match(df_cols: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    """
    Best effort match:
      1) Exact case-insensitive match
      2) Substring match (alias contained in column name)
    """
    if not df_cols or not candidates:
        return None

    df_cols_lower = {c.lower(): c for c in df_cols}

    # exact
    for cand in candidates:
        if cand is None:
            continue
        key = str(cand).lower()
        if key in df_cols_lower:
            return df_cols_lower[key]

    # substring
    for cand in candidates:
        if cand is None:
            continue
        key = str(cand).lower()
        for col_lower, col_actual in df_cols_lower.items():
            if key and key in col_lower:
                return col_actual

    return None

def infer_columns(df: pd.DataFrame, preset_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Try to guess time, domain, performance, energy columns based on:
      1) The active preset keyword lists
      2) Generic COLUMN_ALIASES fallbacks
      3) Simple numeric fallbacks
      4) As a last resort for domain: pick the best categorical column
         (low cardinality, not time-like) so multi-domain still works.
    """
    cols = list(df.columns)
    preset = PRESETS.get(preset_name) if preset_name and preset_name in PRESETS else None

    guesses: Dict[str, Any] = {}

    # Time
    time_candidates: List[str] = []
    if preset is not None:
        time_candidates.extend(getattr(preset, "time", []))
    time_candidates.extend(COLUMN_ALIASES.get("time", []))
    guesses["time"] = _best_match(cols, time_candidates)

    # Domain
    domain_candidates: List[str] = []
    if preset is not None and getattr(preset, "domain", None):
        domain_candidates.append(getattr(preset, "domain"))
    domain_candidates.extend(COLUMN_ALIASES.get("domain", []))
    guesses["domain"] = _best_match(cols, domain_candidates)

    # Performance
    perf_candidates: List[str] = []
    if preset is not None:
        perf_candidates.extend(getattr(preset, "performance", []))
    perf_candidates.extend(COLUMN_ALIASES.get("performance", []))
    guesses["performance"] = _best_match(cols, perf_candidates)

    # Energy
    energy_candidates: List[str] = []
    if preset is not None:
        energy_candidates.extend(getattr(preset, "energy", []))
    energy_candidates.extend(COLUMN_ALIASES.get("energy", []))
    guesses["energy"] = _best_match(cols, energy_candidates)

    # Fallbacks: use numeric columns if we still have gaps
    numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    if guesses.get("performance") is None and numeric_cols:
        guesses["performance"] = numeric_cols[0]
    if guesses.get("energy") is None:
        if len(numeric_cols) > 1:
            guesses["energy"] = numeric_cols[1]
        elif numeric_cols:
            guesses["energy"] = numeric_cols[0]
    if guesses.get("time") is None:
        t_like = [
            c
            for c in cols
            if any(k in c.lower() for k in ("time", "date", "epoch", "step", "iteration", "year", "season"))
        ]
        if t_like:
            guesses["time"] = t_like[0]

    # Extra alias sweep for domain
    if guesses.get("domain") is None:
        for cand in COLUMN_ALIASES.get("domain", []):
            m = _best_match(cols, [cand])
            if m:
                guesses["domain"] = m
                break

    # Smart categorical fallback for domain:
    # if no alias matched, pick the most reasonable categorical column
    if guesses.get("domain") is None and len(df) > 0:
        n_rows = len(df)
        cat_candidates: List[tuple] = []

        for c in cols:
            # Avoid reusing the time column as domain
            if c == guesses.get("time"):
                continue

            cl = c.lower()
            # Skip obvious time-like columns
            if any(
                key in cl
                for key in ("time", "date", "epoch", "step", "iteration", "year", "season")
            ):
                continue

            s = df[c]
            try:
                nunique = int(s.nunique(dropna=True))
            except Exception:
                continue

            # Need at least 2 levels
            if nunique < 2:
                continue

            frac = float(nunique) / float(max(1, n_rows))

            # Skip near-unique IDs (almost one value per row)
            if frac > 0.9:
                continue

            is_catlike = (
                s.dtype == "object"
                or pd.api.types.is_categorical_dtype(s)
                or nunique <= 50
            )
            if not is_catlike:
                continue

            # Lower frac = fewer groups relative to rows (more domain-like)
            cat_candidates.append((frac, nunique, c))

        if cat_candidates:
            # Sort by fraction of unique values, then by absolute cardinality
            cat_candidates.sort(key=lambda x: (x[0], x[1]))
            guesses["domain"] = cat_candidates[0][2]

    return guesses

# ------------------------------
# Numerics and helpers
# ------------------------------
def safe_float(x) -> float:
    try:
        if x is None:
            return float("nan")
        if isinstance(x, (float, int, np.floating, np.integer)):
            return float(x)
        s = str(x).strip().replace(",", "")
        return float(s)
    except Exception:
        return float("nan")

def _coerce_numeric(series: Iterable) -> np.ndarray:
    return np.array([safe_float(v) for v in series], dtype=float)

def scale_units(arr: Sequence[float], factor: float) -> np.ndarray:
    a = np.array(arr, dtype=float)
    return a * float(factor)

# ------------------------------
# RYE core
# ------------------------------
def compute_rye_from_df(
    df: pd.DataFrame,
    repair_col: str = "performance",
    energy_col: str = "energy",
    time_col: Optional[str] = None,
    clamp_negative_delta: bool = True,
    energy_floor: float = 1e-9,
) -> np.ndarray:
    """
    RYE per step = max(delta performance, 0) divided by max(energy, energy_floor)

    repair_col: column with performance or repair signal
    energy_col: column with effort or cost or energy or tokens or spend
    time_col:   currently unused, kept for future time-aware variants

    Set clamp_negative_delta=False to allow negative deltas.
    """

    # Legacy aliasing: if requested names are missing but canonical names exist,
    # transparently switch to them.
    if repair_col not in df.columns and "performance" in df.columns:
        repair_col = "performance"
    if energy_col not in df.columns and "energy" in df.columns:
        energy_col = "energy"

    missing = [c for c in (repair_col, energy_col) if c not in df.columns]
    if missing:
        # This message is what Streamlit will show if data really is missing.
        raise ValueError(f"Missing columns: {', '.join(missing)}")

    perf = _coerce_numeric(df[repair_col])
    energy = _coerce_numeric(df[energy_col])

    dperf = np.diff(perf, prepend=perf[:1])
    dperf = np.where(np.isfinite(dperf), dperf, 0.0)
    if clamp_negative_delta:
        dperf = np.maximum(dperf, 0.0)

    denom = np.where(np.isfinite(energy) & (energy > 0.0), energy, energy_floor)
    rye = dperf / denom
    rye = np.where(np.isfinite(rye), rye, 0.0)
    return rye

def compute_rye(
    df: pd.DataFrame,
    repair_col: str = "accuracy",
    energy_col: str = "tokens",
    **kwargs,
) -> np.ndarray:
    """
    Backwards compatible wrapper.

    Older code may call compute_rye(df) expecting accuracy or tokens.
    This wrapper:
      - maps accuracy to performance and tokens to energy when needed
      - raises a clear error only if no suitable columns exist
    """
    # If legacy defaults are missing but canonical names exist, swap.
    if repair_col not in df.columns and "performance" in df.columns:
        repair_col = "performance"
    if energy_col not in df.columns and "energy" in df.columns:
        energy_col = "energy"

    missing = [c for c in (repair_col, energy_col) if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")

    return compute_rye_from_df(df, repair_col=repair_col, energy_col=energy_col, **kwargs)

def compute_rye_cumulative(rye_series: Sequence[float]) -> np.ndarray:
    a = np.asarray(rye_series, dtype=float)
    a[~np.isfinite(a)] = 0.0
    return np.cumsum(a)

# ------------------------------
# Rolling helpers
# ------------------------------
def rolling_series(series: Sequence[float], window: int) -> np.ndarray:
    s = pd.Series(series, dtype=float)
    if window <= 1:
        return s.values
    return s.rolling(window=window, min_periods=1).mean().values

def ema_series(series: Sequence[float], span: int) -> np.ndarray:
    if span is None or span <= 1:
        return np.asarray(series, dtype=float)
    s = pd.Series(series, dtype=float)
    return s.ewm(span=span, adjust=False).mean().values

def recommend_window(n_rows: int, preset_default: Optional[int]) -> int:
    if preset_default and preset_default > 0:
        return int(preset_default)
    if n_rows <= 0:
        return 10
    guess = max(3, min(200, int(round(max(3, n_rows * 0.05)))))
    return guess

# ------------------------------
# Summaries
# ------------------------------
def summarize_series(series: Sequence[float], with_shape: bool = False) -> Dict[str, float]:
    a = np.array(series, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        base = {
            "mean": 0.0,
            "median": 0.0,
            "min": 0.0,
            "max": 0.0,
            "count": 0.0,
            "std": 0.0,
            "resilience": 0.0,
            "p10": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "iqr": 0.0,
            "nonzero_fraction": 0.0,
            "positive_fraction": 0.0,
        }
        if with_shape:
            base.update({"skew": 0.0, "kurtosis": 0.0})
        return base

    mean = float(np.nanmean(a))
    std = float(np.nanstd(a))
    cv = std / (abs(mean) + 1e-9)
    resilience = float(np.clip(1.0 - cv, 0.0, 1.0))
    p10, p50, p90 = np.nanpercentile(a, [10, 50, 90])
    q1, q3 = np.nanpercentile(a, [25, 75])

    # extra fractions used by interpretation and marketing text
    nonzero = float(np.count_nonzero(a != 0.0))
    positive = float(np.count_nonzero(a > 0.0))
    total = float(a.size)

    out = {
        "mean": mean,
        "median": float(np.nanmedian(a)),
        "min": float(np.nanmin(a)),
        "max": float(np.nanmax(a)),
        "count": total,
        "std": std,
        "resilience": resilience,
        "p10": float(p10),
        "p50": float(p50),
        "p90": float(p90),
        "iqr": float(q3 - q1),
        "nonzero_fraction": nonzero / total if total > 0 else 0.0,
        "positive_fraction": positive / total if total > 0 else 0.0,
    }
    if with_shape:
        try:
            from scipy.stats import skew, kurtosis  # type: ignore

            out["skew"] = float(skew(a, nan_policy="omit"))
            out["kurtosis"] = float(kurtosis(a, nan_policy="omit"))
        except Exception:
            out["skew"] = float("nan")
            out["kurtosis"] = float("nan")
    return out

def summarize_by_domain(
    df: pd.DataFrame,
    domain_col: str,
    repair_col: str,
    energy_col: str,
    window: Optional[int] = None,
) -> pd.DataFrame:
    """
    Compute per domain RYE mean and resilience. If window is given, also include rolling mean.
    """
    rows = []
    for dom, sub in df.groupby(domain_col):
        rye = compute_rye_from_df(sub, repair_col=repair_col, energy_col=energy_col)
        rec = {"domain": dom}
        rec.update({f"rye_{k}": v for k, v in summarize_series(rye).items()})
        if window and window > 1:
            rroll = rolling_series(rye, window)
            rec.update({f"rye_roll_{k}": v for k, v in summarize_series(rroll).items()})
        rows.append(rec)
    return pd.DataFrame(rows)

# ------------------------------
# Optional analytics
# ------------------------------
def detect_regimes(
    series: Sequence[float],
    min_len: int = 5,
    gap: float = 0.05,
) -> List[Dict[str, Union[int, str]]]:
    x = np.array(series, dtype=float)
    if x.size == 0:
        return []
    roll = rolling_series(x, max(3, min_len))
    regimes: List[Dict[str, Union[int, str]]] = []
    s = 0
    for i in range(1, len(roll)):
        if abs(roll[i] - roll[i - 1]) > gap:
            if (i - 1) - s + 1 >= min_len:
                mean_seg = float(np.nanmean(x[s:i]))
                regimes.append(
                    {"start": int(s), "end": int(i - 1), "label": f"mean≈{mean_seg:.3f}"}
                )
            s = i
    if len(roll) - s >= min_len:
        mean_seg = float(np.nanmean(x[s:]))
        regimes.append(
            {"start": int(s), "end": int(len(roll) - 1), "label": f"mean≈{mean_seg:.3f}"}
        )
    return regimes

def energy_delta_performance_correlation(
    df: pd.DataFrame,
    perf_col: str,
    energy_col: str,
) -> Dict[str, float]:
    perf = _coerce_numeric(df[perf_col])
    dperf = np.diff(perf, prepend=perf[:1])
    dperf = np.where(np.isfinite(dperf), dperf, 0.0)

    energy = _coerce_numeric(df[energy_col])
    m = np.isfinite(dperf) & np.isfinite(energy)
    if m.sum() < 3:
        return {"pearson": float("nan"), "spearman": float("nan")}

    try:
        from scipy.stats import pearsonr, spearmanr  # type: ignore

        pr = float(pearsonr(energy[m], dperf[m]).statistic)
        sr = float(spearmanr(energy[m], dperf[m]).statistic)
        return {"pearson": pr, "spearman": sr}
    except Exception:
        pass

    try:
        pr = float(np.corrcoef(energy[m], dperf[m])[0, 1])
    except Exception:
        pr = float("nan")
    try:
        def _rank(v):
            order = v.argsort(kind="mergesort")
            ranks = np.empty_like(order, dtype=float)
            ranks[order] = np.arange(1, len(v) + 1, dtype=float)
            return ranks

        r_energy = _rank(energy[m])
        r_dperf = _rank(dperf[m])
        sr = float(np.corrcoef(r_energy, r_dperf)[0, 1])
    except Exception:
        sr = float("nan")
    return {"pearson": pr, "spearman": sr}

def estimate_noise_floor(series: Sequence[float]) -> Dict[str, float]:
    a = np.array(series, dtype=float)
    a = a[np.isfinite(a)]
    if a.size < 3:
        return {"diff_std": float("nan"), "iqr": float("nan")}
    d = np.diff(a)
    diff_std = float(np.nanstd(d))
    q1, q3 = np.nanpercentile(a, [25, 75])
    return {"diff_std": diff_std, "iqr": float(q3 - q1)}

def bootstrap_rolling_mean(
    series: Sequence[float],
    window: int,
    n_boot: int = 100,
    q_low: float = 0.10,
    q_mid: float = 0.50,
    q_high: float = 0.90,
) -> Dict[str, List[float]]:
    x = np.array(series, dtype=float)
    x = np.where(np.isfinite(x), x, 0.0)
    if x.size == 0:
        return {"low": [], "mid": [], "high": []}

    rolls = []
    rng = np.random.default_rng(12345)
    n = len(x)
    for _ in range(max(10, n_boot)):
        idx = rng.integers(0, n, size=n)
        rs = rolling_series(x[idx], max(1, window))
        if len(rs) < n:
            rs = np.pad(
                rs,
                (0, n - len(rs)),
                constant_values=rs[-1] if len(rs) > 0 else 0.0,
            )
        elif len(rs) > n:
            rs = rs[:n]
        rolls.append(rs)

    R = np.vstack(rolls)
    low = np.nanquantile(R, q_low, axis=0)
    mid = np.nanquantile(R, q_mid, axis=0)
    high = np.nanquantile(R, q_high, axis=0)
    return {"low": low.tolist(), "mid": mid.tolist(), "high": high.tolist()}

# ------------------------------
# Outliers
# ------------------------------
def flag_outliers(series: Sequence[float], z: float = 3.0) -> np.ndarray:
    a = np.asarray(series, dtype=float)
    mu = np.nanmean(a)
    sd = np.nanstd(a) + 1e-12
    zscores = (a - mu) / sd
    return np.where(np.abs(zscores) >= z, 1, 0)

# ------------------------------
# Explicit export surface
# ------------------------------
__all__ = [
    "load_table",
    "normalize_columns",
    "infer_columns",
    "COLUMN_ALIASES",
    "DOMAIN_ALIASES",
    "safe_float",
    "compute_rye_from_df",
    "compute_rye",
    "compute_rye_cumulative",
    "rolling_series",
    "ema_series",
    "recommend_window",
    "summarize_series",
    "summarize_by_domain",
    "detect_regimes",
    "energy_delta_performance_correlation",
    "estimate_noise_floor",
    "bootstrap_rolling_mean",
    "flag_outliers",
    "scale_units",
    "PRESETS",
]
