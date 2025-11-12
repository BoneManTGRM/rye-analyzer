from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass(frozen=True)
class Preset:
    name: str
    time: List[str]
    performance: List[str]
    energy: List[str]
    domain: Optional[str] = None
    default_rolling: int = 10

# Helper to keep lists short but useful
def _kw(*items):
    return list(dict.fromkeys([s.strip() for s in items if s]))

PRESETS: Dict[str, Preset] = {
    # --- Core set you already had ---
    "AI": Preset(
        "AI",
        time=_kw("step", "iteration", "epoch", "t", "time"),
        performance=_kw("accuracy", "acc", "f1", "reward", "score", "coherence", "loss_inv", "bleu", "rouge"),
        energy=_kw("tokens", "compute", "energy", "cost", "gradient_updates", "lr", "batch_tokens"),
        domain="ai",
    ),
    "Biology": Preset(
        "Biology",
        time=_kw("time", "t", "hours", "days", "samples"),
        performance=_kw("viability", "function", "yield", "recovery", "signal", "od", "growth", "fitness"),
        energy=_kw("dose", "stressor", "input", "energy", "treatment", "drug", "radiation"),
        domain="bio",
    ),
    "Robotics": Preset(
        "Robotics",
        time=_kw("t", "time", "cycle", "episode"),
        performance=_kw("task_success", "score", "stability", "tracking_inv", "uptime", "mean_reward"),
        energy=_kw("power", "torque_int", "battery_used", "energy", "effort", "cpu_load"),
        domain="robot",
    ),

    # --- Engineering / Ops ---
    "Manufacturing": Preset(
        "Manufacturing",
        time=_kw("time", "shift", "batch", "lot", "t"),
        performance=_kw("yield", "throughput", "quality", "uptime", "oee", "scrap_inv"),
        energy=_kw("power", "energy", "rework", "downtime", "input_cost", "labor_hours"),
        domain="mfg",
    ),
    "DevOps/SRE": Preset(
        "DevOps/SRE",
        time=_kw("time", "t", "minute", "hour", "deploy", "build"),
        performance=_kw("uptime", "slo_attained", "latency_inv", "error_rate_inv", "success_rate"),
        energy=_kw("cpu", "mem", "cost", "requests", "writes", "build_minutes"),
        domain="sre",
    ),
    "Networking": Preset(
        "Networking",
        time=_kw("time", "t", "slot", "interval"),
        performance=_kw("throughput", "goodput", "availability", "pkt_success", "latency_inv"),
        energy=_kw("tx_power", "bandwidth", "retries", "cost", "hops"),
        domain="net",
    ),
    "Aerospace": Preset(
        "Aerospace",
        time=_kw("time", "t", "cycle", "flight"),
        performance=_kw("mission_success", "stability", "tracking_inv", "fault_free_time"),
        energy=_kw("fuel", "power", "thrust_effort", "thermal_load"),
        domain="aero",
    ),
    "Materials": Preset(
        "Materials",
        time=_kw("time", "cycle", "step"),
        performance=_kw("strength", "conductivity", "hardness", "toughness", "yield_strength"),
        energy=_kw("temp", "load", "stress", "dose", "cycles"),
        domain="mat",
    ),

    # --- Science / Lab ---
    "Chemistry": Preset(
        "Chemistry",
        time=_kw("time", "t", "minutes", "hours"),
        performance=_kw("yield", "purity", "selectivity", "signal", "conversion"),
        energy=_kw("reagent", "dose", "temp", "pressure", "energy"),
        domain="chem",
    ),
    "Physics Lab": Preset(
        "Physics Lab",
        time=_kw("time", "t", "run", "shot"),
        performance=_kw("signal", "stability", "coherence", "snr", "resolution"),
        energy=_kw("power", "fluence", "current", "shots", "cost"),
        domain="phys",
    ),
    "Neuroscience": Preset(
        "Neuroscience",
        time=_kw("time", "trial", "t"),
        performance=_kw("accuracy", "auc", "firing_sync", "connectivity", "score"),
        energy=_kw("stimulus", "dose", "current", "effort", "trials"),
        domain="neuro",
    ),
    "Clinical Trials": Preset(
        "Clinical Trials",
        time=_kw("visit", "day", "week", "time"),
        performance=_kw("response", "remission", "score", "survival_inv", "function"),
        energy=_kw("dose", "sessions", "adherence", "cost", "exposure"),
        domain="ct",
    ),
    "Healthcare": Preset(
        "Healthcare",
        time=_kw("time", "visit", "day", "t"),
        performance=_kw("recovery", "response", "function", "score", "readmit_inv"),
        energy=_kw("dose", "sessions", "cost", "effort", "treatment"),
        domain="hc",
    ),
    "Epidemiology/Public Health": Preset(
        "Epidemiology/Public Health",
        time=_kw("date", "week", "time"),
        performance=_kw("rt_inv", "incidence_inv", "coverage", "response_rate"),
        energy=_kw("doses", "tests", "cost", "interventions"),
        domain="epi",
    ),

    # --- Energy / Industry / Infra ---
    "Energy/Grid": Preset(
        "Energy/Grid",
        time=_kw("time", "interval", "t"),
        performance=_kw("uptime", "availability", "efficiency", "output", "losses_inv"),
        energy=_kw("load", "consumption", "cost", "curtailment", "effort"),
        domain="grid",
    ),
    "Battery/EV": Preset(
        "Battery/EV",
        time=_kw("cycle", "time", "t"),
        performance=_kw("capacity_retained", "soh", "range", "efficiency"),
        energy=_kw("charge", "discharge", "power", "c_rate", "temp"),
        domain="ev",
    ),
    "Oil & Gas": Preset(
        "Oil & Gas",
        time=_kw("time", "day", "shift"),
        performance=_kw("production", "throughput", "uptime", "quality"),
        energy=_kw("power", "gas_injected", "steam", "cost", "water_cut"),
        domain="oag",
    ),
    "Water/Wastewater": Preset(
        "Water/Wastewater",
        time=_kw("time", "day", "interval"),
        performance=_kw("removal_eff", "compliance", "uptime", "quality"),
        energy=_kw("flow", "power", "chem_dose", "cost"),
        domain="water",
    ),

    # --- Business / Data ---
    "Finance": Preset(
        "Finance",
        time=_kw("date", "time", "t", "bar", "index"),
        performance=_kw("return", "pnl", "sharpe", "alpha", "win_rate"),
        energy=_kw("risk", "drawdown", "exposure", "volume", "cost"),
        domain="fin",
    ),
    "Economics/Macro": Preset(
        "Economics/Macro",
        time=_kw("date", "quarter", "month", "t"),
        performance=_kw("growth", "employment", "inflation_inv", "output"),
        energy=_kw("spend", "rates", "debt", "cost"),
        domain="econ",
    ),

    # --- Marketing (enhanced) ---
    "Marketing": Preset(
        "Marketing",
        # typical time axes for campaigns and reports
        time=_kw(
            "date", "day", "week", "month", "time", "t",
            "campaign_start", "campaign_end", "timestamp"
        ),
        # "performance" = outcomes / success: higher is better
        performance=_kw(
            "conversion_rate", "conv_rate", "cvr",
            "ctr", "click_through_rate",
            "roas", "rpc", "revenue", "sales",
            "signups", "leads", "purchases", "orders",
            "subscriptions", "new_customers", "retention",
            "engagement", "engagement_rate",
            "open_rate", "reply_rate",
            "bounce_rate_inv", "unsub_rate_inv",
            "ltv", "clv"
        ),
        # "energy" = effort / cost / exposure spent to get those outcomes
        energy=_kw(
            "spend", "ad_spend", "budget", "media_spend", "cost",
            "cpc_denom", "cpa_denom", "cpv_denom",
            "impressions", "imps", "reach", "views",
            "sessions", "visits", "traffic",
            "clicks",
            "emails_sent", "sms_sent", "messages_sent",
            "touches"
        ),
        domain="mkt",
    ),

    "Sales/CRM": Preset(
        "Sales/CRM",
        time=_kw("date", "stage_time", "time"),
        performance=_kw("close_rate", "revenue", "aov", "win_rate"),
        energy=_kw("leads", "calls", "emails", "meetings", "spend"),
        domain="sales",
    ),
    "Web Analytics": Preset(
        "Web Analytics",
        time=_kw("date", "time", "t"),
        performance=_kw("conversion", "retention", "latency_inv", "availability"),
        energy=_kw("traffic", "requests", "cost", "build_minutes"),
        domain="web",
    ),
    "E-commerce": Preset(
        "E-commerce",
        time=_kw("date", "time", "week"),
        performance=_kw("gmv", "orders", "aov", "repeat_rate"),
        energy=_kw("ad_spend", "discount", "inventory", "shipping_cost"),
        domain="ecom",
    ),
    "Customer Support": Preset(
        "Customer Support",
        time=_kw("date", "time", "t"),
        performance=_kw("csat", "fst_inv", "nps", "resolution_rate"),
        energy=_kw("tickets", "agent_hours", "escalations", "cost"),
        domain="cs",
    ),

    # --- Supply / Mobility ---
    "Supply Chain": Preset(
        "Supply Chain",
        time=_kw("date", "time", "week", "month"),
        performance=_kw("on_time_rate", "otif", "fill_rate", "service_level", "inventory_turns"),
        energy=_kw("orders", "shipments", "miles", "transport_cost", "handling_cost"),
        domain="supply",
    ),
}
