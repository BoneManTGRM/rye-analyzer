# presets.py
from dataclasses import dataclass

@dataclass(frozen=True)
class Preset:
    name: str
    default_window: int
    min_window: int
    max_window: int
    plot_title: str
    y_label: str
    series_label: str = "RYE"
    rolling_label: str = "RYE rolling"
    interpretation_hint: str = ""

PRESETS = {
    "AI": Preset(
        name="AI",
        default_window=100, min_window=20, max_window=500,
        plot_title="Repair Yield per Energy",
        y_label="RYE",
        interpretation_hint=(
            "AI preset: window tuned for iteration-scale smoothing (50–200 typical). "
            "Use for training runs, eval sweeps, or online adaptation."
        ),
    ),
    "Biology": Preset(
        name="Biology",
        default_window=10, min_window=3, max_window=30,
        plot_title="Repair Yield per Energy (Biological time-course)",
        y_label="RYE",
        interpretation_hint=(
            "Biology preset: window sized for short time-course assays (5–15 typical). "
            "Use for recovery curves, stress-response, or clinical markers."
        ),
    ),
    "Robotics": Preset(
        name="Robotics",
        default_window=30, min_window=10, max_window=120,
        plot_title="Repair Yield per Energy (Robotics/Systems)",
        y_label="RYE",
        interpretation_hint=(
            "Robotics preset: window sized for cycles/episodes (20–60 typical). "
            "Good for actuator wear, task success, or maintenance intervals."
        ),
    ),
}

DEFAULT_PRESET = PRESETS["AI"]

def get_preset(name: str) -> Preset:
    return PRESETS.get(name, DEFAULT_PRESET)
