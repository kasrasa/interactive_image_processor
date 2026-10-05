"""Page configuration and light/dark visual theme setup."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

_STYLESHEET = Path(__file__).resolve().parents[2] / "assets" / "styles.css"

_DARK_TOKENS = """
    :root {
        --lab-bg: #0b1120;
        --lab-color-scheme: dark;
        --lab-sidebar-bg: #111827;
        --lab-surface: #111827;
        --lab-surface-soft: #182234;
        --lab-control-bg: #172033;
        --lab-header-bg: rgba(11, 17, 32, 0.92);
        --lab-ink: #f8fafc;
        --lab-muted: #cbd5e1;
        --lab-accent: #a3e635;
        --lab-accent-bright: #a3e635;
        --lab-accent-soft: #365314;
        --lab-line: rgba(203, 213, 225, 0.22);
        --lab-hero-glow: rgba(163, 230, 53, 0.20);
        --lab-hero-start: rgba(255, 255, 255, 0.035);
        --lab-hero-end: rgba(77, 124, 15, 0.22);
        --lab-slogan: #e5e7eb;
        --lab-badge-bg: rgba(15, 23, 42, 0.88);
        --lab-badge-text: #d0d5dd;
        --lab-metric-bg: rgba(17, 24, 39, 0.84);
        --lab-callout-bg: rgba(54, 83, 20, 0.36);
        --lab-callout-border: rgba(163, 230, 53, 0.28);
    }
"""

_LIGHT_TOKENS = """
    :root {
        --lab-bg: #fbfcfe;
        --lab-color-scheme: light;
        --lab-sidebar-bg: #f1f5f9;
        --lab-surface: #ffffff;
        --lab-surface-soft: #f8fafc;
        --lab-control-bg: #ffffff;
        --lab-header-bg: rgba(251, 252, 254, 0.92);
        --lab-ink: #0b1220;
        --lab-muted: #475467;
        --lab-accent: #4d7c0f;
        --lab-accent-bright: #a3e635;
        --lab-accent-soft: #ecfccb;
        --lab-line: rgba(71, 84, 103, 0.22);
        --lab-hero-glow: rgba(163, 230, 53, 0.25);
        --lab-hero-start: rgba(15, 23, 42, 0.06);
        --lab-hero-end: rgba(132, 204, 22, 0.11);
        --lab-slogan: #344054;
        --lab-badge-bg: rgba(255, 255, 255, 0.76);
        --lab-badge-text: #344054;
        --lab-metric-bg: rgba(248, 250, 252, 0.72);
        --lab-callout-bg: rgba(236, 252, 203, 0.46);
        --lab-callout-border: rgba(77, 124, 15, 0.20);
    }
"""


def configure_page() -> None:
    """Set browser metadata and the initial Streamlit layout."""

    st.set_page_config(
        page_title="Interactive Image Processor | Kasra Sadatsharifi",
        page_icon="◉",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def render_theme_selector() -> bool:
    """Render the appearance control, apply its CSS, and return dark-mode state."""

    st.sidebar.markdown("## Experiment setup")
    selected_theme = st.sidebar.segmented_control(
        "Appearance",
        ("Light", "Dark"),
        default="Light",
        key="appearance_theme",
        help="Switch the entire application between light and dark mode.",
        width="stretch",
    )
    dark_mode = selected_theme == "Dark"
    theme_tokens = _DARK_TOKENS if dark_mode else _LIGHT_TOKENS
    stylesheet = _STYLESHEET.read_text(encoding="utf-8")
    st.markdown(
        f"<style>{theme_tokens}{stylesheet}</style>",
        unsafe_allow_html=True,
    )
    return dark_mode
