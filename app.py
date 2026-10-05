"""Interactive Streamlit interface for exploring classical OpenCV techniques."""

from __future__ import annotations

import cv2
import streamlit as st

from src.operations import OperationResult, apply_operation
from src.ui.components import render_footer, render_hero
from src.ui.feedback_panel import render_feedback_panel
from src.ui.results import render_results
from src.ui.sidebar import render_sidebar
from src.ui.theme import configure_page, render_theme_selector


def main() -> None:
    """Compose the app from focused sidebar, result, and feedback views."""

    configure_page()
    dark_mode = render_theme_selector()
    render_hero()
    selection = render_sidebar()

    try:
        result: OperationResult = apply_operation(
            selection.operation.key,
            selection.source_image,
            selection.parameters,
        )
    except (cv2.error, KeyError, TypeError, ValueError) as error:
        st.error(f"The operation could not be rendered: {error}")
        st.stop()

    render_results(selection, result, dark_mode)
    render_feedback_panel(f"{selection.category} / {selection.operation.name}")
    render_footer()


if __name__ == "__main__":
    main()
