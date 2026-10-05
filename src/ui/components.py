"""Reusable static presentation components for the application page."""

from __future__ import annotations

import streamlit as st

from src.operation_registry import OperationSpec


def render_hero() -> None:
    """Render the project introduction banner."""

    st.markdown(
        """
        <div class="lab-hero">
            <div class="lab-eyebrow">Kasra Sadatsharifi · Interactive computer vision project</div>
            <h1>Interactive Image Processor</h1>
            <p>
                See what each parameter changes, compare settings side by side, and copy the
                exact Python behind the result. Start with the built-in test card or upload your own image.
            </p>
            <div class="lab-slogan">Research mindset. Production habits.</div>
            <div class="lab-powered">
                Created by <strong>Kasra Sadatsharifi</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_technique_heading(category: str, operation: OperationSpec) -> None:
    """Render the selected technique name and summary."""

    st.markdown(
        f"""
        <div class="lab-technique">
            <strong>{category} / {operation.name}</strong>
            <span>{operation.summary}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    """Render the personal project attribution."""

    st.markdown(
        """
        <section class="project-footer" aria-label="About the creator">
            <p>
                Created as an independent computer-vision learning project by
                <a
                    href="https://github.com/kasrasa"
                    target="_blank"
                    rel="noopener noreferrer"
                >Kasra Sadatsharifi</a>.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )
