"""Anonymous feedback form and private delivery configuration."""

from __future__ import annotations

import os

import streamlit as st

from src.feedback import (
    FeedbackConfigurationError,
    FeedbackDeliveryError,
    FeedbackSubmission,
    submit_feedback,
)

FEEDBACK_RATINGS = {
    0: "Very difficult",
    1: "Difficult",
    2: "Neutral",
    3: "Useful",
    4: "Very useful",
}

FEEDBACK_REASONS = (
    "Easy to understand",
    "Useful controls",
    "Helpful code examples",
    "Needs clearer explanations",
    "Add more techniques",
    "Something did not work",
)


def _feedback_endpoint() -> str:
    """Read the private delivery endpoint from the host environment."""

    environment_endpoint = os.getenv("FORMSPREE_ENDPOINT", "").strip()
    if environment_endpoint:
        return environment_endpoint
    try:
        return str(st.secrets.get("FORMSPREE_ENDPOINT", "")).strip()
    except FileNotFoundError:
        return ""


def render_feedback_panel(technique: str) -> None:
    """Render the anonymous feedback form and handle one submission."""

    with st.container(border=True):
        st.markdown("### How useful was this tool?")
        st.caption(
            "Let me know what stood out, what could be improved, or what features "
            "you'd like to see next. No account, sign-in, or email is required."
        )
        with st.form("feedback_form", clear_on_submit=True):
            feedback_rating = st.feedback(
                "faces",
                key="feedback_rating",
            )
            feedback_reasons = st.pills(
                "What stood out? (optional)",
                FEEDBACK_REASONS,
                selection_mode="multi",
                key="feedback_reasons",
                help="Choose as many as you like.",
                wrap=True,
            )
            with st.expander("Add a short note or feature idea (optional)"):
                feedback_message = st.text_area(
                    "Anything else I should know?",
                    placeholder=(
                        "For example: add adaptive thresholding or explain Canny "
                        "edges more."
                    ),
                    max_chars=2_000,
                    height=100,
                    label_visibility="collapsed",
                )
            feedback_submitted = st.form_submit_button(
                "Send anonymous feedback",
                type="primary",
            )

        if not feedback_submitted:
            return

        message = feedback_message.strip()
        reasons = tuple(feedback_reasons or ())
        endpoint = _feedback_endpoint()

        if feedback_rating is None and not reasons and not message:
            st.warning(
                "Choose a face, a quick reason, or add a short note before sending."
            )
        elif not endpoint:
            st.error(
                "Sorry, something went wrong with feedback delivery. We are looking "
                "into it."
            )
        else:
            rating_label = FEEDBACK_RATINGS.get(feedback_rating, "Not rated")
            try:
                submit_feedback(
                    endpoint,
                    FeedbackSubmission(
                        category=rating_label,
                        message=message,
                        technique=technique,
                        rating=feedback_rating,
                        highlights=reasons,
                    ),
                )
            except FeedbackConfigurationError:
                st.error("Feedback delivery is temporarily misconfigured.")
            except FeedbackDeliveryError:
                st.error(
                    "Your feedback could not be sent right now. Please try again in "
                    "a moment."
                )
            else:
                st.success("Thank you. your anonymous feedback was sent privately.")
