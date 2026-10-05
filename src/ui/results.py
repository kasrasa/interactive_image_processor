"""Result comparison, learning material, code, sweeps, and histogram views."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.explorer import (
    format_parameter_value,
    image_as_png,
    luminance_histogram,
    sweep_variants,
)
from src.image_utils import image_dimensions, is_grayscale_image, mean_luminance
from src.operation_registry import code_snippet
from src.operations import OperationResult, apply_operation
from src.ui.components import render_technique_heading
from src.ui.sidebar import ProcessingSelection


def _render_image_comparison(
    selection: ProcessingSelection,
    result: OperationResult,
) -> None:
    original_column, result_column = st.columns(2, gap="large")
    with original_column, st.container(border=True):
        st.subheader("Original")
        st.image(selection.source_image, width="stretch")
        input_kind = "Grayscale" if selection.grayscale_mode else "RGB"
        st.caption(f"{input_kind} input · {image_dimensions(selection.source_image)}")

    with result_column, st.container(border=True):
        st.subheader("Result")
        st.image(result.image, width="stretch", clamp=True)
        output_kind = "Grayscale" if is_grayscale_image(result.image) else "RGB"
        st.caption(f"{output_kind} output · {image_dimensions(result.image)}")
        st.download_button(
            "Download result as PNG",
            data=image_as_png(result.image),
            file_name=f"{selection.input_name}-{selection.operation.key}.png",
            mime="image/png",
            width="stretch",
        )


def _render_metrics(
    selection: ProcessingSelection,
    result: OperationResult,
) -> None:
    metric_items = [
        ("Input size", image_dimensions(selection.source_image)),
        ("Output size", image_dimensions(result.image)),
        ("Mean brightness", f"{mean_luminance(result.image):.1f} / 255"),
    ]
    metric_items.extend(result.metrics.items())
    metric_columns = st.columns(len(metric_items))
    for column, (label, value) in zip(metric_columns, metric_items):
        column.metric(label, value)


def _render_learning_tab(selection: ProcessingSelection) -> None:
    operation = selection.operation
    st.markdown(f"### How {operation.name.lower()} works")
    st.write(operation.explanation)
    use_column, warning_column = st.columns(2, gap="large")
    with use_column:
        st.success(f"**Good for**  \n{operation.best_for}")
    with warning_column:
        st.warning(f"**Watch for**  \n{operation.watch_for}")

    st.markdown("#### Current parameters")
    parameter_rows = [
        {
            "Parameter": control.label,
            "Current value": format_parameter_value(selection.parameters[control.key]),
            "What it controls": control.help,
        }
        for control in operation.controls
    ]
    st.dataframe(parameter_rows, hide_index=True, width="stretch")


def _render_code_tab(selection: ProcessingSelection) -> None:
    st.markdown("### Reproduce this result")
    st.caption(
        "The snippet assumes `image_rgb` has already been loaded as an RGB uint8 "
        "NumPy array."
    )
    st.code(
        code_snippet(
            selection.operation.key,
            selection.parameters,
            grayscale_input=selection.grayscale_mode,
        ),
        language="python",
        line_numbers=True,
    )
    st.markdown(
        """
        <div class="lab-callout">
            OpenCV often reads files in BGR order. If you used <code>cv2.imread</code>, convert once with
            <code>cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)</code> before using this snippet.
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_comparison_tab(selection: ProcessingSelection) -> None:
    comparison = sweep_variants(selection.operation, selection.parameters)
    if comparison is None:
        st.info(
            "This technique uses a categorical or automatic choice. Change its sidebar "
            "setting and compare it directly with the original above."
        )
        return

    sweep_control, variants = comparison
    st.markdown(f"### Sweep: {sweep_control.label}")
    st.caption(
        "The middle view is your current setting; the neighboring views show a lower "
        "and higher value while every other parameter stays fixed."
    )
    comparison_columns = st.columns(len(variants), gap="medium")
    for column, (value_label, variant_parameters) in zip(comparison_columns, variants):
        variant_result = apply_operation(
            selection.operation.key,
            selection.source_image,
            variant_parameters,
        )
        with column:
            st.image(variant_result.image, width="stretch", clamp=True)
            st.markdown(f"**{sweep_control.label}: {value_label}**")


def _render_histogram_tab(
    selection: ProcessingSelection,
    result: OperationResult,
    dark_mode: bool,
) -> None:
    st.markdown("### Luminance distribution")
    st.caption(
        "Each line shows the fraction of pixels at every brightness level from "
        "black (0) to white (255)."
    )
    histogram_data = pd.DataFrame(
        {
            "Original": luminance_histogram(selection.source_image),
            "Result": luminance_histogram(result.image),
        }
    )
    histogram_data.index.name = "Brightness"
    chart_colors = ["#94a3b8", "#a3e635"] if dark_mode else ["#64748b", "#65a30d"]
    st.line_chart(histogram_data, color=chart_colors)


def render_results(
    selection: ProcessingSelection,
    result: OperationResult,
    dark_mode: bool,
) -> None:
    """Render all output and educational views for the selected operation."""

    render_technique_heading(selection.category, selection.operation)
    _render_image_comparison(selection, result)
    _render_metrics(selection, result)

    learn_tab, code_tab, compare_tab, histogram_tab = st.tabs(
        ("Learn", "Copy the code", "Compare settings", "Histogram")
    )
    with learn_tab:
        _render_learning_tab(selection)
    with code_tab:
        _render_code_tab(selection)
    with compare_tab:
        _render_comparison_tab(selection)
    with histogram_tab:
        _render_histogram_tab(selection, result, dark_mode)
