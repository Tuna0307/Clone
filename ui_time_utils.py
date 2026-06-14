"""
UI date/time formatting helpers.

Streamlit returns date and time objects separately. The pipeline expects a
single timestamp string, and an empty string means "do not apply this bound".
"""

from __future__ import annotations

from datetime import date, datetime, time


def format_optional_datetime(enabled: bool, selected_date: date, selected_time: time) -> str:
    """
    Format optional sidebar date/time controls for the pipeline.

    Args:
        enabled: Whether the datetime filter should be applied
        selected_date: Date selected in the UI
        selected_time: Time selected in the UI

    Returns:
        Pipeline-compatible timestamp string, or empty string when disabled
    """
    if not enabled:
        return ""

    selected_datetime = datetime.combine(selected_date, selected_time)
    return selected_datetime.strftime("%Y-%m-%d %H:%M:%S")
