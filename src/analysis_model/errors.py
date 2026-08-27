"""User-facing analysis failures (shown in Streamlit without a stack trace)."""


class AnalysisError(Exception):
    """Short, safe message for invalid tickers, bad API keys, or empty statements."""
