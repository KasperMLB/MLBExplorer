from __future__ import annotations

import streamlit as st

from .backtesting_view import render_backtesting_tab
from .branding import page_icon_path
from .config import AppConfig


def main() -> None:
    st.set_page_config(page_title="Backtesting", page_icon=page_icon_path(), layout="wide")
    render_backtesting_tab(AppConfig())


if __name__ == "__main__":
    main()
