"""Lightweight repository validation.

This test can run before Streamlit is installed because it stubs the small
Streamlit surface used at import time. It validates the CSV, core scoring, and map.
"""
from __future__ import annotations

import datetime as dt
import sys
import types

import pandas as pd


streamlit_stub = types.ModuleType("streamlit")


def cache_data(*args, **kwargs):
    def decorator(func):
        return func
    return decorator


streamlit_stub.cache_data = cache_data
streamlit_stub.error = lambda *args, **kwargs: None
sys.modules.setdefault("streamlit", streamlit_stub)

streamlit_folium_stub = types.ModuleType("streamlit_folium")
streamlit_folium_stub.st_folium = lambda *args, **kwargs: None
sys.modules.setdefault("streamlit_folium", streamlit_folium_stub)

import app  # noqa: E402


def main() -> None:
    assert app.APP_NAME == "AI Education Center Finder — Uzbekistan"
    assert app.AUTHOR_NAME == "Mukhammadjon Khojikulov"
    assert app.MENTOR_NAME == "Dr. Qingyang Xiao"

    centers = app.load_centers()
    assert len(centers) == 28
    assert set(centers["country"].unique()) == {"Uzbekistan"}
    assert centers["region"].nunique() == 14

    enriched = app.add_computed_columns(
        centers,
        selected_date=dt.date(2026, 8, 3),
        selected_time=dt.time(16, 0),
        grade=9,
        age=15,
        has_membership=False,
        user_lat=41.3111,
        user_lon=69.2797,
        commute_mode="Public transport",
    )
    ranked = app.recommend_centers(
        enriched,
        role="Student / O‘quvchi",
        selected_subjects=["Mathematics", "Python"],
        selected_resources=["Computer lab"],
        center_types=[],
        goal_text="Matematika va dasturlash o‘rganmoqchiman",
        preferred_languages=["Uzbek"],
    )

    assert not ranked.empty
    assert ranked["ai_match_score"].between(0, 100).all()
    assert ranked["distance_km"].notna().all()
    assert ranked["commute_minutes"].notna().all()

    fmap = app.build_map(ranked.head(5), 41.3111, 69.2797)
    html = fmap.get_root().render()
    assert isinstance(html, str)
    assert "leaflet" in html.lower()
    assert len(html) > 1000

    print("PASS: Uzbekistan-only dataset, core recommendation logic, and map generation")
    print(ranked[["name", "region", "ai_match_score"]].head(3).to_string(index=False))


if __name__ == "__main__":
    main()
