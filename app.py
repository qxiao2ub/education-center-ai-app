# app.py
# AI-Based Education Center Cross-Function Website App
# Run locally: streamlit run app.py

from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path
from typing import Any, Iterable

import folium
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_folium import st_folium

DATA_PATH = Path("education_centers_seed.csv")
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
CENTER_TYPES = [
    "Tutoring Center",
    "STEM / Makerspace",
    "Community Center",
    "Language / ESL Center",
    "Museum Education Center",
    "College Prep Center",
    "Library Learning Studio",
    "Robotics Club",
    "Math Enrichment Center",
    "Technology Academy",
    "Career / Coding Center",
    "Homework Help Center",
    "Research Skills Center",
    "Writing Center",
    "Environmental Education Center",
    "Science Discovery Center",
    "Family Literacy Center",
    "Teacher Resource Center",
]

SPEED_KMPH = {
    "Walk": 5,
    "Bike": 15,
    "Transit": 24,
    "Drive": 40,
}
MODE_BUFFER_MINUTES = {
    "Walk": 2,
    "Bike": 4,
    "Transit": 12,
    "Drive": 6,
}


def safe_json_loads(value: Any, default: Any) -> Any:
    """Safely parse a JSON string and return default when parsing fails."""
    if value is None:
        return default
    if isinstance(value, (list, dict)):
        return value
    if isinstance(value, float) and np.isnan(value):
        return default
    try:
        return json.loads(str(value))
    except Exception:
        return default


@st.cache_data(show_spinner=False)
def load_centers(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Load center data from CSV and normalize common fields."""
    path = Path(path)
    if not path.exists():
        st.error(
            "Missing education_centers_seed.csv. Run the setup notebook cell first or upload your own dataset."
        )
        return pd.DataFrame()
    df = pd.read_csv(path)
    for col in ["subjects", "resources", "utilities", "affiliations", "languages", "description"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
    bool_map = {"yes": True, "no": False, "true": True, "false": False, "1": True, "0": False}
    df["membership_required_bool"] = (
        df.get("membership_required", "No").fillna("No").astype(str).str.lower().map(bool_map).fillna(False)
    )
    return df


def split_pipe(value: Any) -> list[str]:
    """Split pipe-delimited strings into clean unique terms."""
    if value is None:
        return []
    return [v.strip() for v in str(value).split("|") if v.strip()]


def unique_terms(df: pd.DataFrame, column: str) -> list[str]:
    terms: set[str] = set()
    if column not in df.columns:
        return []
    for value in df[column].fillna(""):
        terms.update(split_pipe(value))
    return sorted(terms)


def parse_time(value: str | dt.time) -> dt.time | None:
    if isinstance(value, dt.time):
        return value
    try:
        return dt.datetime.strptime(str(value), "%H:%M").time()
    except Exception:
        return None


def time_in_slot(selected_time: dt.time, start_text: str, end_text: str) -> bool:
    start = parse_time(start_text)
    end = parse_time(end_text)
    if start is None or end is None:
        return False
    # Normal same-day opening hours.
    if start <= end:
        return start <= selected_time <= end
    # Overnight slot such as 22:00-02:00.
    return selected_time >= start or selected_time <= end


def slots_for_day(row: pd.Series, selected_date: dt.date) -> list[dict[str, str]]:
    schedule = safe_json_loads(row.get("open_schedule_json"), [])
    day_name = selected_date.strftime("%A")
    return [slot for slot in schedule if slot.get("day") == day_name]


def is_open_at(row: pd.Series, selected_date: dt.date, selected_time: dt.time) -> bool:
    return any(time_in_slot(selected_time, slot.get("start", ""), slot.get("end", "")) for slot in slots_for_day(row, selected_date))


def hours_for_day(row: pd.Series, selected_date: dt.date) -> str:
    slots = slots_for_day(row, selected_date)
    if not slots:
        return "Closed"
    return ", ".join([f"{slot.get('start')} - {slot.get('end')}" for slot in slots])


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two coordinates in kilometers."""
    r = 6371.0
    phi1 = math.radians(float(lat1))
    phi2 = math.radians(float(lat2))
    d_phi = math.radians(float(lat2) - float(lat1))
    d_lambda = math.radians(float(lon2) - float(lon1))
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def estimate_commute_minutes(distance_km: float | None, mode: str) -> float | None:
    if distance_km is None or pd.isna(distance_km):
        return None
    speed = SPEED_KMPH.get(mode, 24)
    buffer = MODE_BUFFER_MINUTES.get(mode, 8)
    return round((float(distance_km) / speed) * 60 + buffer, 1)


def eligibility_status(row: pd.Series, grade: int, age: int, has_membership: bool) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    min_grade, max_grade = int(row.get("min_grade", 0)), int(row.get("max_grade", 12))
    min_age, max_age = int(row.get("min_age", 0)), int(row.get("max_age", 99))
    membership_required = bool(row.get("membership_required_bool", False))

    if not (min_grade <= grade <= max_grade):
        reasons.append(f"grade {grade} outside supported grade range {min_grade}-{max_grade}")
    if not (min_age <= age <= max_age):
        reasons.append(f"age {age} outside supported age range {min_age}-{max_age}")
    if membership_required and not has_membership:
        reasons.append("membership or referral may be required")
    return len(reasons) == 0, reasons


def text_blob(row: pd.Series) -> str:
    pieces = [
        row.get("name", ""),
        row.get("center_type", ""),
        row.get("city", ""),
        row.get("subjects", ""),
        row.get("resources", ""),
        row.get("utilities", ""),
        row.get("affiliations", ""),
        row.get("languages", ""),
        row.get("description", ""),
        row.get("eligibility_notes", ""),
    ]
    events = safe_json_loads(row.get("events_json"), [])
    for item in events:
        pieces.append(str(item.get("title", "")))
        pieces.append(str(item.get("audience", "")))
    return " ".join(str(p) for p in pieces if p)


def cost_score(cost_level: str) -> float:
    mapping = {
        "Free": 1.0,
        "Low-cost": 0.85,
        "Moderate": 0.62,
        "Paid": 0.42,
    }
    return mapping.get(str(cost_level), 0.6)


def build_user_query(
    role: str,
    selected_subjects: Iterable[str],
    selected_resources: Iterable[str],
    center_types: Iterable[str],
    goal_text: str,
) -> str:
    parts = [role, goal_text]
    parts.extend(list(selected_subjects))
    parts.extend(list(selected_resources))
    parts.extend(list(center_types))
    query = " ".join([p for p in parts if str(p).strip()])
    return query or "learning homework tutoring STEM reading writing college readiness"


def add_computed_columns(
    df: pd.DataFrame,
    selected_date: dt.date,
    selected_time: dt.time,
    grade: int,
    age: int,
    has_membership: bool,
    user_lat: float | None,
    user_lon: float | None,
    commute_mode: str,
) -> pd.DataFrame:
    df = df.copy()
    df["open_selected_time"] = df.apply(lambda row: is_open_at(row, selected_date, selected_time), axis=1)
    df["hours_today"] = df.apply(lambda row: hours_for_day(row, selected_date), axis=1)

    eligibilities = df.apply(lambda row: eligibility_status(row, grade, age, has_membership), axis=1)
    df["eligible"] = [status for status, _ in eligibilities]
    df["eligibility_reasons"] = ["; ".join(reasons) if reasons else "Eligible based on grade, age, and membership inputs" for _, reasons in eligibilities]

    if user_lat is not None and user_lon is not None:
        df["distance_km"] = df.apply(lambda row: haversine_km(user_lat, user_lon, row["latitude"], row["longitude"]), axis=1)
        df["commute_minutes"] = df["distance_km"].apply(lambda d: estimate_commute_minutes(d, commute_mode))
    else:
        df["distance_km"] = np.nan
        df["commute_minutes"] = np.nan
    return df


def recommend_centers(
    df: pd.DataFrame,
    role: str,
    selected_subjects: list[str],
    selected_resources: list[str],
    center_types: list[str],
    goal_text: str,
) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    df = df.copy()
    query = build_user_query(role, selected_subjects, selected_resources, center_types, goal_text)
    documents = df.apply(text_blob, axis=1).tolist() + [query]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    matrix = vectorizer.fit_transform(documents)
    similarities = cosine_similarity(matrix[:-1], matrix[-1]).ravel()

    if np.max(similarities) > 0:
        content_scores = similarities / np.max(similarities)
    else:
        content_scores = np.zeros_like(similarities)

    eligible_scores = df["eligible"].astype(float).to_numpy() if "eligible" in df.columns else np.ones(len(df))
    open_scores = df["open_selected_time"].astype(float).to_numpy() if "open_selected_time" in df.columns else np.ones(len(df))
    cost_scores = df["cost_level"].apply(cost_score).to_numpy()

    if "distance_km" in df.columns and df["distance_km"].notna().any():
        distance_scores = 1 / (1 + df["distance_km"].fillna(df["distance_km"].median()) / 8)
        distance_scores = distance_scores.to_numpy()
    else:
        distance_scores = np.full(len(df), 0.65)

    final = (
        0.42 * content_scores
        + 0.18 * eligible_scores
        + 0.16 * open_scores
        + 0.14 * distance_scores
        + 0.10 * cost_scores
    )
    df["ai_match_score"] = np.round(final * 100, 1)
    df["content_match"] = np.round(content_scores * 100, 1)
    df["ai_reason"] = df.apply(lambda row: build_reason(row, selected_subjects, selected_resources), axis=1)
    return df.sort_values(["ai_match_score", "open_selected_time", "eligible"], ascending=[False, False, False])


def build_reason(row: pd.Series, selected_subjects: list[str], selected_resources: list[str]) -> str:
    row_terms = set(split_pipe(row.get("subjects", "")) + split_pipe(row.get("resources", "")))
    selected = selected_subjects + selected_resources
    matches = [term for term in selected if term in row_terms]
    parts = []
    if matches:
        parts.append("matches " + ", ".join(matches[:4]))
    if row.get("open_selected_time"):
        parts.append("open at selected time")
    if row.get("eligible"):
        parts.append("eligible for your profile")
    if pd.notna(row.get("commute_minutes", np.nan)):
        parts.append(f"about {row.get('commute_minutes')} min by selected mode")
    return "; ".join(parts) or "broad match based on center profile and user goals"


def filter_centers(
    df: pd.DataFrame,
    country: str,
    regions: list[str],
    cities: list[str],
    center_types: list[str],
    open_only: bool,
    eligible_only: bool,
    max_commute: int,
    keyword: str,
) -> pd.DataFrame:
    results = df.copy()
    if country != "All":
        results = results[results["country"] == country]
    if regions:
        results = results[results["region"].isin(regions)]
    if cities:
        results = results[results["city"].isin(cities)]
    if center_types:
        results = results[results["center_type"].isin(center_types)]
    if open_only:
        results = results[results["open_selected_time"]]
    if eligible_only:
        results = results[results["eligible"]]
    if "commute_minutes" in results.columns and results["commute_minutes"].notna().any():
        results = results[results["commute_minutes"] <= max_commute]
    if keyword.strip():
        key = keyword.lower().strip()
        mask = results.apply(lambda row: key in text_blob(row).lower(), axis=1)
        results = results[mask]
    return results


def build_map(df: pd.DataFrame, user_lat: float | None, user_lon: float | None) -> folium.Map:
    if user_lat is not None and user_lon is not None:
        start = [user_lat, user_lon]
    elif not df.empty:
        start = [df["latitude"].mean(), df["longitude"].mean()]
    else:
        start = [39.0, -76.8]

    fmap = folium.Map(location=start, zoom_start=9, control_scale=True)
    if user_lat is not None and user_lon is not None:
        folium.Marker(
            location=[user_lat, user_lon],
            popup="Your selected location",
            tooltip="Your location",
            icon=folium.Icon(color="blue", icon="home"),
        ).add_to(fmap)

    for _, row in df.iterrows():
        status = "Open" if row.get("open_selected_time") else "Closed at selected time"
        eligible = "Eligible" if row.get("eligible") else "Check eligibility"
        color = "green" if row.get("open_selected_time") and row.get("eligible") else "orange"
        popup_html = f"""
        <b>{row.get('name')}</b><br>
        {row.get('center_type')}<br>
        {row.get('city')}, {row.get('region')}<br>
        Hours: {row.get('hours_today')}<br>
        Status: {status}<br>
        Eligibility: {eligible}<br>
        AI match: {row.get('ai_match_score', 'N/A')}<br>
        """
        folium.Marker(
            location=[row["latitude"], row["longitude"]],
            popup=folium.Popup(popup_html, max_width=320),
            tooltip=row.get("name"),
            icon=folium.Icon(color=color, icon="info-sign"),
        ).add_to(fmap)
    return fmap


def render_events(row: pd.Series) -> None:
    events = safe_json_loads(row.get("events_json"), [])
    if not events:
        st.write("No upcoming events listed.")
        return
    event_df = pd.DataFrame(events)
    st.dataframe(event_df, use_container_width=True, hide_index=True)


def render_center_card(row: pd.Series) -> None:
    header = f"{row.get('name')} - {row.get('ai_match_score', 'N/A')}% AI match"
    with st.expander(header, expanded=False):
        left, right = st.columns([1.35, 1])
        with left:
            st.markdown(f"**Type:** {row.get('center_type')}")
            st.markdown(f"**Location:** {row.get('address')}")
            st.markdown(f"**Open hours on selected day:** {row.get('hours_today')}")
            st.markdown(f"**Status:** {'Open at selected time' if row.get('open_selected_time') else 'Closed at selected time'}")
            st.markdown(f"**Eligibility:** {'Eligible' if row.get('eligible') else 'Needs review'}")
            st.markdown(f"**Eligibility notes:** {row.get('eligibility_notes')}")
            st.markdown(f"**AI reason:** {row.get('ai_reason')}")
            if pd.notna(row.get("commute_minutes", np.nan)):
                st.markdown(f"**Estimated commute:** {row.get('commute_minutes')} minutes ({row.get('distance_km'):.1f} km)")
            st.markdown(f"**Description:** {row.get('description')}")
        with right:
            st.markdown("**Subjects**")
            st.write(", ".join(split_pipe(row.get("subjects", ""))))
            st.markdown("**Resources**")
            st.write(", ".join(split_pipe(row.get("resources", ""))))
            st.markdown("**Utilities**")
            st.write(", ".join(split_pipe(row.get("utilities", ""))))
            st.markdown("**Languages**")
            st.write(row.get("languages", ""))
        st.markdown("**Events**")
        render_events(row)
        st.link_button("Open example website", row.get("website", "https://example.org"))


def main() -> None:
    st.set_page_config(
        page_title="AI Education Center Finder",
        page_icon="school",
        layout="wide",
    )
    st.title("AI-Based Education Center Cross-Function Website App")
    st.caption(
        "A portfolio-ready Streamlit app where students, parents, and teachers can find, filter, map, and compare education centers."
    )

    df = load_centers()
    if df.empty:
        st.stop()

    with st.sidebar:
        st.header("1. User profile")
        role = st.selectbox("I am using this as a", ["Student", "Parent", "Teacher"])
        grade = st.slider("Student grade", 0, 12, 9)
        age = st.number_input("Student age", min_value=4, max_value=25, value=15, step=1)
        has_membership = st.checkbox("Student/family has center membership or referral", value=False)
        goal_text = st.text_area(
            "Personal goal or need",
            value="I want a center with coding, tutoring, college readiness, and a safe study space.",
            height=90,
        )

        st.header("2. Date and availability")
        selected_date = st.date_input("Choose a date", value=dt.date.today())
        selected_time = st.time_input("Choose a time", value=dt.datetime.now().time().replace(second=0, microsecond=0))
        open_only = st.checkbox("Only show centers open at selected time", value=False)
        eligible_only = st.checkbox("Only show eligible centers", value=False)

        st.header("3. Location and commute")
        location_mode = st.radio("Location input", ["Use demo location", "Enter manually", "No commute filter"])
        if location_mode == "Use demo location":
            user_lat, user_lon = 39.2037, -76.8610
            st.caption("Demo location: Columbia, Maryland")
        elif location_mode == "Enter manually":
            user_lat = st.number_input("Your latitude", value=39.2037, format="%.6f")
            user_lon = st.number_input("Your longitude", value=-76.8610, format="%.6f")
        else:
            user_lat, user_lon = None, None
        commute_mode = st.selectbox("Commute mode", list(SPEED_KMPH.keys()), index=2)
        max_commute = st.slider("Maximum estimated commute time", 5, 180, 60, step=5)

        st.header("4. Filters")
        countries = ["All"] + sorted(df["country"].dropna().unique().tolist())
        country = st.selectbox("Country", countries, index=1 if len(countries) > 1 else 0)
        regions = st.multiselect("Region/state", sorted(df["region"].dropna().unique().tolist()))
        cities = st.multiselect("City", sorted(df["city"].dropna().unique().tolist()))
        center_types = st.multiselect("Center type", sorted(df["center_type"].dropna().unique().tolist()))
        selected_subjects = st.multiselect("Subject interests", unique_terms(df, "subjects"), default=[])
        selected_resources = st.multiselect("Resource needs", unique_terms(df, "resources"), default=[])
        keyword = st.text_input("Keyword search", placeholder="Example: robotics, essay, ESL, printing")

    enriched = add_computed_columns(
        df,
        selected_date,
        selected_time,
        int(grade),
        int(age),
        has_membership,
        user_lat,
        user_lon,
        commute_mode,
    )
    filtered = filter_centers(
        enriched,
        country,
        regions,
        cities,
        center_types,
        open_only,
        eligible_only,
        max_commute,
        keyword,
    )
    ranked = recommend_centers(
        filtered,
        role,
        selected_subjects,
        selected_resources,
        center_types,
        goal_text,
    )

    metric_cols = st.columns(4)
    metric_cols[0].metric("Matching centers", len(ranked))
    metric_cols[1].metric("Open now", int(ranked["open_selected_time"].sum()) if not ranked.empty else 0)
    metric_cols[2].metric("Eligible", int(ranked["eligible"].sum()) if not ranked.empty else 0)
    if not ranked.empty and ranked["commute_minutes"].notna().any():
        metric_cols[3].metric("Median commute", f"{ranked['commute_minutes'].median():.0f} min")
    else:
        metric_cols[3].metric("Median commute", "N/A")

    tab_recs, tab_map, tab_compare, tab_data, tab_method = st.tabs(
        ["Recommendations", "Regional map", "Compare centers", "Data editor", "How it works"]
    )

    with tab_recs:
        st.subheader("Personalized recommendations")
        if ranked.empty:
            st.warning("No centers match the current filters. Try expanding commute time, removing center type filters, or turning off open/eligible only.")
        else:
            for _, row in ranked.head(12).iterrows():
                render_center_card(row)

    with tab_map:
        st.subheader("Regional education-center map")
        fmap = build_map(ranked, user_lat, user_lon)
        st_folium(fmap, width=None, height=620)
        st.caption("Green markers are open and eligible for the selected profile. Orange markers need schedule or eligibility review.")

    with tab_compare:
        st.subheader("Comparison table")
        if ranked.empty:
            st.info("No results to compare.")
        else:
            cols = [
                "name",
                "city",
                "region",
                "center_type",
                "ai_match_score",
                "open_selected_time",
                "hours_today",
                "eligible",
                "eligibility_reasons",
                "commute_minutes",
                "cost_level",
                "subjects",
                "resources",
                "utilities",
                "languages",
                "website",
            ]
            display_cols = [c for c in cols if c in ranked.columns]
            st.dataframe(ranked[display_cols], use_container_width=True, hide_index=True)
            st.download_button(
                "Download filtered results as CSV",
                data=ranked[display_cols].to_csv(index=False).encode("utf-8"),
                file_name="filtered_education_centers.csv",
                mime="text/csv",
            )

    with tab_data:
        st.subheader("Dataset customization")
        st.markdown(
            "Edit the sample dataset in `education_centers_seed.csv` or replace it with open/public data for your chosen country. "
            "The app will automatically read the CSV when it starts."
        )
        edited = st.data_editor(df, use_container_width=True, num_rows="dynamic")
        st.download_button(
            "Download edited dataset",
            data=edited.to_csv(index=False).encode("utf-8"),
            file_name="education_centers_seed_edited.csv",
            mime="text/csv",
        )

    with tab_method:
        st.subheader("AI recommendation method")
        st.markdown(
            "The recommendation engine uses a transparent content-based approach: "
            "TF-IDF text similarity compares the user's goals, selected subjects, resources, and center type preferences against each center's profile. "
            "The final score also rewards eligibility, open status, lower commute time, and lower cost."
        )
        st.code(
            "score = 0.42*content + 0.18*eligibility + 0.16*open + 0.14*distance + 0.10*cost",
            language="python",
        )
        st.markdown(
            "For a more advanced version, replace the TF-IDF step with embeddings, add real transit APIs, connect to a public school/open-data portal, "
            "or add secure user accounts and saved preferences. Do not upload private student data unless the app has proper consent, security, and compliance controls."
        )


if __name__ == "__main__":
    main()
