from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import folium
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_folium import st_folium

APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "uzbekistan_education_centers.csv"
UZBEKISTAN_TIMEZONE = ZoneInfo("Asia/Tashkent")
APP_NAME = "AI Education Center Finder — Uzbekistan"
AUTHOR_NAME = "Mukhammadjon Khojikulov"
MENTOR_NAME = "Dr. Qingyang Xiao"
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

UZBEKISTAN_REGIONS = [
    "Republic of Karakalpakstan",
    "Andijan Region",
    "Bukhara Region",
    "Fergana Region",
    "Jizzakh Region",
    "Namangan Region",
    "Navoiy Region",
    "Qashqadaryo Region",
    "Samarqand Region",
    "Sirdaryo Region",
    "Surxondaryo Region",
    "Tashkent Region",
    "Khorezm Region",
    "Tashkent City",
]

LOCATION_PRESETS = {
    "Tashkent": (41.3111, 69.2797),
    "Samarkand": (39.6542, 66.9597),
    "Bukhara": (39.7681, 64.4556),
    "Andijan": (40.7821, 72.3442),
    "Fergana": (40.3894, 71.7870),
    "Namangan": (40.9983, 71.6726),
    "Nukus": (42.4602, 59.6166),
    "Qarshi": (38.8610, 65.7847),
    "Termez": (37.2242, 67.2783),
    "Urgench": (41.5500, 60.6333),
    "Navoiy": (40.1039, 65.3688),
    "Jizzakh": (40.1250, 67.8808),
    "Gulistan": (40.4897, 68.7842),
    "Chirchiq": (41.4689, 69.5822),
}

SPEED_KMPH = {
    "Walking": 5,
    "Bicycle": 15,
    "Public transport": 24,
    "Car / taxi": 42,
}
MODE_BUFFER_MINUTES = {
    "Walking": 2,
    "Bicycle": 4,
    "Public transport": 14,
    "Car / taxi": 8,
}

TRANSLATIONS = {
    "English": {
        "title": "AI Education Center Finder — Uzbekistan",
        "subtitle": "Find, compare, and map education centers across Uzbekistan.",
        "demo": "Demonstration dataset: center records are synthetic and must be replaced or verified before public use.",
        "profile": "1. User profile",
        "availability": "2. Date and availability",
        "location": "3. Location and commute",
        "filters": "4. Uzbekistan filters",
        "recommendations": "Recommendations",
        "map": "Uzbekistan map",
        "compare": "Compare centers",
        "analytics": "Regional analytics",
        "data": "Data and project notes",
    },
    "O‘zbekcha": {
        "title": "AI Ta’lim Markazlari Qidiruvi — O‘zbekiston",
        "subtitle": "O‘zbekiston bo‘ylab ta’lim markazlarini toping, solishtiring va xaritada ko‘ring.",
        "demo": "Namoyish ma’lumotlari: markazlar sintetik. Ommaviy foydalanishdan oldin tekshiring yoki haqiqiy ma’lumot bilan almashtiring.",
        "profile": "1. Foydalanuvchi profili",
        "availability": "2. Sana va mavjudlik",
        "location": "3. Joylashuv va yo‘l vaqti",
        "filters": "4. O‘zbekiston filtrlari",
        "recommendations": "Tavsiyalar",
        "map": "O‘zbekiston xaritasi",
        "compare": "Markazlarni solishtirish",
        "analytics": "Hududiy tahlil",
        "data": "Ma’lumotlar va loyiha izohlari",
    },
    "Русский": {
        "title": "AI-поиск образовательных центров — Узбекистан",
        "subtitle": "Ищите, сравнивайте и просматривайте образовательные центры Узбекистана на карте.",
        "demo": "Демонстрационные данные: записи центров синтетические. Перед публикацией замените или проверьте их.",
        "profile": "1. Профиль пользователя",
        "availability": "2. Дата и доступность",
        "location": "3. Местоположение и дорога",
        "filters": "4. Фильтры по Узбекистану",
        "recommendations": "Рекомендации",
        "map": "Карта Узбекистана",
        "compare": "Сравнение центров",
        "analytics": "Региональная аналитика",
        "data": "Данные и описание проекта",
    },
}


def safe_json_loads(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (list, dict)):
        return value
    if isinstance(value, float) and np.isnan(value):
        return default
    try:
        return json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return default


@st.cache_data(show_spinner=False)
def load_centers(path: str | Path = DATA_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        st.error("Missing uzbekistan_education_centers.csv in the repository root.")
        return pd.DataFrame()

    df = pd.read_csv(path)
    required = {
        "center_id",
        "name",
        "country",
        "region",
        "city",
        "center_type",
        "latitude",
        "longitude",
        "min_grade",
        "max_grade",
        "min_age",
        "max_age",
        "cost_level",
        "subjects",
        "resources",
        "utilities",
        "languages",
        "open_schedule_json",
        "events_json",
        "description",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        st.error("Dataset is missing required columns: " + ", ".join(missing))
        return pd.DataFrame()

    df = df[df["country"].astype(str).str.strip().eq("Uzbekistan")].copy()
    for col in [
        "subjects",
        "resources",
        "utilities",
        "affiliations",
        "languages",
        "description",
        "eligibility_notes",
        "search_keywords",
        "district",
        "phone",
        "telegram",
        "website",
        "verification_status",
    ]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)

    bool_map = {"yes": True, "no": False, "true": True, "false": False, "1": True, "0": False}
    membership_values = df.get("membership_required", pd.Series("No", index=df.index))
    df["membership_required_bool"] = (
        membership_values.fillna("No").astype(str).str.lower().map(bool_map).fillna(False)
    )
    return df


def split_pipe(value: Any) -> list[str]:
    if value is None:
        return []
    return [item.strip() for item in str(value).split("|") if item.strip()]


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
    except (TypeError, ValueError):
        return None


def time_in_slot(selected_time: dt.time, start_text: str, end_text: str) -> bool:
    start = parse_time(start_text)
    end = parse_time(end_text)
    if start is None or end is None:
        return False
    if start <= end:
        return start <= selected_time <= end
    return selected_time >= start or selected_time <= end


def slots_for_day(row: pd.Series, selected_date: dt.date) -> list[dict[str, str]]:
    schedule = safe_json_loads(row.get("open_schedule_json"), [])
    day_name = selected_date.strftime("%A")
    return [slot for slot in schedule if slot.get("day") == day_name]


def is_open_at(row: pd.Series, selected_date: dt.date, selected_time: dt.time) -> bool:
    return any(
        time_in_slot(selected_time, slot.get("start", ""), slot.get("end", ""))
        for slot in slots_for_day(row, selected_date)
    )


def hours_for_day(row: pd.Series, selected_date: dt.date) -> str:
    slots = slots_for_day(row, selected_date)
    if not slots:
        return "Closed"
    return ", ".join(f"{slot.get('start')}–{slot.get('end')}" for slot in slots)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0
    phi1 = math.radians(float(lat1))
    phi2 = math.radians(float(lat2))
    delta_phi = math.radians(float(lat2) - float(lat1))
    delta_lambda = math.radians(float(lon2) - float(lon1))
    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    return 2 * radius_km * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def estimate_commute_minutes(distance_km: float | None, mode: str) -> float | None:
    if distance_km is None or pd.isna(distance_km):
        return None
    speed = SPEED_KMPH.get(mode, 24)
    buffer = MODE_BUFFER_MINUTES.get(mode, 8)
    return round((float(distance_km) / speed) * 60 + buffer, 1)


def eligibility_status(
    row: pd.Series,
    grade: int,
    age: int,
    has_membership: bool,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    min_grade, max_grade = int(row.get("min_grade", 1)), int(row.get("max_grade", 11))
    min_age, max_age = int(row.get("min_age", 6)), int(row.get("max_age", 99))
    membership_required = bool(row.get("membership_required_bool", False))

    if not min_grade <= grade <= max_grade:
        reasons.append(f"grade {grade} is outside {min_grade}–{max_grade}")
    if not min_age <= age <= max_age:
        reasons.append(f"age {age} is outside {min_age}–{max_age}")
    if membership_required and not has_membership:
        reasons.append("membership or referral may be required")
    return not reasons, reasons


def text_blob(row: pd.Series) -> str:
    pieces = [
        row.get("name", ""),
        row.get("center_type", ""),
        row.get("region", ""),
        row.get("city", ""),
        row.get("district", ""),
        row.get("subjects", ""),
        row.get("resources", ""),
        row.get("utilities", ""),
        row.get("affiliations", ""),
        row.get("languages", ""),
        row.get("description", ""),
        row.get("eligibility_notes", ""),
        row.get("search_keywords", ""),
    ]
    for event in safe_json_loads(row.get("events_json"), []):
        pieces.extend([str(event.get("title", "")), str(event.get("audience", ""))])
    return " ".join(str(piece) for piece in pieces if str(piece).strip())


def cost_score(cost_level: str) -> float:
    mapping = {
        "Free": 1.00,
        "Free / subsidized": 0.92,
        "Low-cost": 0.82,
        "Moderate": 0.62,
        "Premium": 0.38,
    }
    return mapping.get(str(cost_level), 0.58)


def build_user_query(
    role: str,
    selected_subjects: Iterable[str],
    selected_resources: Iterable[str],
    center_types: Iterable[str],
    goal_text: str,
    preferred_languages: Iterable[str],
) -> str:
    parts = [role, goal_text]
    parts.extend(selected_subjects)
    parts.extend(selected_resources)
    parts.extend(center_types)
    parts.extend(preferred_languages)
    query = " ".join(str(part) for part in parts if str(part).strip())
    return query or "ta'lim education o'quv learning tutoring matematika ingliz tili IT"


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
    result = df.copy()
    result["open_selected_time"] = result.apply(
        lambda row: is_open_at(row, selected_date, selected_time), axis=1
    )
    result["hours_selected_day"] = result.apply(
        lambda row: hours_for_day(row, selected_date), axis=1
    )

    eligibilities = result.apply(
        lambda row: eligibility_status(row, grade, age, has_membership), axis=1
    )
    result["eligible"] = [status for status, _ in eligibilities]
    result["eligibility_reasons"] = [
        "; ".join(reasons) if reasons else "Eligible based on grade, age, and membership inputs"
        for _, reasons in eligibilities
    ]

    if user_lat is not None and user_lon is not None:
        result["distance_km"] = result.apply(
            lambda row: haversine_km(user_lat, user_lon, row["latitude"], row["longitude"]),
            axis=1,
        )
        result["commute_minutes"] = result["distance_km"].apply(
            lambda distance: estimate_commute_minutes(distance, commute_mode)
        )
    else:
        result["distance_km"] = np.nan
        result["commute_minutes"] = np.nan
    return result


def build_reason(
    row: pd.Series,
    selected_subjects: list[str],
    selected_resources: list[str],
    preferred_languages: list[str],
) -> str:
    row_terms = set(
        split_pipe(row.get("subjects", ""))
        + split_pipe(row.get("resources", ""))
        + split_pipe(row.get("languages", ""))
    )
    selected = selected_subjects + selected_resources + preferred_languages
    matches = [term for term in selected if term in row_terms]
    reasons: list[str] = []
    if matches:
        reasons.append("matches " + ", ".join(matches[:5]))
    if row.get("open_selected_time"):
        reasons.append("open at the selected time")
    if row.get("eligible"):
        reasons.append("eligible for the selected profile")
    if pd.notna(row.get("commute_minutes", np.nan)):
        reasons.append(f"estimated {row.get('commute_minutes')} min away")
    return "; ".join(reasons) or "broad profile match"


def recommend_centers(
    df: pd.DataFrame,
    role: str,
    selected_subjects: list[str],
    selected_resources: list[str],
    center_types: list[str],
    goal_text: str,
    preferred_languages: list[str],
) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    ranked = df.copy()
    query = build_user_query(
        role,
        selected_subjects,
        selected_resources,
        center_types,
        goal_text,
        preferred_languages,
    )
    documents = ranked.apply(text_blob, axis=1).tolist() + [query]

    # Character n-grams work well for English, Uzbek, and Russian text without an API key.
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True)
    matrix = vectorizer.fit_transform(documents)
    similarities = cosine_similarity(matrix[:-1], matrix[-1]).ravel()
    content_scores = similarities / np.max(similarities) if np.max(similarities) > 0 else np.zeros_like(similarities)

    eligible_scores = ranked["eligible"].astype(float).to_numpy()
    open_scores = ranked["open_selected_time"].astype(float).to_numpy()
    cost_scores = ranked["cost_level"].apply(cost_score).to_numpy()

    if ranked["distance_km"].notna().any():
        median_distance = ranked["distance_km"].median()
        distance_scores = 1 / (1 + ranked["distance_km"].fillna(median_distance) / 10)
        distance_scores = distance_scores.to_numpy()
    else:
        distance_scores = np.full(len(ranked), 0.65)

    final_score = (
        0.43 * content_scores
        + 0.19 * eligible_scores
        + 0.15 * open_scores
        + 0.13 * distance_scores
        + 0.10 * cost_scores
    )
    ranked["ai_match_score"] = np.round(final_score * 100, 1)
    ranked["content_match"] = np.round(content_scores * 100, 1)
    ranked["ai_reason"] = ranked.apply(
        lambda row: build_reason(row, selected_subjects, selected_resources, preferred_languages),
        axis=1,
    )
    return ranked.sort_values(
        ["ai_match_score", "open_selected_time", "eligible"],
        ascending=[False, False, False],
    )


def filter_centers(
    df: pd.DataFrame,
    regions: list[str],
    cities: list[str],
    center_types: list[str],
    preferred_languages: list[str],
    open_only: bool,
    eligible_only: bool,
    max_commute: int,
    keyword: str,
) -> pd.DataFrame:
    results = df.copy()
    if regions:
        results = results[results["region"].isin(regions)]
    if cities:
        results = results[results["city"].isin(cities)]
    if center_types:
        results = results[results["center_type"].isin(center_types)]
    if preferred_languages:
        language_mask = results["languages"].apply(
            lambda value: any(language in split_pipe(value) for language in preferred_languages)
        )
        results = results[language_mask]
    if open_only:
        results = results[results["open_selected_time"]]
    if eligible_only:
        results = results[results["eligible"]]
    if results["commute_minutes"].notna().any():
        results = results[results["commute_minutes"] <= max_commute]
    if keyword.strip():
        normalized = keyword.casefold().strip()
        mask = results.apply(lambda row: normalized in text_blob(row).casefold(), axis=1)
        results = results[mask]
    return results


def build_map(df: pd.DataFrame, user_lat: float | None, user_lon: float | None) -> folium.Map:
    if user_lat is not None and user_lon is not None:
        start = [user_lat, user_lon]
        zoom = 9
    elif not df.empty:
        start = [df["latitude"].mean(), df["longitude"].mean()]
        zoom = 6
    else:
        start = [41.1, 64.5]
        zoom = 6

    fmap = folium.Map(
        location=start,
        zoom_start=zoom,
        tiles="CartoDB positron",
        control_scale=True,
        min_zoom=5,
        max_bounds=True,
    )
    fmap.fit_bounds([[37.0, 55.5], [46.0, 74.5]])

    if user_lat is not None and user_lon is not None:
        folium.Marker(
            location=[user_lat, user_lon],
            popup="Selected user location",
            tooltip="Your location",
            icon=folium.Icon(color="blue", icon="home"),
        ).add_to(fmap)

    for _, row in df.iterrows():
        open_text = "Open" if row.get("open_selected_time") else "Closed at selected time"
        eligibility_text = "Eligible" if row.get("eligible") else "Eligibility review needed"
        marker_color = "green" if row.get("open_selected_time") and row.get("eligible") else "orange"
        popup_html = f"""
        <div style='width:260px'>
        <b>{row.get('name')}</b><br>
        {row.get('center_type')}<br>
        {row.get('city')}, {row.get('region')}<br>
        Hours: {row.get('hours_selected_day')}<br>
        Status: {open_text}<br>
        Eligibility: {eligibility_text}<br>
        AI match: {row.get('ai_match_score', 'N/A')}%<br>
        <small>{row.get('verification_status')}</small>
        </div>
        """
        folium.Marker(
            location=[row["latitude"], row["longitude"]],
            popup=folium.Popup(popup_html, max_width=340),
            tooltip=row.get("name"),
            icon=folium.Icon(color=marker_color, icon="info-sign"),
        ).add_to(fmap)
    return fmap


def render_events(row: pd.Series) -> None:
    events = safe_json_loads(row.get("events_json"), [])
    if not events:
        st.write("No events listed.")
        return
    st.dataframe(pd.DataFrame(events), use_container_width=True, hide_index=True)


def render_contact(row: pd.Series) -> None:
    contact_items = []
    if row.get("phone"):
        contact_items.append(f"Phone: {row.get('phone')}")
    if row.get("telegram"):
        contact_items.append(f"Telegram: {row.get('telegram')}")
    if contact_items:
        st.write(" | ".join(contact_items))
    website = str(row.get("website", "")).strip()
    if website.startswith("http"):
        st.link_button("Open website", website)


def render_center_card(row: pd.Series) -> None:
    label = f"{row.get('name')} · {row.get('ai_match_score', 'N/A')}% match"
    with st.expander(label, expanded=False):
        left, right = st.columns([1.35, 1])
        with left:
            st.markdown(f"**Type:** {row.get('center_type')}")
            st.markdown(
                f"**Location:** {row.get('address')}, {row.get('district')}, {row.get('city')}, {row.get('region')}"
            )
            st.markdown(f"**Hours on selected day:** {row.get('hours_selected_day')}")
            st.markdown(
                f"**Selected-time status:** {'Open' if row.get('open_selected_time') else 'Closed'}"
            )
            st.markdown(f"**Eligibility:** {'Eligible' if row.get('eligible') else 'Review needed'}")
            st.markdown(f"**Eligibility result:** {row.get('eligibility_reasons')}")
            st.markdown(f"**AI explanation:** {row.get('ai_reason')}")
            if pd.notna(row.get("commute_minutes", np.nan)):
                st.markdown(
                    f"**Estimated commute:** {row.get('commute_minutes')} minutes · {row.get('distance_km'):.1f} km"
                )
            st.markdown(f"**Cost:** {row.get('cost_level')} · {row.get('monthly_fee_uzs', 'Contact center')} UZS/month")
            st.markdown(f"**Description:** {row.get('description')}")
            st.warning(str(row.get("verification_status")))
            render_contact(row)
        with right:
            st.markdown("**Subjects**")
            st.write(", ".join(split_pipe(row.get("subjects", ""))))
            st.markdown("**Resources**")
            st.write(", ".join(split_pipe(row.get("resources", ""))))
            st.markdown("**Utilities**")
            st.write(", ".join(split_pipe(row.get("utilities", ""))))
            st.markdown("**Languages**")
            st.write(", ".join(split_pipe(row.get("languages", ""))))
            st.markdown("**Affiliations / partners**")
            st.write(", ".join(split_pipe(row.get("affiliations", ""))) or "Not listed")
        st.markdown("**Programs and events**")
        render_events(row)


def regional_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["region", "centers", "open_at_selected_time", "eligible", "average_ai_match"])
    summary = (
        df.groupby("region", as_index=False)
        .agg(
            centers=("center_id", "count"),
            open_at_selected_time=("open_selected_time", "sum"),
            eligible=("eligible", "sum"),
            average_ai_match=("ai_match_score", "mean"),
        )
        .sort_values("centers", ascending=False)
    )
    summary["average_ai_match"] = summary["average_ai_match"].round(1)
    return summary


def main() -> None:
    st.set_page_config(
        page_title=APP_NAME,
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.6rem; padding-bottom: 3rem;}
        [data-testid="stMetricValue"] {font-size: 1.65rem;}
        .uz-card {padding: 0.8rem 1rem; border: 1px solid rgba(128,128,128,.25); border-radius: 0.8rem;}
        .sidebar-project-card {
            padding: 1rem 1rem 0.9rem 1rem;
            margin: 0 0 1rem 0;
            border: 1px solid rgba(0, 153, 181, 0.28);
            border-radius: 0.9rem;
            background: linear-gradient(145deg, rgba(0,153,181,.10), rgba(255,255,255,.70));
        }
        .sidebar-app-label {
            font-size: 0.68rem;
            font-weight: 800;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: #007f98;
            margin-bottom: 0.3rem;
        }
        .sidebar-app-name {
            font-size: 1.02rem;
            line-height: 1.28;
            font-weight: 800;
            color: #172B3A;
            margin-bottom: 0.8rem;
        }
        .sidebar-credit {
            font-size: 0.83rem;
            line-height: 1.45;
            color: #344b59;
            margin-top: 0.32rem;
        }
        .project-declaration {
            padding: 1rem 1.1rem;
            border-left: 5px solid #0099B5;
            border-radius: 0.55rem;
            background: rgba(0,153,181,.07);
            margin: 0.35rem 0 1.1rem 0;
            line-height: 1.65;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown(
            f"""
            <div class="sidebar-project-card">
                <div class="sidebar-app-label">Application</div>
                <div class="sidebar-app-name">{APP_NAME}</div>
                <div class="sidebar-credit"><strong>Author</strong><br>{AUTHOR_NAME}</div>
                <div class="sidebar-credit"><strong>Mentor</strong><br>{MENTOR_NAME}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        ui_language = st.selectbox("Interface language", list(TRANSLATIONS), index=0)
    text = TRANSLATIONS[ui_language]

    st.title(text["title"])
    st.caption(text["subtitle"])
    st.warning(text["demo"])

    df = load_centers()
    if df.empty:
        st.stop()

    now_tashkent = dt.datetime.now(UZBEKISTAN_TIMEZONE)

    with st.sidebar:
        st.header(text["profile"])
        role = st.selectbox("User role", ["Student / O‘quvchi", "Parent / Ota-ona", "Teacher / O‘qituvchi"])
        grade = st.slider("Student grade (Uzbekistan system)", 1, 11, 8)
        age = st.number_input("Student age", min_value=6, max_value=25, value=14, step=1)
        has_membership = st.checkbox("Has membership or referral", value=False)
        goal_text = st.text_area(
            "Learning goal (English, Uzbek, or Russian)",
            value="Matematika, ingliz tili va IT bo‘yicha darslar kerak.",
            height=90,
        )

        st.header(text["availability"])
        selected_date = st.date_input("Choose date", value=now_tashkent.date())
        selected_time = st.time_input(
            "Choose local time (Asia/Tashkent)",
            value=now_tashkent.time().replace(second=0, microsecond=0),
        )
        open_only = st.checkbox("Only open at selected time", value=False)
        eligible_only = st.checkbox("Only eligible for profile", value=False)

        st.header(text["location"])
        location_mode = st.radio(
            "Location input",
            ["Choose Uzbekistan city", "Enter coordinates", "No commute filter"],
        )
        if location_mode == "Choose Uzbekistan city":
            preset_name = st.selectbox("Your city", list(LOCATION_PRESETS), index=0)
            user_lat, user_lon = LOCATION_PRESETS[preset_name]
            st.caption(f"Map origin: {preset_name}")
        elif location_mode == "Enter coordinates":
            user_lat = st.number_input("Latitude", min_value=37.0, max_value=46.0, value=41.3111, format="%.6f")
            user_lon = st.number_input("Longitude", min_value=55.5, max_value=74.5, value=69.2797, format="%.6f")
        else:
            user_lat, user_lon = None, None

        commute_mode = st.selectbox("Commute mode", list(SPEED_KMPH), index=2)
        max_commute = st.slider("Maximum estimated commute", 5, 300, 75, step=5)
        st.caption("Commute time is an estimate, not live routing.")

        st.header(text["filters"])
        regions = st.multiselect("Region", UZBEKISTAN_REGIONS)
        available_cities = sorted(
            df[df["region"].isin(regions)]["city"].unique().tolist()
            if regions
            else df["city"].unique().tolist()
        )
        cities = st.multiselect("City", available_cities)
        center_types = st.multiselect("Center type", sorted(df["center_type"].unique().tolist()))
        preferred_languages = st.multiselect("Preferred teaching language", unique_terms(df, "languages"))
        selected_subjects = st.multiselect("Subject interests", unique_terms(df, "subjects"))
        selected_resources = st.multiselect("Resource needs", unique_terms(df, "resources"))
        keyword = st.text_input("Keyword", placeholder="robotics, IELTS, matematika, русский...")

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
        regions,
        cities,
        center_types,
        preferred_languages,
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
        preferred_languages,
    )

    metric_cols = st.columns(5)
    metric_cols[0].metric("Matching centers", len(ranked))
    metric_cols[1].metric("Regions represented", ranked["region"].nunique() if not ranked.empty else 0)
    metric_cols[2].metric("Open", int(ranked["open_selected_time"].sum()) if not ranked.empty else 0)
    metric_cols[3].metric("Eligible", int(ranked["eligible"].sum()) if not ranked.empty else 0)
    median_commute = ranked["commute_minutes"].median() if not ranked.empty else np.nan
    metric_cols[4].metric("Median commute", f"{median_commute:.0f} min" if pd.notna(median_commute) else "N/A")

    tab_recs, tab_map, tab_compare, tab_analytics, tab_data = st.tabs(
        [text["recommendations"], text["map"], text["compare"], text["analytics"], text["data"]]
    )

    with tab_recs:
        st.subheader("Personalized Uzbekistan recommendations")
        if ranked.empty:
            st.warning("No centers match the current filters. Remove one or more filters or increase the commute limit.")
        else:
            for _, row in ranked.head(14).iterrows():
                render_center_card(row)

    with tab_map:
        st.subheader("Education-center map of Uzbekistan")
        fmap = build_map(ranked, user_lat, user_lon)
        st_folium(fmap, width=None, height=640, key="uzbekistan_center_map")
        st.caption("Green: open and eligible. Orange: schedule or eligibility review needed. Blue: selected user location.")

    with tab_compare:
        st.subheader("Comparison table")
        if ranked.empty:
            st.info("No results to compare.")
        else:
            columns = [
                "name",
                "region",
                "city",
                "district",
                "center_type",
                "ai_match_score",
                "open_selected_time",
                "hours_selected_day",
                "eligible",
                "commute_minutes",
                "cost_level",
                "monthly_fee_uzs",
                "subjects",
                "resources",
                "languages",
                "verification_status",
            ]
            display_columns = [column for column in columns if column in ranked.columns]
            st.dataframe(ranked[display_columns], use_container_width=True, hide_index=True)
            st.download_button(
                "Download filtered Uzbekistan results",
                data=ranked[display_columns].to_csv(index=False).encode("utf-8-sig"),
                file_name="uzbekistan_education_center_results.csv",
                mime="text/csv",
            )

    with tab_analytics:
        st.subheader("Regional coverage and match analytics")
        summary = regional_summary(ranked)
        if summary.empty:
            st.info("No data is available for the current filters.")
        else:
            st.dataframe(summary, use_container_width=True, hide_index=True)
            st.markdown("**Centers by region**")
            st.bar_chart(summary.set_index("region")[["centers"]])
            st.markdown("**Center types in current results**")
            type_counts = ranked["center_type"].value_counts().rename_axis("center_type").to_frame("centers")
            st.bar_chart(type_counts)

    with tab_data:
        st.subheader("Dataset, algorithms, authorship, and responsible deployment")
        st.markdown(
            f"""
            <div class="project-declaration">
                <strong>Application:</strong> {APP_NAME}<br>
                <strong>Author:</strong> {AUTHOR_NAME}<br>
                <strong>Mentor:</strong> {MENTOR_NAME}<br>
                <strong>Scope:</strong> Uzbekistan-only education-center discovery, filtering, mapping, comparison, and explainable recommendation support.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### Calculation and algorithm declaration")
        st.markdown(
            "The app applies deterministic filters first and then calculates an explainable recommendation score. "
            "The result is decision support only; it is not a guarantee of center quality, admission, availability, or travel time."
        )

        with st.expander("1. How the AI recommendation score is calculated", expanded=True):
            st.markdown(
                "**Step A — Build the text compared by the model.** For each center, the app joins its name, center type, "
                "region, city, district, subjects, resources, utilities, affiliations, languages, description, eligibility notes, "
                "search keywords, and event titles/audiences. The user query joins the selected role, learning goal, subjects, "
                "resources, center types, and preferred languages."
            )
            st.markdown(
                "**Step B — Local multilingual text matching.** The app uses `TfidfVectorizer` with "
                "`analyzer='char_wb'`, character n-grams from 3 to 5 characters, `min_df=1`, and `sublinear_tf=True`. "
                "Cosine similarity compares every center vector with the user-query vector. The similarities are divided by "
                "the highest similarity in the current filtered result set, producing a relative content score from 0 to 1. "
                "If every similarity is zero, every content score is set to zero."
            )
            st.code(
                """score_percent = 100 * (
    0.43 * content_score
  + 0.19 * eligibility_score
  + 0.15 * open_score
  + 0.13 * distance_score
  + 0.10 * affordability_score
)""",
                language="python",
            )
            st.markdown(
                """- **Content score:** normalized TF-IDF cosine similarity, from 0 to 1.
- **Eligibility score:** 1 only when grade, age, and any membership/referral requirement pass; otherwise 0.
- **Open score:** 1 when at least one schedule interval contains the selected local date/time; otherwise 0.
- **Distance score:** `1 / (1 + distance_km / 10)`. When no user location is supplied, the neutral value 0.65 is used.
- **Affordability score:** Free=1.00, Free/subsidized=0.92, Low-cost=0.82, Moderate=0.62, Premium=0.38; unrecognized values=0.58.
- **Final ordering:** descending AI score, then open status, then eligibility."""
            )

        with st.expander("2. Eligibility and opening-hours calculations"):
            st.markdown(
                "A center is marked **eligible** only when the selected grade is between `min_grade` and `max_grade`, "
                "the selected age is between `min_age` and `max_age`, and a required membership/referral is present. "
                "Each comparison includes both endpoints."
            )
            st.markdown(
                "Opening hours are read from `open_schedule_json`. The selected date is converted to its English weekday name, "
                "and the app checks whether the selected time falls inside any interval for that weekday. Start and end times "
                "are inclusive. An interval whose end time is earlier than its start time is treated as an overnight interval."
            )

        with st.expander("3. Distance and commute-time calculations"):
            st.markdown(
                "Straight-line geographic distance is calculated with the Haversine equation using an Earth radius of 6,371 km."
            )
            st.code(
                """a = sin²(Δlatitude/2) + cos(latitude_1) * cos(latitude_2) * sin²(Δlongitude/2)
distance_km = 2 * 6371 * atan2(sqrt(a), sqrt(1-a))""",
                language="text",
            )
            st.markdown(
                "The commute estimate is not live road routing. It is calculated as "
                "`round((distance_km / assumed_speed_kmph) * 60 + mode_buffer_minutes, 1)`."
            )
            commute_assumptions = pd.DataFrame(
                [
                    {"Mode": "Walking", "Assumed speed (km/h)": 5, "Fixed buffer (minutes)": 2},
                    {"Mode": "Bicycle", "Assumed speed (km/h)": 15, "Fixed buffer (minutes)": 4},
                    {"Mode": "Public transport", "Assumed speed (km/h)": 24, "Fixed buffer (minutes)": 14},
                    {"Mode": "Car / taxi", "Assumed speed (km/h)": 42, "Fixed buffer (minutes)": 8},
                ]
            )
            st.dataframe(commute_assumptions, use_container_width=True, hide_index=True)

        with st.expander("4. Dashboard metric definitions"):
            st.markdown(
                """- **Matching centers:** number of records remaining after the selected region, city, center-type, language, open-only, eligible-only, commute, and keyword filters.
- **Regions represented:** number of distinct Uzbekistan region values among matching centers.
- **Open:** count of matching centers open at the selected local date and time.
- **Eligible:** count of matching centers passing the grade, age, and membership/referral rules.
- **Median commute:** median of the estimated commute minutes among matching centers; shown as N/A when no location is used."""
            )

        st.markdown("### Uzbekistan-only data declaration")
        st.markdown(
            "This repository is restricted to Uzbekistan: the app filters the CSV to `country == Uzbekistan`, "
            "the country selector is removed, the map is bounded to Uzbekistan, the grade range is 1–11, "
            "the timezone is `Asia/Tashkent`, and the location presets contain Uzbekistan cities only."
        )
        st.warning(
            "The bundled center records are synthetic portfolio data. Before a public launch, replace them with verified, "
            "permissioned records and establish a recurring process for checking hours, fees, eligibility, phone numbers, "
            "addresses, accessibility information, and event dates."
        )

        st.markdown("### Authorship and responsible-use declaration")
        st.markdown(
            f"This application was authored by **{AUTHOR_NAME}** under the mentorship of **{MENTOR_NAME}**. "
            "The recommendation factors and assumptions are disclosed above to make the ranking auditable. "
            "No OpenAI key, paid API, or external AI service is required. User-entered profile information is processed "
            "during the current Streamlit session and should not be treated as a secure student-record system."
        )
        st.markdown(
            "For production use, add verified data governance, privacy and parental-consent controls, accessibility review, "
            "center correction mechanisms, audit logs, secure authentication, and real routing before relying on the app for decisions."
        )
        st.download_button(
            "Download bundled demonstration dataset",
            data=df.drop(columns=["membership_required_bool"], errors="ignore").to_csv(index=False).encode("utf-8-sig"),
            file_name="uzbekistan_education_centers.csv",
            mime="text/csv",
        )



if __name__ == "__main__":
    main()
