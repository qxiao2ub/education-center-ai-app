# AI Education Center Finder — Uzbekistan

A GitHub- and Streamlit-ready portfolio project for students, parents, and teachers to discover, filter, map, compare, and rank education centers **inside Uzbekistan only**.

**Author:** Mukhammadjon Khojikulov  
**Mentor:** Dr. Qingyang Xiao

## Important data notice

The bundled center records are **synthetic demonstration data**. The project intentionally labels every sample record as `[Demo]`. Before a public launch, replace the CSV with verified and permissioned information from education centers or suitable public/open-data sources.

## What makes this Uzbekistan-only

- The application loads records only when `country == Uzbekistan`.
- There is no country selector.
- The region list covers the Republic of Karakalpakstan, 12 regions, and Tashkent City.
- The map is centered and bounded to Uzbekistan.
- Location presets contain Uzbekistan cities only.
- The student grade selector follows grades 1–11.
- Local date/time defaults use `Asia/Tashkent`.
- Fees are represented in Uzbekistani soʻm (UZS).
- English, Uzbek, and Russian interface headings are included.

## Main features

- User profiles for students, parents, and teachers
- Region, city, center type, subject, resource, language, and keyword filters
- Date/time opening-hours checks
- Grade, age, and membership eligibility checks
- Estimated commute time by walking, bicycle, public transport, or car/taxi
- Interactive Folium map of Uzbekistan
- Explainable AI recommendation score
- Multilingual text matching for English, Uzbek, and Russian queries
- Center comparison table and CSV download
- Regional coverage analytics
- No OpenAI key or paid API required

## Explainable AI and calculation declaration

The app applies deterministic filters first, then calculates a local, auditable recommendation score:

```text
score_percent = 100 × (
    0.43 × content_score
  + 0.19 × eligibility_score
  + 0.15 × open_score
  + 0.13 × distance_score
  + 0.10 × affordability_score
)
```

### Content score

The app combines each center's name, type, Uzbekistan location, subjects, resources, utilities, affiliations, languages, description, eligibility notes, search keywords, and event information. The user query combines role, learning goal, subjects, resources, preferred center types, and teaching languages.

It then uses:

```python
TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 5),
    min_df=1,
    sublinear_tf=True,
)
```

Cosine similarity compares each center with the user query. Similarities are normalized by the highest similarity in the current filtered result set. This makes the content component a relative score from 0 to 1. No user text is sent to an external AI service.

### Eligibility and schedule scores

- `eligibility_score = 1` only when grade, age, and any membership/referral requirement pass; otherwise 0.
- `open_score = 1` when at least one schedule interval for the selected weekday contains the selected local time; otherwise 0.
- Start and end times are inclusive, and overnight schedule intervals are supported.

### Distance and commute calculations

Straight-line distance is calculated with the Haversine equation using an Earth radius of 6,371 km:

```text
a = sin²(Δlatitude/2) + cos(latitude_1) × cos(latitude_2) × sin²(Δlongitude/2)
distance_km = 2 × 6371 × atan2(sqrt(a), sqrt(1-a))
```

The ranking distance component is:

```text
distance_score = 1 / (1 + distance_km / 10)
```

When the user does not provide a location, the app uses a neutral distance score of `0.65`.

Commute time is a transparent approximation, not live routing:

```text
commute_minutes = round((distance_km / assumed_speed_kmph) × 60 + fixed_buffer_minutes, 1)
```

| Mode | Assumed speed | Fixed buffer |
| --- | ---: | ---: |
| Walking | 5 km/h | 2 min |
| Bicycle | 15 km/h | 4 min |
| Public transport | 24 km/h | 14 min |
| Car / taxi | 42 km/h | 8 min |

### Affordability score

| Cost level | Score |
| --- | ---: |
| Free | 1.00 |
| Free / subsidized | 0.92 |
| Low-cost | 0.82 |
| Moderate | 0.62 |
| Premium | 0.38 |
| Unrecognized value | 0.58 |

Final results are sorted by AI match score, then open status, then eligibility. The score supports discovery and comparison; it does not measure teaching quality or guarantee admission, availability, or travel time.

### Dashboard metric definitions

- **Matching centers:** records left after all active filters.
- **Regions represented:** distinct Uzbekistan regions in the matching records.
- **Open:** matching centers open at the selected date and time.
- **Eligible:** matching centers passing grade, age, and membership/referral rules.
- **Median commute:** median estimated commute time among matching centers, or N/A without a location.

## Repository structure

```text
uzbekistan_education_center_app/
├── .streamlit/
│   └── config.toml
├── app.py
├── uzbekistan_education_centers.csv
├── Uzbekistan_Education_Center_App_Colab.ipynb
├── requirements.txt
├── smoke_test.py
├── README.md
├── LICENSE
└── .gitignore
```

## Run locally

### 1. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install packages

```bash
pip install -r requirements.txt
```

### 3. Start Streamlit

```bash
streamlit run app.py
```

## Upload to GitHub

1. Extract the ZIP file.
2. Create a new empty GitHub repository.
3. Open a terminal inside the extracted project folder.
4. Run:

```bash
git init
git add .
git commit -m "Create Uzbekistan education center finder"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
git push -u origin main
```

You can also upload the extracted files through the GitHub website. Make sure `app.py`, `requirements.txt`, and the CSV remain in the repository root.

## Deploy on Streamlit Community Cloud

1. Push the extracted project to a public GitHub repository.
2. Sign in to Streamlit Community Cloud.
3. Select **Create app**.
4. Choose the GitHub repository and `main` branch.
5. Set **Main file path** to:

```text
app.py
```

6. Deploy the app.

No secrets are required for the included version.

## Validate the repository

After installing the dependencies, run:

```bash
python smoke_test.py
```

The smoke test checks the Uzbekistan-only dataset, recommendation logic, commute calculations, and map generation.

## Use in Google Colab

Open `Uzbekistan_Education_Center_App_Colab.ipynb` in Colab. The notebook:

- installs dependencies
- validates the Uzbekistan-only CSV
- explains the recommendation architecture
- can package the repository as a ZIP
- provides optional commands for a temporary Streamlit preview

For permanent deployment, GitHub plus Streamlit Community Cloud is recommended instead of a Colab tunnel.

## Replace the demonstration data

Keep the same CSV filename:

```text
uzbekistan_education_centers.csv
```

Important columns include:

| Column | Purpose |
| --- | --- |
| `name` | Center name |
| `country` | Must be `Uzbekistan` |
| `region`, `city`, `district` | Uzbekistan location fields |
| `latitude`, `longitude` | Map coordinates |
| `min_grade`, `max_grade` | Grade eligibility |
| `min_age`, `max_age` | Age eligibility |
| `subjects`, `resources`, `utilities`, `languages` | Pipe-separated values |
| `open_schedule_json` | Weekly opening schedule as JSON |
| `events_json` | Programs/events as JSON |
| `monthly_fee_uzs` | Approximate monthly fee in UZS |
| `verification_status` | Verification and review status |

Example pipe-separated field:

```text
Mathematics|Physics|Python
```

Example schedule:

```json
[
  {"day": "Monday", "start": "14:00", "end": "20:00"},
  {"day": "Saturday", "start": "09:00", "end": "17:00"}
]
```

## Production improvements

For a real launch, consider adding:

- a verified center onboarding and approval workflow
- center-owner accounts for updating schedules and events
- PostgreSQL or Supabase storage
- real road and transit routing
- geocoding and address validation
- user favorites and notifications
- moderation, audit logs, and data-quality checks
- accessibility verification
- Uzbek Latin, Uzbek Cyrillic, Karakalpak, Russian, and English localization
- privacy controls and parental consent for student accounts

## Authorship declaration

This application was authored by **Mukhammadjon Khojikulov** under the mentorship of **Dr. Qingyang Xiao**. The recommendation factors and assumptions are disclosed in the app and this README so the results can be reviewed and audited.

## Responsible AI and privacy

- Do not collect unnecessary student personal information.
- Do not store sensitive student records in the public GitHub repository.
- Treat ranking as decision support, not a guarantee of education quality.
- Explain recommendation factors to users.
- Give centers a way to correct inaccurate information.
- Verify opening hours, prices, eligibility, and accessibility before public use.
- Do not publish phone numbers, names, or addresses without a lawful basis and permission.

## Portfolio description

> Built an Uzbekistan-focused AI-assisted education-center discovery platform with Python and Streamlit. Implemented multilingual content matching, eligibility and schedule checks, commute estimation, interactive mapping, transparent recommendation scoring, analytics, and GitHub/Streamlit deployment packaging.

## License

MIT License. The synthetic dataset is included only for demonstration and testing.
