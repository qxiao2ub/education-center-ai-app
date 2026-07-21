# AI-Based Education Center Cross-Function Website App

A portfolio-ready Streamlit web app that helps students, parents, and teachers find education centers based on location, availability, eligibility, commute time, learning needs, events, resources, affiliations, and utilities.

> Note: the starter dataset in this repository is synthetic demo data. Replace it with verified public/open data for the country or region you want to support.

## Project goals

This project demonstrates how to build an AI-assisted education-center finder that can:

- filter education centers by country, region, city, center type, subject, resource, date, and time
- check whether a center is open for a chosen day and time
- estimate commute time from a selected location
- check grade, age, and membership/referral eligibility
- display centers on an interactive regional map
- show events, resources, languages, accessibility notes, utilities, and affiliations
- rank centers using a transparent AI recommendation score
- package the project as a Streamlit website app

## Features

| Feature | Description |
| --- | --- |
| Personalized profile | Students, parents, and teachers can enter grade, age, membership/referral status, and learning goals. |
| Smart filters | Filters for center type, region, city, subject interests, resources, keyword, eligibility, and open hours. |
| Availability engine | Reads JSON-style weekly schedules and determines whether a center is open at the selected date/time. |
| Eligibility logic | Checks grade range, age range, and membership/referral requirements. |
| Commute estimator | Uses latitude/longitude and a haversine distance formula to estimate commute time by walk, bike, transit, or drive. |
| Interactive map | Uses Folium and Streamlit-Folium to show education centers on a regional map. |
| AI recommendation | Uses TF-IDF text matching plus availability, eligibility, commute, and cost signals to rank centers. |
| GitHub-ready | Includes `app.py`, `requirements.txt`, `education_centers_seed.csv`, and this `README.md`. |

## Tech stack

- Python
- Google Colab / Jupyter Notebook
- Streamlit
- Pandas
- NumPy
- scikit-learn
- Folium
- Streamlit-Folium

## Repository structure

```text
.
├── app.py
├── education_centers_seed.csv
├── requirements.txt
└── README.md
```

## Run locally

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## Run in Google Colab

1. Open the notebook `AI_Education_Center_App_Colab.ipynb` in Google Colab.
2. Run the setup cells to create the dataset, `app.py`, `requirements.txt`, and `README.md`.
3. Run the Streamlit launch cell.
4. Open the public localtunnel link generated in the notebook output.

## Deploy on Streamlit Community Cloud

1. Create a public GitHub repository.
2. Upload these files:
   - `app.py`
   - `education_centers_seed.csv`
   - `requirements.txt`
   - `README.md`
3. Go to Streamlit Community Cloud.
4. Connect your GitHub account and choose the repository.
5. Set the main file path to `app.py`.
6. Deploy the app.

## Customize for your country or region

Replace the synthetic CSV with verified open/public data. Keep these columns when possible:

- `center_id`
- `name`
- `country`
- `region`
- `city`
- `center_type`
- `address`
- `latitude`
- `longitude`
- `min_grade`
- `max_grade`
- `min_age`
- `max_age`
- `cost_level`
- `membership_required`
- `subjects`
- `resources`
- `utilities`
- `affiliations`
- `languages`
- `accessibility`
- `website`
- `contact`
- `open_schedule_json`
- `eligibility_notes`
- `events_json`
- `description`

For pipe-delimited fields such as `subjects`, use this format:

```text
Math|Reading|Writing|Homework Help
```

For `open_schedule_json`, use a JSON list like this:

```json
[
  {"day": "Monday", "start": "09:00", "end": "17:00"},
  {"day": "Tuesday", "start": "09:00", "end": "17:00"}
]
```

For `events_json`, use a JSON list like this:

```json
[
  {"date": "2026-07-08", "title": "SAT Strategy Night", "audience": "High school students"}
]
```

## AI scoring method

The app uses a transparent content-based recommendation score:

```python
score = 0.42 * content_match + 0.18 * eligibility + 0.16 * open_status + 0.14 * distance + 0.10 * cost
```

The content score is computed with TF-IDF similarity between the user's selected interests/goals and each center's subjects, resources, events, description, languages, and utilities.

## Portfolio and college application angle

This project is suitable for a student portfolio because it demonstrates:

- data modeling
- full-stack Python web app development
- applied AI recommendation logic
- geospatial visualization
- UI/UX design for students, parents, and teachers
- ethical handling of education data
- deployment and GitHub documentation

Suggested resume bullet:

> Built an AI-assisted Streamlit web app that helps students, parents, and teachers search, filter, map, and rank education centers using availability logic, eligibility checks, commute estimation, and a transparent recommendation score.

## Responsible data note

Do not store private student records, exact home addresses, medical information, protected education records, or other sensitive personal data in this app. For a production system, add secure authentication, consent, encryption, audit logs, and compliance review before handling real users or student data.

## License

MIT License is recommended for an open-source portfolio project. Add a `LICENSE` file if you publish this repository.
