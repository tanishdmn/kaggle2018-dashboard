# Kaggle 2018 Data Science Survey Dashboard

An interactive dashboard for exploring responses from the Kaggle 2018 Machine Learning and Data Science Survey. It includes demographic and technology analysis, plus machine learning tools for Python recommendation and salary-band prediction.

## Features

- **Overview KPIs:** respondent and country counts, role, education, age, experience, and compensation summaries.
- **Demographics:** charts for country, age, gender, roles, industries, education, and compensation.
- **Tools and attitudes:** analysis of programming and visualization tools, machine learning frameworks, cloud services, and survey responses about ML.
- **Skills and time:** programming languages, work activities, and time spent on data science tasks.
- **ML predictors:** estimates whether a profile would recommend Python and predicts a salary band from profile details.
- **Interactive filters:** narrow dashboard results by available respondent characteristics.

## Data

The dashboard loads prepared CSV tables stored in the repository:

| File | Contents |
|---|---|
| `dim_demographics_prepared.csv` | Respondent demographics, role, education, experience, industry, and compensation |
| `dim_tools_prepared.csv` | Analysis tools and attitudes toward machine learning |
| `dim_ml_framworks_prepared.csv` | Machine learning frameworks used by respondents |
| `dim_skills_experience_prepared.csv` | Skills, programming languages, work activities, and time allocation |
| `fact_respondent_clean.csv` | Cleaned respondent data used for additional survey charts and filters |

The survey is historical and self-reported. Results describe the survey respondents; they should not be treated as a representative measure of every data science professional.

## Machine Learning Models

The ML tab uses saved model artifacts in the repository:

- `python_recommender_model.joblib` estimates whether a profile would recommend Python as a first programming language.
- `salary_prediction_model.joblib` predicts a salary range and provides a regression estimate.

The app reports model scores based on its training and evaluation process. Predictions are estimates based on the survey data, not guarantees about an individual’s recommendation or salary.

## Run Locally

You’ll need Python 3.9 or newer.

1. Clone the repository and open its folder.
2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Configure the login credentials as described below.
4. Start the dashboard:

   ```bash
   streamlit run app.py
   ```

## Login Configuration

The app reads its login credentials from Streamlit secrets. For local development, add this to `.streamlit/secrets.toml`:

```toml
[auth]
username = "your_username"
password = "your_password"
```

Keep `secrets.toml` private. Do not commit real credentials to GitHub. For Streamlit Community Cloud, add the same `[auth]` settings in the app’s **Settings → Secrets** page.

## Deploy on Streamlit Community Cloud

1. Push the project to a GitHub repository.
2. In Streamlit Community Cloud, create an app and select the repository and branch.
3. Set the app’s entrypoint to `app.py`.
4. Add the `[auth]` credentials under the app’s Secrets settings.
5. Deploy the app.

## Main Files

- `app.py` — dashboard, filters, charts, and data loading.
- `ml_predictor_tab.py` — Python recommender and salary predictor interfaces.
- `requirements.txt` — Python dependencies.
- `*.csv` — prepared survey tables.
- `*.joblib` — saved machine learning model artifacts.

## Technology

Python · Streamlit · Pandas · NumPy · Plotly · Joblib
