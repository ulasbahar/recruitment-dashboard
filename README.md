# Recruitment Dashboard

## Overview
This repository contains a Streamlit-based recruitment dashboard designed to analyze and visualize recruitment data for a company's hiring process. The dashboard helps HR teams track candidate performance, identify trends, and optimize hiring strategies by providing interactive insights into candidate applications, interviews, and outcomes.

## Initial Objective
The primary goal of this project was to create a user-friendly tool to:
- Analyze recruitment data to identify the best-performing countries, roles, and candidate profiles.
- Track the recruitment funnel, bottlenecks, and time-to-hire metrics.
- Explore correlations between candidate experience, interview scores, and hiring success.
- Provide predictive insights and advanced analytics to improve decision-making.

We started with a dataset of 100 candidate records, including details like application dates, interview results, years of experience, and more, stored in `recruitment_data.csv`.

## Steps Taken
Here’s an overview of the development process:

1. **Data Preparation**:
   - Loaded and cleaned the `recruitment_data.csv` file using Pandas.
   - Converted dates, handled missing values, and calculated metrics like total time to hire.

2. **Dashboard Development**:
   - Built a multi-page Streamlit app with pages for:
     - **Overview**: Key metrics and recruitment funnel visualization.
     - **Country Analysis**: Hires, success rates, and experience by country.
     - **Role Analysis**: Success rates and salary expectations by role.
     - **Experience Analysis**: Experience distributions and correlations with hiring outcomes.
     - **Time-Based Trends**: Monthly hiring trends and time in process.
     - **Funnel & Bottlenecks**: Failure rates and role-specific funnel breakdowns.
     - **Predictive Insights**: Machine learning predictions for hiring likelihood.
     - **Advanced Analytics**: Statistical tests, time series analysis, and correlation matrices.
   - Used Plotly for interactive visualizations (funnels, bar charts, box plots, etc.).

3. **Interactivity**:
   - Added filters for roles, countries, and date ranges to allow users to explore data dynamically.
   - Included drill-downs and a candidate scoring tool for deeper insights.

4. **Testing and Refinement**:
   - Tested the dashboard with the dataset, fixing errors like datetime comparisons and serialization issues.
   - Enhanced analytics with statistical tests, seasonal decomposition, and feature importance analysis.

5. **Version Control**:
   - Initialized a Git repository, committed files (`app.py`, `recruitment_data.csv`, etc.), and pushed to GitHub using the guidance from xAI.

## Results
The dashboard successfully achieves its objectives, providing:
- **Clear Insights**: Identifies top-performing countries (e.g., Germany) and roles (e.g., PHP) based on hire rates.
- **Efficient Tracking**: Visualizes the recruitment funnel, showing drop-off points and average time to hire (e.g., ~10–15 days).
- **Data-Driven Decisions**: Predicts hiring likelihood with ~85% accuracy and highlights key factors like experience and interview scores.
- **Advanced Analytics**: Uncovers seasonal hiring patterns, significant experience differences, and feature correlations.

The dashboard is interactive, user-friendly, and ready for HR teams to use for ongoing recruitment analysis.

## Getting Started
### Prerequisites
- Python 3.9 or higher
- Required libraries: `streamlit`, `pandas`, `plotly`, `numpy`, `scikit-learn`, `statsmodels`, `scipy`

### Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/ulasbahar/recruitment-dashboard.git
   cd recruitment-dashboard
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
3. Ensure `recruitment_data.csv` is in the project folder with the correct structure (see `app.py` for details).

## Running the Dashboard
1. Run the Streamlit app:
    ```bash
    streamlit run app.py
2. Open your browser at http://localhost:8501 to interact with the dashboard.

## Files
 app.py: Main Streamlit application code.
 recruitment_data.csv: Dataset with candidate recruitment data.
 requirements.txt: List of Python dependencies.
 pages/1_Country_Analysis.py: Additional page for country-specific analysis (if applicable).