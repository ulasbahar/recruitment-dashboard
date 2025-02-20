import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import stats
import plotly.figure_factory as ff

st.set_page_config(
    page_title="Recruitment Dashboard",
    page_icon="👥",
    layout="wide"
)

# Load and prepare data
@st.cache_data
def load_data():
    try:
        # Read CSV with explicit encoding
        df = pd.read_csv("recruitment_data.csv", encoding='utf-8')
        
        # Log available columns for debugging
        st.write("Available columns in dataset:", df.columns.tolist())
        
        # Convert Application Date to date objects (matching your current setup)
        df['Application Date'] = pd.to_datetime(df['Application Date']).dt.date
        
        # Handle missing values in interview result columns
        interview_columns = [
            'Result of First Interview', 'Result of Product Interview',
            'Result of Code Live', 'Result of Culture Fit', 'Result of Offer'
        ]
        for col in interview_columns:
            if col in df.columns:
                df[col] = df[col].fillna('Did Not Reach')
            else:
                st.warning(f"Column '{col}' not found in dataset. Skipping...")
        
        # Handle missing values in time columns (convert to numeric, NA stays NA)
        time_columns = [
            'Time: App to First (days)', 'Time: First to Product/Code (days)',
            'Time: Product/Code to Culture (days)'
        ]
        for col in time_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            else:
                st.warning(f"Time column '{col}' not found in dataset. Skipping...")
        
        # Calculate total time to hire for hired candidates
        df['Total Time to Hire'] = df.apply(
            lambda row: (
                row['Time: App to First (days)'] +
                (row['Time: First to Product/Code (days)'] if pd.notna(row['Time: First to Product/Code (days)']) else 0) +
                (row['Time: Product/Code to Culture (days)'] if pd.notna(row['Time: Product/Code to Culture (days)']) else 0)
            ) if row['Result of Offer'] == 'Hired' else np.nan,
            axis=1
        )
        return df
    except FileNotFoundError:
        st.error("File 'recruitment_data.csv' not found. Please ensure it's in the same directory as this script.")
        return None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Overview Page
def show_overview(df):
    st.header("Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Candidates", len(df))
    with col2:
        hires = len(df[df['Result of Offer'] == 'Hired'])
        st.metric("Total Hires", hires)
    with col3:
        st.metric("Success Rate", f"{(hires / len(df) * 100):.1f}%")
    with col4:
        avg_time = df[df['Result of Offer'] == 'Hired']['Total Time to Hire'].mean()
        st.metric("Avg Time to Hire", f"{avg_time:.1f} days" if pd.notna(avg_time) else "N/A")

    st.header("Recruitment Funnel")
    # Filters for interactivity
    roles = st.multiselect("Select Roles", df['Role'].unique(), default=df['Role'].unique())
    countries = st.multiselect("Select Countries", df['Country'].unique(), default=df['Country'].unique())
    date_range = st.date_input("Select Date Range", [df['Application Date'].min(), df['Application Date'].max()], key="overview_date")
    
    # Convert date_range to date objects (matching your Application Date format)
    start_date = date_range[0]
    end_date = date_range[1]
    
    df_filtered = df[df['Role'].isin(roles) & df['Country'].isin(countries) & 
                    (df['Application Date'] >= start_date) & (df['Application Date'] <= end_date)]

    # Overall funnel
    total_applications = len(df_filtered)
    first_interview_pass = len(df_filtered[df_filtered['Result of First Interview'] == 'Pass'])
    product_interview_pass = len(df_filtered[df_filtered['Result of Product Interview'] == 'Pass'])
    code_live_pass = len(df_filtered[df_filtered['Result of Code Live'] == 'Pass'])
    culture_fit_pass = len(df_filtered[df_filtered['Result of Culture Fit'] == 'Pass'])
    hired = len(df_filtered[df_filtered['Result of Offer'] == 'Hired'])
    
    stages = ['Applied', 'First Interview', 'Product Interview (PM)', 'Code Live (PHP/JS)', 'Culture Fit', 'Hired']
    stage_counts = [
        total_applications,
        first_interview_pass,
        product_interview_pass,
        code_live_pass,
        culture_fit_pass,
        hired
    ]
    
    fig = go.Figure(go.Funnel(
        y=stages,
        x=stage_counts,
        textinfo="value+percent initial"
    ))
    fig.update_layout(title_text="Overall Recruitment Funnel")
    st.plotly_chart(fig, use_container_width=True)

    # Role-specific funnels
    st.subheader("Role-Specific Funnels")
    for role in df_filtered['Role'].unique():
        role_df = df_filtered[df_filtered['Role'] == role]
        if role == 'PM':
            stages_role = ['Applied', 'First Interview', 'Product Interview', 'Culture Fit', 'Hired']
            counts = [
                len(role_df),
                len(role_df[role_df['Result of First Interview'] == 'Pass']),
                len(role_df[role_df['Result of Product Interview'] == 'Pass']),
                len(role_df[role_df['Result of Culture Fit'] == 'Pass']),
                len(role_df[role_df['Result of Offer'] == 'Hired'])
            ]
        else:  # PHP or JS
            stages_role = ['Applied', 'First Interview', 'Code Live', 'Culture Fit', 'Hired']
            counts = [
                len(role_df),
                len(role_df[role_df['Result of First Interview'] == 'Pass']),
                len(role_df[role_df['Result of Code Live'] == 'Pass']),
                len(role_df[role_df['Result of Culture Fit'] == 'Pass']),
                len(role_df[role_df['Result of Offer'] == 'Hired'])
            ]
        fig_role = go.Figure(go.Funnel(
            y=stages_role,
            x=counts,
            textinfo="value+percent initial"
        ))
        fig_role.update_layout(title_text=f"Recruitment Funnel for {role}")
        st.plotly_chart(fig_role, use_container_width=True)

# Country Analysis Page
def show_country_analysis(df):
    st.header("Country Analysis")
    # Filters
    roles = st.multiselect("Select Roles", df['Role'].unique(), default=df['Role'].unique())
    date_range = st.date_input("Select Date Range", [df['Application Date'].min(), df['Application Date'].max()], key="country_date")
    
    # Convert date_range to date objects (matching your Application Date format)
    start_date = date_range[0]
    end_date = date_range[1]
    
    df_filtered = df[df['Role'].isin(roles) & 
                    (df['Application Date'] >= start_date) & (df['Application Date'] <= end_date)]

    hires_by_country = df_filtered[df_filtered['Result of Offer'] == 'Hired'].groupby('Country').size().reset_index(name='Hires')
    total_by_country = df_filtered.groupby('Country').size()
    success_rate = (df_filtered[df_filtered['Result of Offer'] == 'Hired'].groupby('Country').size() / 
                    total_by_country * 100).reset_index(name='Success Rate')
    
    avg_exp_by_country = df_filtered.groupby('Country')['Total Years of Experience'].mean().reset_index(name='Avg Experience')
    
    fig1 = px.bar(hires_by_country, x='Country', y='Hires', title="Hires by Country")
    fig2 = px.bar(success_rate, x='Country', y='Success Rate', title="Success Rate by Country (%)")
    fig3 = px.bar(avg_exp_by_country, x='Country', y='Avg Experience', title="Average Experience by Country")
    
    st.plotly_chart(fig1, use_container_width=True)
    st.plotly_chart(fig2, use_container_width=True)
    st.plotly_chart(fig3, use_container_width=True)

    # Correlation between experience and success
    success_exp = pd.merge(avg_exp_by_country, success_rate, on='Country')
    fig4 = px.scatter(success_exp, x='Avg Experience', y='Success Rate', 
                     title="Experience vs. Success Rate by Country", trendline="ols")
    st.plotly_chart(fig4, use_container_width=True)

    # Drill-down: Show top candidates by country
    country = st.selectbox("Select Country for Details", df_filtered['Country'].unique())
    country_df = df_filtered[df_filtered['Country'] == country]
    st.subheader(f"Candidates from {country}")
    st.dataframe(country_df[['Candidate Name', 'Role', 'Result of Offer', 'Total Years of Experience']])

# Role Analysis Page
def show_role_analysis(df):
    st.header("Role Analysis")
    # Filters
    countries = st.multiselect("Select Countries", df['Country'].unique(), default=df['Country'].unique())
    date_range = st.date_input("Select Date Range", [df['Application Date'].min(), df['Application Date'].max()], key="role_date")
    
    # Convert date_range to date objects (matching your Application Date format)
    start_date = date_range[0]
    end_date = date_range[1]
    
    df_filtered = df[df['Country'].isin(countries) & 
                    (df['Application Date'] >= start_date) & (df['Application Date'] <= end_date)]

    hires_by_role = df_filtered[df_filtered['Result of Offer'] == 'Hired'].groupby('Role').size()
    total_by_role = df_filtered.groupby('Role').size()
    success_by_role = (hires_by_role / total_by_role * 100).reset_index(name='Success Rate')
    success_by_role.columns = ['Role', 'Success Rate']
    
    fig = px.bar(success_by_role, x='Role', y='Success Rate', 
                 title="Success Rate by Role (%)")
    st.plotly_chart(fig, use_container_width=True)

    # Success rate by role and country
    success_by_role_country = (df_filtered[df_filtered['Result of Offer'] == 'Hired'].groupby(['Role', 'Country']).size() / 
                             df_filtered.groupby(['Role', 'Country']).size() * 100).reset_index(name='Success Rate')
    fig2 = px.bar(success_by_role_country, x='Role', y='Success Rate', color='Country',
                 title="Success Rate by Role and Country")
    st.plotly_chart(fig2, use_container_width=True)

    # Salary expectations by role
    salary_by_role = df_filtered.groupby('Role')['Salary Expectations'].mean().reset_index()
    fig3 = px.bar(salary_by_role, x='Role', y='Salary Expectations',
                 title="Average Salary Expectations by Role")
    st.plotly_chart(fig3, use_container_width=True)

# Experience Analysis Page
def show_experience_analysis(df):
    st.header("Experience Analysis")
    # Filters
    roles = st.multiselect("Select Roles", df['Role'].unique(), default=df['Role'].unique())
    countries = st.multiselect("Select Countries", df['Country'].unique(), default=df['Country'].unique())
    date_range = st.date_input("Select Date Range", [df['Application Date'].min(), df['Application Date'].max()], key="exp_date")
    
    # Convert date_range to date objects (matching your Application Date format)
    start_date = date_range[0]
    end_date = date_range[1]
    
    df_filtered = df[df['Role'].isin(roles) & df['Country'].isin(countries) & 
                    (df['Application Date'] >= start_date) & (df['Application Date'] <= end_date)]

    # Calculate average experience for different stages
    exp_first = df_filtered[df_filtered['Result of First Interview'] == 'Pass']['Total Years of Experience'].mean()
    exp_hired = df_filtered[df_filtered['Result of Offer'] == 'Hired']['Total Years of Experience'].mean()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Avg Experience (First Interview Pass)", f"{exp_first:.1f} years")
    with col2:
        st.metric("Avg Experience (Hired)", f"{exp_hired:.1f} years")
    
    # Experience distribution by interview stage
    df_filtered['Interview Stage'] = df_filtered['Result of First Interview'].map({
        'Pass': 'Passed First Interview', 
        'Fail': 'Failed First Interview', 
        'Did Not Reach': 'Did Not Reach First Interview'
    })
    fig = px.box(df_filtered, x='Interview Stage', y='Total Years of Experience', 
                 title="Experience Distribution by Interview Stage")
    st.plotly_chart(fig, use_container_width=True)

    # Experience vs. Offer Outcome with optimal range
    df_filtered['Hiring Status'] = df_filtered['Result of Offer'].map({
        'Hired': 'Hired', 
        'Pass': 'Not Hired', 
        'Fail': 'Not Hired', 
        'Did Not Reach': 'Not Hired'
    })
    fig2 = px.box(df_filtered, x='Hiring Status', y='Total Years of Experience', color='Role',
                  title="Experience Distribution by Offer Outcome and Role")
    st.plotly_chart(fig2, use_container_width=True)

    # Optimal experience range per role
    for role in df_filtered['Role'].unique():
        role_exp = df_filtered[df_filtered['Role'] == role]['Total Years of Experience']
        q25, q75 = role_exp.quantile([0.25, 0.75])
        optimal_range = f"{q25:.1f}–{q75:.1f} years"
        st.write(f"Optimal Experience Range for {role}: {optimal_range}")

# Time-Based Trends Page
def show_time_trends(df):
    st.header("Time-Based Trends")
    # Filters
    roles = st.multiselect("Select Roles", df['Role'].unique(), default=df['Role'].unique())
    countries = st.multiselect("Select Countries", df['Country'].unique(), default=df['Country'].unique())
    date_range = st.date_input("Select Date Range", [df['Application Date'].min(), df['Application Date'].max()], key="time_date")
    
    # Convert date_range to date objects (matching your Application Date format)
    start_date = date_range[0]
    end_date = date_range[1]
    
    df_filtered = df[df['Role'].isin(roles) & df['Country'].isin(countries) & 
                    (df['Application Date'] >= start_date) & (df['Application Date'] <= end_date)]

    # Monthly hiring trends (convert Period to string)
    df_filtered['Month'] = pd.to_datetime(df_filtered['Application Date']).dt.to_period('M').astype(str)
    monthly_hires = df_filtered[df_filtered['Result of Offer'] == 'Hired'].groupby('Month').size().reset_index(name='Hires')
    
    fig1 = px.line(monthly_hires, x='Month', y='Hires', 
                   title="Monthly Hiring Trends")
    st.plotly_chart(fig1, use_container_width=True)
    
    # Average time in process and applications vs. hires
    time_cols = ['Time: App to First (days)', 'Time: First to Product/Code (days)', 
                 'Time: Product/Code to Culture (days)']
    avg_times = df_filtered[df_filtered['Result of Offer'] == 'Hired'][time_cols].mean()
    
    st.subheader("Average Time in Process (Hired Candidates)")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Application to First", f"{avg_times['Time: App to First (days)']:.1f} days")
    with col2:
        st.metric("First to Product/Code", f"{avg_times['Time: First to Product/Code (days)']:.1f} days")
    with col3:
        st.metric("Product/Code to Culture", f"{avg_times['Time: Product/Code to Culture (days)']:.1f} days")

    # Applications vs. Hires over time
    apps_by_month = df_filtered.groupby('Month').size().reset_index(name='Applications')
    combined = pd.merge(monthly_hires, apps_by_month, on='Month')
    fig2 = px.line(combined, x='Month', y=['Hires', 'Applications'], 
                  title="Applications vs. Hires Over Time")
    st.plotly_chart(fig2, use_container_width=True)

# Funnel and Bottlenecks Page
def show_funnel_bottlenecks(df):
    st.header("Funnel & Bottlenecks")
    # Filters
    roles = st.multiselect("Select Roles", df['Role'].unique(), default=df['Role'].unique())
    countries = st.multiselect("Select Countries", df['Country'].unique(), default=df['Country'].unique())
    date_range = st.date_input("Select Date Range", [df['Application Date'].min(), df['Application Date'].max()], key="funnel_date")
    
    # Convert date_range to date objects (matching your Application Date format)
    start_date = date_range[0]
    end_date = date_range[1]
    
    df_filtered = df[df['Role'].isin(roles) & df['Country'].isin(countries) & 
                    (df['Application Date'] >= start_date) & (df['Application Date'] <= end_date)]

    # Calculate stage-wise failure rates
    stages = {
        'First Interview': df_filtered['Result of First Interview'],
        'Product Interview': df_filtered[df_filtered['Role'] == 'PM']['Result of Product Interview'],
        'Code Live': df_filtered[df_filtered['Role'].isin(['PHP', 'JS'])]['Result of Code Live'],
        'Culture Fit': df_filtered['Result of Culture Fit']
    }
    
    failure_rates = {}
    for stage, data in stages.items():
        total = len(data.dropna())
        if total > 0:
            failures = len(data[data != 'Pass'])
            failure_rates[stage] = (failures / total * 100)
        else:
            failure_rates[stage] = 0
    
    # Create failure rates visualization
    failure_df = pd.DataFrame({
        'Stage': list(failure_rates.keys()),
        'Failure Rate': list(failure_rates.values())
    })
    
    fig = px.bar(failure_df, x='Stage', y='Failure Rate',
                 title="Failure Rates by Interview Stage (%)")
    fig.update_traces(marker_color='red')
    st.plotly_chart(fig, use_container_width=True)
    
    # Role-specific failure rates
    for role in df_filtered['Role'].unique():
        role_data = df_filtered[df_filtered['Role'] == role]
        role_stages = {
            'First Interview': role_data['Result of First Interview'],
            'Product Interview' if role == 'PM' else 'Code Live': 
                role_data['Result of Product Interview'] if role == 'PM' else role_data['Result of Code Live'],
            'Culture Fit': role_data['Result of Culture Fit']
        }
        role_failure_rates = {}
        for stage, data in role_stages.items():
            total = len(data.dropna())
            if total > 0:
                failures = len(data[data != 'Pass'])
                role_failure_rates[stage] = (failures / total * 100)
            else:
                role_failure_rates[stage] = 0
        st.subheader(f"Failure Rates for {role}")
        role_failure_df = pd.DataFrame({
            'Stage': list(role_failure_rates.keys()),
            'Failure Rate': list(role_failure_rates.values())
        })
        fig_role = px.bar(role_failure_df, x='Stage', y='Failure Rate',
                         title=f"Failure Rates for {role}")
        st.plotly_chart(fig_role, use_container_width=True)

    # Time analysis with distribution
    st.subheader("Time Spent in Each Stage")
    time_cols = ['Time: App to First (days)', 'Time: First to Product/Code (days)', 
                 'Time: Product/Code to Culture (days)']
    
    time_stats = df_filtered[time_cols].agg(['mean', 'median', 'std']).round(1)
    st.dataframe(time_stats)
    
    # Distribution of time spent
    for col in time_cols:
        if col in df_filtered.columns and not df_filtered[col].isna().all():
            fig_time = px.histogram(df_filtered, x=col, title=f"Distribution of {col}")
            st.plotly_chart(fig_time, use_container_width=True)

# Predictive Insights Page
def show_predictive_insights(df):
    st.header("Predictive Insights")
    # Filters
    roles = st.multiselect("Select Roles", df['Role'].unique(), default=df['Role'].unique())
    countries = st.multiselect("Select Countries", df['Country'].unique(), default=df['Country'].unique())
    date_range = st.date_input("Select Date Range", [df['Application Date'].min(), df['Application Date'].max()], key="pred_date")
    
    # Convert date_range to date objects (matching your Application Date format)
    start_date = date_range[0]
    end_date = date_range[1]
    
    df_filtered = df[df['Role'].isin(roles) & df['Country'].isin(countries) & 
                    (df['Application Date'] >= start_date) & (df['Application Date'] <= end_date)]

    # Prepare data for prediction
    X = df_filtered[['Total Years of Experience', 'Interview Score: First', 'Salary Expectations']]
    y = df_filtered['Result of Offer'].apply(lambda x: 1 if x == 'Hired' else 0)
    
    if len(X) > 0 and X.notna().all().all():
        X = X.fillna(X.mean())  # Handle missing values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        
        st.metric("Prediction Accuracy", f"{accuracy:.2f}")
        
        # Feature importance
        importance = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
        fig = px.bar(importance, x='Feature', y='Importance', 
                    title="Feature Importance for Hiring Prediction")
        st.plotly_chart(fig, use_container_width=True)
        
        # Predict for new candidates (example)
        st.subheader("Predict Hiring Likelihood")
        exp = st.slider("Years of Experience", 3, 9, 5)
        score = st.slider("First Interview Score", 30, 100, 80)
        salary = st.slider("Salary Expectation", 45000, 60000, 50000)
        
        prediction = model.predict_proba([[exp, score, salary]])[0]
        st.write(f"Probability of Being Hired: {prediction[1]:.2%}")
    else:
        st.warning("Not enough data for predictive modeling. Please adjust filters.")

# Advanced Analytics Page
def show_advanced_analytics(df):
    st.header("Advanced Analytics")
    
    # Filters
    roles = st.multiselect("Select Roles", df['Role'].unique(), default=df['Role'].unique(), key="adv_roles")
    countries = st.multiselect("Select Countries", df['Country'].unique(), default=df['Country'].unique(), key="adv_countries")
    date_range = st.date_input("Select Date Range", [df['Application Date'].min(), df['Application Date'].max()], key="adv_date")
    
    # Convert date_range to date objects (matching your Application Date format)
    start_date = date_range[0]
    end_date = date_range[1]
    
    df_filtered = df[df['Role'].isin(roles) & df['Country'].isin(countries) & 
                    (df['Application Date'] >= start_date) & (df['Application Date'] <= end_date)]

    # 1. Interview Score Analysis
    st.subheader("Interview Score Analysis")
    score_cols = ['Interview Score: First', 'Interview Score: Product', 'Interview Score: Code Live', 'Interview Score: Culture Fit']
    
    # Score distribution by outcome
    for col in score_cols:
        if col in df_filtered.columns:
            fig = px.box(df_filtered, x='Result of Offer', y=col, 
                        title=f"{col} Distribution by Hiring Outcome")
            st.plotly_chart(fig, use_container_width=True)
    
    # 2. Statistical Tests
    st.subheader("Statistical Analysis")
    
    # T-test for experience between hired and not hired
    hired_exp = df_filtered[df_filtered['Result of Offer'] == 'Hired']['Total Years of Experience']
    not_hired_exp = df_filtered[df_filtered['Result of Offer'] != 'Hired']['Total Years of Experience']
    
    if len(hired_exp) > 0 and len(not_hired_exp) > 0:
        t_stat, p_value = stats.ttest_ind(hired_exp, not_hired_exp)
        st.write(f"T-test p-value for experience difference: {p_value:.4f}")
        if p_value < 0.05:
            st.write("There is a significant difference in experience between hired and not hired candidates")
    
    # 3. Time Series Decomposition
    st.subheader("Hiring Trends Decomposition")
    
    # Prepare time series data
    df_ts = df_filtered[df_filtered['Result of Offer'] == 'Hired'].copy()
    df_ts['Month'] = pd.to_datetime(df_ts['Application Date']).dt.to_period('M').astype(str)
    monthly_hires = df_ts.groupby('Month').size()
    
    if len(monthly_hires) >= 24:  # Require at least 2 years (24 months) for seasonal decomposition
        # Convert to datetime index
        monthly_hires.index = pd.to_datetime(monthly_hires.index, format='%Y-%m')
        
        # Perform decomposition
        decomposition = seasonal_decompose(monthly_hires, period=12)
        
        # Plot components
        fig1 = px.line(x=decomposition.trend.index, y=decomposition.trend.values, 
                      title="Trend Component")
        fig2 = px.line(x=decomposition.seasonal.index, y=decomposition.seasonal.values, 
                      title="Seasonal Component")
        fig3 = px.line(x=decomposition.resid.index, y=decomposition.resid.values, 
                      title="Residual Component")
        
        st.plotly_chart(fig1, use_container_width=True)
        st.plotly_chart(fig2, use_container_width=True)
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.warning("Not enough data (less than 24 months) for seasonal decomposition. Showing raw hiring trends instead.")
        fig = px.line(monthly_hires.reset_index(), x='Month', y=0, 
                     title="Raw Monthly Hiring Trends (Insufficient Data for Seasonal Analysis)")
        st.plotly_chart(fig, use_container_width=True)
    
    # 4. Correlation Analysis
    st.subheader("Correlation Analysis")
    
    numeric_cols = ['Total Years of Experience', 'Interview Score: First', 
                   'Salary Expectations', 'Time: App to First (days)']
    
    corr_df = df_filtered[numeric_cols].corr()
    
    fig = px.imshow(corr_df, 
                    labels=dict(color="Correlation"),
                    x=corr_df.columns,
                    y=corr_df.columns)
    fig.update_layout(title="Correlation Matrix of Numeric Features")
    st.plotly_chart(fig, use_container_width=True)
    
    # 5. Advanced ML Insights
    st.subheader("Advanced ML Insights")
    
    # Prepare features for ML
    features = ['Total Years of Experience', 'Interview Score: First', 
               'Salary Expectations', 'Time: App to First (days)']
    
    X = df_filtered[features].fillna(method='ffill')
    y = (df_filtered['Result of Offer'] == 'Hired').astype(int)
    
    if len(X) > 20:  # Minimum samples for ML
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Feature importance
        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig = px.bar(importance_df, x='Feature', y='Importance',
                    title="Feature Importance Analysis")
        st.plotly_chart(fig, use_container_width=True)
        
        # Confusion Matrix
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        fig = px.imshow(cm,
                       labels=dict(x="Predicted", y="Actual"),
                       x=['Not Hired', 'Hired'],
                       y=['Not Hired', 'Hired'])
        fig.update_layout(title="Confusion Matrix")
        st.plotly_chart(fig, use_container_width=True)
        
        # Classification Report
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.write("Classification Report:")
        st.dataframe(report_df)
    
    # 6. Candidate Scoring System
    st.subheader("Candidate Scoring System")
    
    if st.checkbox("Show Candidate Scoring Tool"):
        exp = st.slider("Years of Experience", 0, 15, 5)
        score = st.slider("First Interview Score", 0, 100, 75)
        salary = st.slider("Salary Expectations", 40000, 100000, 60000)
        time_to_first = st.slider("Time to First Interview (days)", 1, 30, 7)
        
        # Create feature vector
        candidate_features = np.array([[exp, score, salary, time_to_first]])
        candidate_features_scaled = scaler.transform(candidate_features)
        
        # Get prediction and probability
        prob = model.predict_proba(candidate_features_scaled)[0][1]
        
        # Create gauge chart for probability
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = prob * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Hiring Probability"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': "red"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "green"}
                ]
            }
        ))
        st.plotly_chart(fig, use_container_width=True)

# Main app with sidebar navigation
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", [
        "Overview", "Country Analysis", "Role Analysis", 
        "Experience Analysis", "Time-Based Trends", "Funnel & Bottlenecks",
        "Predictive Insights", "Advanced Analytics"
    ])
    
    df = load_data()
    if df is None:
        st.warning("Please ensure 'recruitment_data.csv' is in the correct directory.")
        return
    
    if page == "Overview":
        show_overview(df)
    elif page == "Country Analysis":
        show_country_analysis(df)
    elif page == "Role Analysis":
        show_role_analysis(df)
    elif page == "Experience Analysis":
        show_experience_analysis(df)
    elif page == "Time-Based Trends":
        show_time_trends(df)
    elif page == "Funnel & Bottlenecks":
        show_funnel_bottlenecks(df)
    elif page == "Predictive Insights":
        show_predictive_insights(df)
    elif page == "Advanced Analytics":
        show_advanced_analytics(df)

if __name__ == "__main__":
    main()