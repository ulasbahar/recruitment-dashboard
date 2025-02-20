import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Country Analysis", page_icon="🌍", layout="wide")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("recruitment_data.csv")

def main():
    st.title("🌍 Country Analysis")
    
    df = load_data()
    
    # Calculate country-wise metrics
    country_stats = df.groupby('Country').agg({
        'Candidate Name': 'count',
        'Offer': lambda x: (x == 'Pass').sum()
    }).reset_index()
    
    country_stats.columns = ['Country', 'Total Applicants', 'Total Hires']
    country_stats['Success Rate'] = (country_stats['Total Hires'] / country_stats['Total Applicants'] * 100).round(1)
    
    # Display metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Hiring Success by Country")
        fig = px.bar(country_stats, 
                    x='Country', 
                    y=['Total Applicants', 'Total Hires'],
                    barmode='group',
                    title="Applications vs Hires by Country")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Success Rate by Country")
        fig = px.bar(country_stats,
                    x='Country',
                    y='Success Rate',
                    title="Success Rate by Country (%)")
        st.plotly_chart(fig, use_container_width=True)
    
    # Experience Analysis by Country
    st.subheader("Experience Distribution by Country")
    fig = px.box(df,
                 x='Country',
                 y='Total Years of Experience',
                 title="Years of Experience Distribution by Country")
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed Country Stats
    st.subheader("Detailed Country Statistics")
    st.dataframe(country_stats)

if __name__ == "__main__":
    main()
