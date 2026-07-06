import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import mlflow
import os

#  Page config 
st.set_page_config(
    page_title="PatrolIQ - Smart Safety Analytics",
    page_icon="🚔",
    layout="wide"
)

@st.cache_data
def load_data():
    file_path = 'data/crime_final.csv'
    if not os.path.exists(file_path):
        os.makedirs('data', exist_ok=True)
        from huggingface_hub import hf_hub_download
        hf_hub_download(
            repo_id="Kaldor4evr/patroliq-chicago-crime",
            filename="crime_final.csv",
            repo_type="dataset",
            local_dir="data"
        )
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

df = load_data()

#  Sidebar navigation 
st.sidebar.title("🚔 PatrolIQ")
st.sidebar.markdown("Smart Safety Analytics Platform")
page = st.sidebar.selectbox("Navigate", [
    "📊 Overview",
    "🗺️ Geographic Clustering",
    "⏰ Temporal Analysis",
    "📉 Dimensionality Reduction",
    "🧪 MLflow Results"
])

# PAGE 1 — OVERVIEW

if page == "📊 Overview":
    st.title("📊 Crime Overview — Chicago")

    # ── Global filter ──────────────────────────────────────────
    crime_types = ['All'] + sorted(df['Primary Type'].unique().tolist())
    selected_crime = st.selectbox("🔍 Filter by Crime Type", crime_types)

    filtered = df if selected_crime == 'All' else df[df['Primary Type'] == selected_crime]

    # ── KPI metrics ────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", f"{len(filtered):,}")
    col2.metric("Crime Types", filtered['Primary Type'].nunique())
    col3.metric("Arrest Rate", f"{filtered['Arrest'].mean()*100:.1f}%")
    col4.metric("Domestic Incidents", f"{filtered['Domestic'].mean()*100:.1f}%")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top 10 Crime Types")
        top10 = filtered['Primary Type'].value_counts().head(10).reset_index()
        top10.columns = ['Crime Type', 'Count']
        fig = px.bar(top10, x='Count', y='Crime Type', orientation='h',
                     color='Count', color_continuous_scale='Reds')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Arrest Rate by Crime Type")
        arrest_rate = filtered.groupby('Primary Type')['Arrest'].mean().sort_values(ascending=False).head(10).reset_index()
        arrest_rate.columns = ['Crime Type', 'Arrest Rate']
        fig = px.bar(arrest_rate, x='Arrest Rate', y='Crime Type', orientation='h',
                     color='Arrest Rate', color_continuous_scale='Greens')
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    st.subheader("📅 Crime Trend Over Years")
    filtered['Year'] = filtered['Date'].dt.year
    yearly = filtered.groupby('Year').size().reset_index(name='Count')
    fig = px.line(yearly, x='Year', y='Count', markers=True,
                  color_discrete_sequence=['orange'],
                  title=f"Yearly Trend — {selected_crime}")
    fig.update_layout(xaxis=dict(dtick=1))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Crime Severity Distribution")
    severity_counts = df['Crime_Severity'].value_counts().sort_index().reset_index()
    severity_counts.columns = ['Severity Score', 'Count']
    severity_counts['Severity Score'] = severity_counts['Severity Score'].astype(str)

    fig = px.bar(
        severity_counts, 
        x='Severity Score', 
        y='Count',
        color='Count',
        color_continuous_scale='RdYlGn_r',
        text='Count',
        labels={'Severity Score': 'Crime Severity Score (1=Low → 10=High)'}
    )
    fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
    fig.update_layout(showlegend=False, xaxis={'categoryorder': 'array', 
                    'categoryarray': [str(i) for i in range(1, 11)]})
    st.plotly_chart(fig, use_container_width=True)

# PAGE 2 — GEOGRAPHIC CLUSTERING

elif page == "🗺️ Geographic Clustering":
    st.title("🗺️ Geographic Crime Hotspots")

    algorithm = st.selectbox("Select Clustering Algorithm", ["KMeans_Cluster", "DBSCAN_Cluster"])
    sample = df.sample(50000, random_state=42)

    st.subheader(f"Crime Hotspot Map — {algorithm}")
    fig = px.scatter_mapbox(
        sample, lat='Latitude', lon='Longitude',
        color=sample[algorithm].astype(str),
        mapbox_style='carto-darkmatter',
        zoom=10, height=600,
        title="Chicago Crime Clusters"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Cluster Summary")
    summary = df.groupby(algorithm).agg(
        Total_Crimes=('Primary Type', 'count'),
        Arrest_Rate=('Arrest', 'mean'),
        Avg_Severity=('Crime_Severity', 'mean'),
        Top_Crime=('Primary Type', lambda x: x.value_counts().index[0])
    ).reset_index()
    summary['Arrest_Rate'] = (summary['Arrest_Rate'] * 100).round(1)
    summary['Avg_Severity'] = summary['Avg_Severity'].round(2)
    st.dataframe(summary, use_container_width=True)

# PAGE 3 — TEMPORAL ANALYSIS

elif page == "⏰ Temporal Analysis":
    st.title("⏰ Temporal Crime Patterns")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Crimes by Hour of Day")
        hourly = df['Hour'].value_counts().sort_index().reset_index()
        hourly.columns = ['Hour', 'Count']
        fig = px.bar(hourly, x='Hour', y='Count', color='Count',
                     color_continuous_scale='Reds')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Crimes by Day of Week")
        day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
        daily = df['Day_of_Week'].value_counts().reindex(day_order).reset_index()
        daily.columns = ['Day', 'Count']
        fig = px.bar(daily, x='Day', y='Count', color='Count',
                     color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Monthly Crime Trend")
    monthly = df.groupby(['Month']).size().reset_index(name='Count')
    fig = px.line(monthly, x='Month', y='Count', markers=True,
                  color_discrete_sequence=['orange'])
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Crime Heatmap — Hour vs Day")
    pivot = df.groupby(['Day_of_Week', 'Hour']).size().unstack(fill_value=0)
    pivot = pivot.reindex(day_order)
    fig = px.imshow(pivot, color_continuous_scale='Reds',
                    labels=dict(x='Hour', y='Day', color='Crimes'))
    st.plotly_chart(fig, use_container_width=True)


# PAGE 4 — DIMENSIONALITY REDUCTION

elif page == "📉 Dimensionality Reduction":
    st.title("📉 Dimensionality Reduction")

    sample = df.sample(20000, random_state=42)

    st.subheader("PCA — 2D Crime Patterns")
    color_by = st.selectbox("Color by", ["Crime_Severity", "KMeans_Cluster", "Arrest"])
    fig = px.scatter(sample, x='PC1', y='PC2',
                     color=sample[color_by].astype(str),
                     opacity=0.5, height=600,
                     title="PCA 2D Projection")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("PCA — 3D Crime Patterns")
    fig = px.scatter_3d(sample, x='PC1', y='PC2', z='PC3',
                        color=sample['KMeans_Cluster'].astype(str),
                        opacity=0.4, height=700,
                        title="PCA 3D Projection")
    st.plotly_chart(fig, use_container_width=True)

# PAGE 5 — MLFLOW RESULTS

elif page == "🧪 MLflow Results":
    st.title("🧪 MLflow Experiment Results")

    db_path = os.path.abspath("mlflow.db")
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")

    runs = mlflow.search_runs(experiment_names=["PatrolIQ_Clustering"])

    if runs.empty:
        st.warning("No runs found. Make sure MLflow notebook was executed.")
    else:
        st.subheader("All Experiment Runs")
        cols = ['tags.mlflow.runName', 'params.algorithm', 'params.n_clusters',
                'metrics.silhouette_score', 'metrics.davies_bouldin']
        cols = [c for c in cols if c in runs.columns]
        st.dataframe(runs[cols].sort_values('metrics.silhouette_score', ascending=False),
                     use_container_width=True)

        st.subheader("Silhouette Score Comparison")
        fig = px.bar(runs, x='tags.mlflow.runName', y='metrics.silhouette_score',
                     color='metrics.silhouette_score', color_continuous_scale='Greens',
                     title="Silhouette Score by Run")
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)