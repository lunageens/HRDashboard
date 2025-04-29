import pandas as pd
import streamlit as st
import plotly.express as px

@st.cache_data
def load_data(path: str = "jobs_final.csv") -> pd.DataFrame:
    """
    Load and preprocess the jobs dataset.

    Parameters
    ----------
    path : str
        Path to the CSV file containing job postings.

    Returns
    -------
    pd.DataFrame
        DataFrame with:
        - Predicted domain / field / title (readable labels)
    """
    df = pd.read_csv(path)
    # Human-readable splits of the Predicted Class hierarchy
    df["Predicted domain"] = (
        df["Predicted Class"].str.split("/").str[0]
        .str.replace("_", " ").str.title()
    )
    df["Predicted field"] = (
        df["Predicted Class"].str.split("/").str[1]
        .str.replace("_", " ").str.title()
    )
    df["Predicted title"] = (
        df["Predicted Class"].str.split("/").str[2]
        .str.replace("_", " ").str.title()
    )
    return df

def domain_share(df_: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Compute normalized share of predicted domains for a given company slice.
    """
    s = df_["Predicted domain"].value_counts(normalize=True)
    return pd.DataFrame({
        "Predicted domain": s.index,
        "Share": s.values,
        "Company": name
    })


def top_locations(df_: pd.DataFrame, name: str, n: int = 5) -> pd.DataFrame:
    """
    Return top N locations with counts for a given company slice.
    """
    s = df_["Location"].value_counts().nlargest(n)
    return pd.DataFrame({
        "Location": s.index,
        "Count": s.values,
        "Company": name
    })

def main():
    """
    Render the Streamlit dashboard:
    - Main filters (company + domain)
    - KPI cards with firm vs. Big4 averages
    - Distribution and location charts
    """
    st.set_page_config(page_title="Big 4 Jobs Dashboard", layout="wide")
    st.title("Big 4 Jobs Benchmarking Dashboard")

    # --- 1 LOAD & FILTER ---
    df = load_data()
    col1, _ = st.columns([1, 4])
    with col1:
        firm = st.selectbox(
            "Select firm", options=sorted(df["Company"].unique())
        )
        domains = st.multiselect(
            "Predicted Domain",
            options=sorted(df["Predicted domain"].unique()),
            default=sorted(df["Predicted domain"].unique()),
        )
    # Filter by selected domains
    df_filtered = df[df["Predicted domain"].isin(domains)]
    sel = df_filtered[df_filtered["Company"] == firm]

    # --- 2. KPI CARDS ---
    # Calculate metrics for selection
    total_posts = len(sel)
    unique_locs = sel["Location"].nunique()
    top_func = sel["Predicted title"].value_counts().idxmax() if total_posts else "N/A"
    # Big4 averages for same domains
    avg_posts = df_filtered.groupby("Company").size().mean()
    avg_unique_locs = df_filtered.groupby("Company")["Location"].nunique().mean()
    overall_top_func = df_filtered["Predicted title"].value_counts().idxmax()
    # Place them into dashboard
    st.subheader("Key Metrics")
    cols = st.columns(3)
    metrics = [
        ("Total Openings", total_posts, avg_posts),
        ("Unique Locations", unique_locs, avg_unique_locs),
        ("Top Job Title", top_func, overall_top_func)
    ]
    for col, (label, firm_val, avg_val) in zip(cols, metrics):
        avg_disp = f"({avg_val:.0f})" if isinstance(avg_val, (int, float)) else f"({avg_val})"
        col.markdown(
            f"""**{label}**<br>
<span style=\"font-size:24px;color:black;\">{firm_val}</span> 
<span style=\"font-size:20px;color:grey;\">{avg_disp}</span>""",
            unsafe_allow_html=True
        )

    # --- 3. CHART 1: Job Function Distribution ---
     # Average share across all companies (within selected domains)
    avg_series = (
        df_filtered.groupby("Company")["Predicted title"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
        .mean()
    )
    # Build avg_dist for functions present in sel only
    sel_funcs = sel["Predicted title"].unique()
    avg_dist = (
        avg_series.reindex(sel_funcs, fill_value=0)
        .reset_index(name="Share")
        .rename(columns={0: "Predicted title"})
    )
    avg_dist["Company"] = "Big 4 Average"

    # Selected firm distribution
    sel_dist = (
        sel["Predicted title"].value_counts(normalize=True)
        .reset_index(name="Share")
        .rename(columns={"index": "Predicted title"})
    )
    sel_dist["Company"] = firm

    df_dist = pd.concat([sel_dist, avg_dist], ignore_index=True)
    fig1 = px.bar(
        df_dist,
        x="Predicted title",
        y="Share",
        color="Company",
        barmode="group",
        title="Job Title Share: Selected vs. Big 4 Average",
    )
    st.plotly_chart(fig1, use_container_width=True)

    
    # --- 3. CHART 2: Job Field Distribution ---
     # Average share across all companies (within selected domains)
    avg_series = (
        df_filtered.groupby("Company")["Predicted field"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
        .mean()
    )
    # Build avg_dist for functions present in sel only
    sel_funcs = sel["Predicted field"].unique()
    avg_dist = (
        avg_series.reindex(sel_funcs, fill_value=0)
        .reset_index(name="Share")
        .rename(columns={0: "Predicted field"})
    )
    avg_dist["Company"] = "Big 4 Average"

    # Selected firm distribution
    sel_dist = (
        sel["Predicted field"].value_counts(normalize=True)
        .reset_index(name="Share")
        .rename(columns={"index": "Job Field"})
    )
    sel_dist["Company"] = firm

    df_dist = pd.concat([sel_dist, avg_dist], ignore_index=True)
    fig1 = px.bar(
        df_dist,
        x="Predicted field",
        y="Share",
        color="Company",
        barmode="group",
        title="Job Field Share: Selected vs. Big 4 Average",
    )
    st.plotly_chart(fig1, use_container_width=True)


    # --- CHART 2: Skill Domain (100% Stacked) ---
    fig2 = px.bar(
        pd.concat([domain_share(sel, firm), domain_share(df_filtered, "Big 4 Average")], ignore_index=True),
        x="Company",
        y="Share",
        color="Predicted domain",
        title="Skill Domain Distribution",
        barmode="stack"
    )
    st.plotly_chart(fig2, use_container_width=True)

    # --- CHART 3: Top Locations ---
    fig3 = px.bar(
        pd.concat([top_locations(sel, firm), top_locations(df_filtered, "Big 4 Average")], ignore_index=True),
        x="Location",
        y="Count",
        color="Company",
        barmode="group",
        title="Top Locations Comparison"
    )
    st.plotly_chart(fig3, use_container_width=True)

if __name__ == "__main__":
    main()
