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
    - KPI cards
    - Distribution and location charts
    """
    st.set_page_config(page_title="Big 4 Jobs Dashboard", layout="wide")
    st.title("📊 Big 4 Jobs Benchmarking Dashboard")

    # --- LOAD & FILTER ---
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
    sel = df[(df["Company"] == firm) & (df["Predicted domain"].isin(domains))]

    # --- KPI CARDS ---
    total_posts = len(sel)
    unique_locs = sel["Location"].nunique()
    top_func = (
        sel["Job Function"].value_counts().idxmax() if total_posts else "N/A"
    )
    k1, k2, k3 = st.columns(3)
    k1.metric("Total Openings", total_posts)
    k2.metric("Unique Locations", unique_locs)
    k3.metric("Top Job Function", top_func)

    # --- AVERAGE DISTRIBUTION PREP ---
    avg_dist = (
        df.groupby("Company")["Job Function"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
        .mean()
        .reset_index(name="Share")
        .melt(id_vars="index", var_name="Job Function", value_name="Share")
        .rename(columns={"index": "Company"})
    )

    # --- CHART 1: Job Function Distribution ---
    sel_dist = (
        sel["Job Function"].value_counts(normalize=True)
        .reset_index(name="Share")
        .assign(Company=firm)
        .rename(columns={"index": "Job Function"})
    )
    df_dist = pd.concat([sel_dist, avg_dist], ignore_index=True)
    fig1 = px.bar(
        df_dist,
        x="Job Function",
        y="Share",
        color="Company",
        barmode="group",
        title="Job Function Share: Selected vs. Big 4 Average",
    )
    st.plotly_chart(fig1, use_container_width=True)

    # --- CHART 2: Skill Domain (100% Stacked) ---
    fig2 = px.bar(
        pd.concat([domain_share(sel, firm), domain_share(df, "Big 4 Average")], ignore_index=True),
        x="Company",
        y="Share",
        color="Predicted domain",
        title="Skill Domain Distribution",
        barmode="stack"
    )
    st.plotly_chart(fig2, use_container_width=True)

    # --- CHART 3: Top Locations ---
    fig3 = px.bar(
        pd.concat([top_locations(sel, firm), top_locations(df, "Big 4 Average")], ignore_index=True),
        x="Location",
        y="Count",
        color="Company",
        barmode="group",
        title="Top Locations Comparison"
    )
    st.plotly_chart(fig3, use_container_width=True)

if __name__ == "__main__":
    main()
