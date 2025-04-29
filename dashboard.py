
import pandas as pd
import streamlit as st
import plotly.express as px

def format_field_name(field):
  return field.replace("_", " ").title()

@st.cache_data
def load_data():
  # Load your data
  df = pd.read_csv("jobs_final.csv")
  df["Predicted domain"] = df["Predicted Class"].apply(lambda x: format_field_name(x.split("/")[0]))
  df["Predicted field"] = df["Predicted Class"].apply(lambda x: format_field_name(x.split("/")[1]))
  df["Predicted title"] = df["Predicted Class"].apply(lambda x: format_field_name(x.split("/")[2]))
  return df

def compute_comparison(df, company):
    metrics = ['salary_estimate','years_experience']
    sel = df[df['Company']==company][metrics].mean()
    avg = df.groupby('Company')[metrics].mean().mean()
    return pd.DataFrame([sel, avg], index=[company,'Big4 Average'])

def main():
    st.title("Big 4 Benchmark")
    df = load_data()
    firm = st.sidebar.selectbox("Select firm", df['Company'].unique())
    comp = compute_comparison(df, firm)
    st.dataframe(comp)
    fig = px.bar(comp.reset_index().melt(id_vars='index'),
                 x='variable', y='value', color='index',
                 barmode='group')
    st.plotly_chart(fig, use_container_width=True)

if __name__=='__main__':
    main()
