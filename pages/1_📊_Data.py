import streamlit as st
from pandas import DataFrame
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def display_dataset(dataset: DataFrame):
    """Displays the 5 first elements and the shape of a given dataset."""
    st.write(dataset.head())
    st.write(dataset.shape)
    
st.set_page_config(page_icon="📊")
st.header('Raw Data')
st.write("""
         This page contains the data as well as some plots to help you understand what was used in order to train the data. The model was trained using a Random Forest Algorithm
         """)
st.sidebar.title("Contributors 👨‍💻")
st.sidebar.write(
    """
    - **Andy Hou**
    - **Ahmet Yilldrim**
    - **Yaman Alhamamy**
    - **Karina Dobkin**
    """
)
st.sidebar.write("[Source Code](%s)" % "https://github.com/andyhou05/regression-ki-predictor")
# The URL of the CSV file to be read into a DataFrame
csv_urls = ["./data/5ht1a_mordred_ki_fingerprints.csv",
            "./data/5ht2a_mordred_ki_fingerprints.csv",
            "./data/d2_mordred_ki_fingerprints.csv",
            "./data/d3_mordred_ki_fingerprints.csv"]

# Reading the CSV data from the specified URL into a list of DataFrames
dfs = []
for i in range(len(csv_urls)):
    dfs.append(pd.read_csv(csv_urls[i]))

# Display the datasets
st.subheader("5-HT1A Receptor")
display_dataset(dfs[0])

st.subheader("5-HT2A Receptor")
display_dataset(dfs[1])

st.subheader("D2 Receptor")
display_dataset(dfs[2])

st.subheader("D3 Receptor")
display_dataset(dfs[3])

df = pd.read_csv("./data/d2_ki_lipinski_descriptors_no_intermediate.csv")
sns.set_theme(style='ticks')
plt.figure(figsize=(5.5, 5.5))
sns.scatterplot(x='MW', y='LogP', data=df, hue='bioactivity_class', size='pchembl_value', edgecolor='black', alpha=0.7)

st.write("We can see that the data is grouped together and isn't very spread out, this means that this data spans a specific chemical space. These plots are shown with data for the D2 dopamine receptor.")
plt.xlabel('MW', fontsize=14, fontweight='bold')
plt.ylabel('LogP', fontsize=14, fontweight='bold')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
st.pyplot(plt)

fig, ax = plt.subplots(figsize=(10, 8))
df[['pchembl_value', 'MW', 'LogP', 'NumHDonors', 'NumHAcceptors']].hist(bins=30, ax=ax)
st.pyplot(fig)