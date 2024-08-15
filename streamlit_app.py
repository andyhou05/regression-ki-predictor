import streamlit as st
from streamlit.logger import get_logger

import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import plotly.express as px
import pickle
from pandas import DataFrame

import matplotlib.pyplot as plt

from scipy import stats
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from rdkit import Chem
import mordred
from mordred import Calculator, descriptors

from scipy.special import inv_boxcox
from scipy.stats import boxcox_normplot

def desc_calc(smiles: str):
    """Calculates molecular descriptors of a compound using Mordred.

    Args:
        smiles (str): SMILES notation of the compound

    Returns:
        DataFrame: A DataFrame object containing the molecular descriptors
    """
    
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    calc = Calculator(descriptors, ignore_3D=True)
    desc = calc(mol)
    return pd.DataFrame([desc.asdict()])

def model_predict(smiles: str):
    """Predicts the pKi values against D2, D3, 5-HT1A, and 5-HT2A receptors for a given compound.

    Args:
        smiles (str): SMILES notation of the compound
    """
    
    # Calculate the descriptors that will be used as features
    input_data = desc_calc(smiles)
    
    # Load the models
    models={
        "D2 Dopamine Receptor":pickle.load(open('./models/d2_ki.pkl', 'rb')),
        "D3 Dopamine Receptor":pickle.load(open('./models/d3_ki.pkl', 'rb')),
        "5-HT1A Serotonin Receptor":pickle.load(open('./models/5ht1a_ki.pkl', 'rb')),
        "5-HT2A Serotonin Receptor":pickle.load(open('./models/5ht2a_ki.pkl', 'rb'))
        }

    predictions = {}
    for protein, model in models.items():
        # Select the particular descriptors that are used by the different models (they don't use the same descriptors)
        x_list = list(model.feature_names_in_)
        subset = input_data[x_list]
        predictions[protein] = model.predict(subset)[0]
        
    # Output the values
    st.header('**Prediction Output (pKi Values)**')
    st.write(pd.DataFrame(data=[predictions], index=[smiles]))
    
def display_dataset(dataset: DataFrame):
    """Displays the 5 first elements and the shape of a given dataset.

    Args:
        dataset (DataFrame): The dataset we want to display
    """
    
    st.write(dataset.head())
    st.write(dataset.shape)
    
    
# UI of the Steamlit app
st.set_page_config(
        page_title="Medical Insurance Charges Regression",
        page_icon="📊",
    )

st.title('Regression')
st.header('Raw Data')

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

st.write("Enter the SMILES notation of the compound you want to predict")
st.write("Not sure what SMILES notation is? It's a string representation of any compound made so that computers can understand them, you can find some examples of [CHEMBL](%s)" % "https://www.ebi.ac.uk/chembl/")
smiles = st.text_input(label="SMILES", label_visibility="collapsed", placeholder="SMILES")

if smiles:
    model_predict(smiles)
