import streamlit as st
import pandas as pd
import pickle
from pandas import DataFrame
import base64
from rdkit import Chem
from rdkit.Chem import AllChem
from mordred import Calculator, descriptors
from openai import OpenAI
import os


def desc_calc(smiles: str):
    """Calculates molecular descriptors of a compound using Mordred."""
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    calc = Calculator(descriptors, ignore_3D=True)
    desc = calc(mol)
    return pd.DataFrame([desc.asdict()])

def model_predict(smiles: str):
    """Predicts the pKi values against D2, D3, 5-HT1A, and 5-HT2A receptors for a given compound."""
    input_data = desc_calc(smiles)
    models = {
        "D2 Dopamine Receptor": pickle.load(open('./models/d2_ki.pkl', 'rb')),
        "D3 Dopamine Receptor": pickle.load(open('./models/d3_ki.pkl', 'rb')),
        "5-HT1A Serotonin Receptor": pickle.load(open('./models/5ht1a_ki.pkl', 'rb')),
        "5-HT2A Serotonin Receptor": pickle.load(open('./models/5ht2a_ki.pkl', 'rb'))
    }
    predictions = {}
    for protein, model in models.items():
        x_list = list(model.feature_names_in_)
        subset = input_data[x_list]
        predictions[protein] = model.predict(subset)[0]
    return pd.DataFrame(data=[predictions], index=[smiles])

def download_link(object_to_download, download_filename, download_link_text):
    """Generates a link to download the data."""
    b64 = base64.b64encode(object_to_download.encode()).decode()  # some strings <-> bytes conversions necessary here
    return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'

def display_dataset(dataset: DataFrame):
    """Displays the 5 first elements and the shape of a given dataset."""
    st.write(dataset.head())
    st.write(dataset.shape)


# UI of the Streamlit app
# Set the page configuration
st.set_page_config(page_title="QSAR pKi Prediction App", page_icon="📊")

# Home page with explanations
st.sidebar.title("About")
st.sidebar.markdown("""
### What are SMILES?
SMILES (Simplified Molecular Input Line Entry System) is a notation that allows a user to represent a chemical structure in a way that can be used by the computer.

### What is QSAR?
Quantitative Structure-Activity Relationship (QSAR) models are used to predict the effects, properties, or activity of a compound based on its chemical structure.

### Why is this important?
Predicting the activity of compounds helps in drug discovery, reducing the cost and time of experiments.
""")

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

# Add drag-and-drop option for SMILES files
uploaded_file = st.file_uploader("Upload a .txt file containing the SMILES you want to predict", type=["txt"])
if uploaded_file is not None:
    try:
        smiles_list = uploaded_file.read().decode("utf-8").splitlines()
        st.write("SMILES loaded:")
        st.write(smiles_list)
        all_predictions = pd.DataFrame()
        for smiles in smiles_list:
            predictions = model_predict(smiles)
            all_predictions = pd.concat([all_predictions, predictions])
        st.header('**Prediction Output (pKi Values)**')
        st.write(all_predictions)
    except:
        st.warning("Your file is not properly formatted, make sure the SMILES notations are correct and that the values are separated with a new line as so:  \n  \nSMILES 1  \nSMILES 2  \nSMILES 3  \nSMILES 4", icon='⚠️')

    # Option to download predictions
    tmp_download_link = download_link(all_predictions.to_csv(index=False), 'predictions.csv', 'Click here to download your predictions!')
    st.markdown(tmp_download_link, unsafe_allow_html=True)
    
else:
    # Single SMILES input option
    smiles = st.text_input(label="SMILES", label_visibility="collapsed", placeholder="SMILES")
    if smiles:
        try:
            predictions = model_predict(smiles)
            st.header('**Prediction Output (pKi Values)**')
            st.write(predictions)
            # Send the pKi values to LLM for response
            with st.spinner("Analyzing data..."):
                client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
                completion = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a drug researcher, the following values are pKi values of a compound against 4 target proteins: D2 and D3 dopamine receptors, 5-HT1A and 5-HT2A receptors, which were predicted from a QSAR model. With your analysis, please explain the meaning behind these values, each in bullet point. Explain whether or not this compound could be used to treat certain disease. Only give an anlysis of every pKi value, do not say anything else. If the value indicates that the compound is not suitable for a certain disease, please mention so, but do not make up anything whatsoever."},
                        {
                            "role": "user",
                            "content": predictions.to_string(index=False)
                        }
                    ]
                )
                result = completion.choices[0].message.content
                if result:
                    st.write(result)
                else:
                    st.write("Something went wrong with the LLM...")
            
            # Option to download the prediction
            tmp_download_link = download_link(predictions.to_csv(index=False), 'prediction.csv', 'Click here to download your prediction!')
            st.markdown(tmp_download_link, unsafe_allow_html=True)
        except Exception as e:
            st.write(e)
            st.write(":red[There seems to be an issue with your SMILES, please double check it.]")
