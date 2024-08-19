import streamlit as st
import pandas as pd
import pickle
import base64
from rdkit import Chem
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
st.set_page_config(page_icon="📊")

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
st.title("Try out our models!")
st.write("Enter the SMILES notation of the compound you want to predict")
st.write("Not sure what SMILES notation is? It's a string representation of any compound made so that computers can understand them, you can find some examples on [CHEMBL](%s) or down below" % "https://www.ebi.ac.uk/chembl/")
st.write("\n- **Aripiprazole**: O=C1CCc2ccc(OCCCCN3CCN(c4cccc(Cl)c4Cl)CC3)cc2N1  \n- **Chlorpromazine**: CN(C)CCCN1c2ccccc2Sc2ccc(Cl)cc21  \n- **Pramipexole**: CCCN[C@H]1CCc2nc(N)sc2C1")
st.warning("These models are more useful to drug researchers. The goal is to predict the bioactivity of compounds that aren't very well known in order to discover new drugs. Using this model with known drugs is not helpful since the compound might have been used to train the model. But you can still test it out even if you aren't an expert!")

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
