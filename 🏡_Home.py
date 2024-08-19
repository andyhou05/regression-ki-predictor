import streamlit as st

# UI of the Streamlit app
# Set the page configuration
st.set_page_config(page_title="QSAR App", page_icon="📊")

st.title("Ki Receptor Predictor 🧬")
st.write(
    """
    This web application gives you the ability to predict inhibition constant (pKi) of any compound in the world, even if that compound has never been synthesized!  \n  \n
    We offer you 4 QSAR models to predict the values against 4 different target proteins (D2 and D3 dopamine receptors, 5-HT1A and 5-HT2A serotonin receptors) so you can see the different binding values.
    All you need to do is enter the SMILES notation of the compound you want to predict! The values then get sent to GPT-4o to get analyzed, to get an idea of what the compound can do!  \n  \n
    You can also enter a .txt file containing multiple compounds if you want to predict multiple compounds.
    """
)

# Home page with explanations
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

# Home page with explanations
st.write("""
### What are SMILES 😀?
SMILES (Simplified Molecular Input Line Entry System) is a notation that allows a user to represent a chemical structure in a way that can be used by the computer.

### What is QSAR 📊?
Quantitative Structure-Activity Relationship (QSAR) models are used to predict the effects, properties, or activity of a compound based on its chemical structure.

### Why is this important 🔓?
Predicting the activity of compounds helps in drug discovery, reducing the cost and time of experiments. For any single known drug, there can be thousands of possible derivatives of that compound, synthesizing all of them takes a lot of time and ressources.
""")

