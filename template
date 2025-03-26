import streamlit as st
import torch
import dgl
import pandas as pd
import networkx as nx
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.preprocessing import StandardScaler
from model import DrugRepurposingModel  # Import AI model

# Load AI model
model = DrugRepurposingModel()
model.load_state_dict(torch.load("gnn_model.pth", map_location=torch.device('cpu')))
model.eval()

# Function to calculate chemical descriptors
def get_molecular_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    features = {
        "MW": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "HBA": Descriptors.NumHAcceptors(mol),
        "HBD": Descriptors.NumHDonors(mol)
    }
    return features

# Streamlit UI
st.set_page_config(page_title="AI Drug Repurposing", layout="wide")

st.title("🔬 AI-Powered Drug Repurposing")

st.sidebar.header("Enter Drug Information")
drug_smiles = st.sidebar.text_input("Enter Drug SMILES String", "")

if st.sidebar.button("Analyze Drug"):
    if drug_smiles:
        st.sidebar.success("Processing...")
        features = get_molecular_features(drug_smiles)

        if features:
            st.write("### Drug Chemical Properties")
            st.json(features)

            # Convert features into a format the model can process
            feature_df = pd.DataFrame([features])
            scaler = StandardScaler()
            feature_df_scaled = scaler.fit_transform(feature_df)

            # Make prediction
            input_tensor = torch.tensor(feature_df_scaled, dtype=torch.float32)
            with torch.no_grad():
                prediction = model(input_tensor)
            
            # Show results
            disease_list = ["Cancer", "Alzheimer's", "Diabetes", "Parkinson's", "Rare Disease"]
            predicted_disease = disease_list[torch.argmax(prediction).item()]
            confidence_score = torch.max(torch.nn.functional.softmax(prediction, dim=1)).item()

            st.write("### AI Prediction")
            st.success(f"🧬 Potential Treatment For: **{predicted_disease}**")
            st.info(f"Confidence Score: {confidence_score:.2f}")

        else:
            st.sidebar.error("Invalid SMILES input. Please check the structure.")
    else:
        st.sidebar.warning("Please enter a valid SMILES string.")
