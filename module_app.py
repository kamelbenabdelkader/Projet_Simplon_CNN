import streamlit as st
import pickle

# Fonction qui charge le fichier css
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Function qui charge le modèle entraîné
def load_model():
    with open('model.pickle', 'rb') as f:
        model = pickle.load(f)
    return model
