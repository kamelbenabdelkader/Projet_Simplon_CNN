import streamlit as st
import numpy as np
from PIL import Image
import streamlit as st
import pickle
import module_app
from recognitiondraw import DigitRecognitionDraw
from recognitionpicture import DigitRecognitionPicture

# Set le titre de la page
st.set_page_config(page_title='My Streamlit App')


# Sidebar
st.sidebar.title("Navigation")

option = st.sidebar.selectbox(" ", ["Prediction Image", "Prediction Draw"])
# realtime_update = st.sidebar.checkbox("Update in realtime", True)

#Import du CSS
module_app.local_css("style.css")

# Charger le modèle entraîné
model = module_app.load_model()

# Page 1
def page1():
    st.title("Digital Prediction Image")
    modelPictureClass = DigitRecognitionPicture(model)
    modelPictureClass.uploadRun()

# Page 2
def page2():
    st.title("Digital Prediction Draw")
    modelClass = DigitRecognitionDraw(model)
    modelClass.imageScale()
    modelClass.predictDraw()

# Afficher la page sélectionnée
if option ==   "Prediction Image":
    page1()
elif option == "Prediction Draw":
    page2()
