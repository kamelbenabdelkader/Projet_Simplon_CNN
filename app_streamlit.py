import streamlit as st
import tensorflow as tf
import numpy as np
import keras
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2
import pickle

# Set le titre de la page
st.set_page_config(page_title='My Streamlit App')


#Importer du CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("style.css")



# Afficher le titre
st.title('PREDICTION CNN STREAMLIT APP!')
# Charger le modèle entraîné
with open('model.pickle', 'rb') as f:
        model = pickle.load(f)

# model = tf.keras.models.load_model("model_test_model.h5")

# Page 1
def page1():
    st.title("Page 1")
    # Function to preprocess the image
    def preprocess_image(image, input_shape):
    # Convert the image to grayscale
        image = image.convert('L')
    # Resize the image to the required input shape of the model
        image = image.resize(input_shape[:2])
    # Invert the pixel values
        image = np.invert(image)
    # Reshape the image to a 4D array with a batch size of 1
        image = np.reshape(image, (1,) + input_shape)
    # Normalize the pixel values
        image = image / 255.0
        return image
    # Define the input shape of the model
    input_shape = (28, 28, 1)

    # Create a file uploader widget
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
    # Display the uploaded image
        image_up = Image.open(uploaded_file)
         # Resize the image to a width of 600 pixels and proportional height
        new_width = 50
        width, height = image_up.size
        new_height = int(height * new_width / width)
        resized_image_up = image_up.resize((new_width, new_height))
        st.image(resized_image_up, caption='Image Télécharger', use_column_width=False)
        if st.button("Prédiction"):
    # Preprocess the image and make a prediction using the model
            prediction = model.predict(preprocess_image(resized_image_up, input_shape))
            predicted_class = np.argmax(prediction)
            st.title(predicted_class)


# Page 2
def page2():

    # model1 = tf.keras.models.load_model("CNN.h5")

    st.title("Page 2")


    # define size of the canvas
    SIZE = 192
    canvas_result = st_canvas(
        fill_color='#000000',
        stroke_width=10,
        stroke_color='#FFFFFF',
        background_color='#000000',
        width=SIZE,
        height=SIZE,
        key='canvas')

    if canvas_result.image_data is not None:
            img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
            rescaled = cv2.resize(
                img, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
            st.write("Input de l'Image pour le model")
            st.image(rescaled)

    if st.button('Prediction du chiffre '):
        test_x = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        test_x = test_x.reshape(1, 28, 28, 1) / 255
        val = model.predict(test_x)
        val = np.around(val, 3)
        st.write(f'La valeur predite est de: {np.argmax(val)}')
        st.write("Probabilité de la Prediction")
        st.write(np.around(val, 4))
        st.bar_chart(val.reshape(10, 1))


st.sidebar.title("Navigation")
page = st.sidebar.radio("Aller à :", ["Page 1", "Page 2"])
# bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
# bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])
realtime_update = st.sidebar.checkbox("Update in realtime", True)



# Afficher la page sélectionnée
if page == "Page 1":
    page1()
elif page == "Page 2":
    page2()
