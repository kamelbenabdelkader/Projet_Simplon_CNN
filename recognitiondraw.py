import streamlit as st
import cv2
import numpy as np
from streamlit_drawable_canvas import st_canvas

class DigitRecognitionDraw:
    def __init__(self, model, canvas_size=192, input_shape=(28, 28, 1)):
        self.model = model
        self.canvas_size = canvas_size
        self.input_shape = input_shape
        self.canvas_result = st_canvas(
            fill_color='#000000',
            stroke_width=10,
            stroke_color='#FFFFFF',
            background_color='#000000',
            width=self.canvas_size,
            height=self.canvas_size,
            key='canvas')
        self.img = None
        self.classId =  None
        self.confidence = None

    def imageScale(self):
        if self.canvas_result.image_data is not None:
            self.img = cv2.resize(self.canvas_result.image_data.astype('uint8'), (28, 28))
            rescaled = cv2.resize(
                self.img, (self.canvas_size, self.canvas_size), interpolation=cv2.INTER_NEAREST)
            st.write("Input de l'Image pour le model")
            st.image(rescaled)

    def predictDraw(self):
        if st.button('Prediction du chiffre '):
            if self.img is not None:
                test_x = cv2.cvtColor(self.img, cv2.COLOR_BGRA2GRAY)
                test_x = test_x.reshape(1, 28, 28, 1) / 255
                val = self.model.predict(test_x)
                val = np.around(val, 3)
                st.write(f'La valeur predite est de: {np.argmax(val)}')
                st.write("Probabilité de la Prediction")
                st.write(np.around(val, 4))
                st.bar_chart(val.reshape(10, 1))
            else:
                st.warning("Veuillez dessiner un chiffre avant de lancer la prédiction.")
