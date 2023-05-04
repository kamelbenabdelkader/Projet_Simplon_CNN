import streamlit as st
import numpy as np
from PIL import Image

class DigitRecognitionPicture:
    def __init__(self, model, canvas_size=192, input_shape=(28, 28, 1)):
        self.model = model
        self.canvas_size = canvas_size
        self.input_shape = input_shape
        self.img = None

    def preprocessing(self, image):
        # Convert the image to grayscale
        image = image.convert('L')
        # Resize the image to the required input shape of the model
        image = image.resize(self.input_shape[:2])
        # Invert the pixel values
        image = np.invert(image)
        # Reshape the image to a 4D array with a batch size of 1
        image = np.reshape(image, (1,) + self.input_shape)
        # Normalize the pixel values
        image = image / 255.0
        return image

    def uploadRun(self) :

        # Create a file uploader widget
        uploaded_file = st.file_uploader("Veuillez upload une image à analyser : ", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            with st.spinner(text='loading'):
                    st.info('Image au bon format et prête à être analyser')
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
                val = self.model.predict(self.preprocessing(resized_image_up))
                val = np.around(val, 3)
                st.write(f'La valeur predite est: {np.argmax(val)}')
                st.write("Probabilité de la Prediction")
                st.write(np.around(val, 4))
                st.bar_chart(val.reshape(10, 1))
