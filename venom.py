import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model

# Load the model
model = load_model(r'C:\Python\models\Diabetic.h5')
# model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

# Define class names (Ensure these match the model's class names)
class_names = ['0','1','2','3','4']
def predict(model, image, class_names):
    # Ensure the image is resized to 256x256 and in RGB format
    img = image.resize((256, 256)).convert('RGB')
    img_array = np.array(img)  # Convert the image to a numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize the image

    # Perform prediction
    predictions = model.predict(img_array)

    # Debugging: Print the shape and content of predictions
    print(f'Predictions shape: {predictions.shape}')
    print(f'Predictions: {predictions}')

    # Ensure that predictions are in the expected shape
    if predictions.ndim == 2 and predictions.shape[1] == len(class_names):
        predicted_class = class_names[np.argmax(predictions[0])]  # Get the class with the highest probability
        confidence = round(np.max(predictions[0]) * 100, 2)
    else:
        # Handle the case where predictions is not in the expected format
        predicted_class = "Error: Invalid Prediction Shape"
        confidence = 0

    return predicted_class, confidence

# def prediction(image_path, class_names):

#     img = Image.open(image_path).resize((256,256))
#     img_array = tf.keras.preprocessing.image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0) 

#     model = tf.keras.models.load_model('model1.h5')
#     prediction = model.predict(img_array)

#     predicted_class = class_names[np.argmax(prediction)]
#     confidence = round(np.max(prediction)*100, 2)

#     print(f'Predicted Class : {predicted_class}')
#     print(f'Confident : {confidence}%')

# prediction(image_path=r"C:\Users\nitis\OneDrive\Desktop\training\Potato___Early_blight\0e0a1b51-f61c-4934-bc57-a820af1faacb___RS_Early.B 7147.JPG")


#Streamlit Interface
st.title("Diabetic Retinopathy Prediction")
st.write("Upload an image of a retina to predict the level of disease.")

# Image upload
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Display the uploaded image
    img = Image.open(uploaded_image)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Predict on button click
    if st.button('Predict'):
        predicted_class, confidence = predict(model, img, class_names)
        st.write(f'Predicted Class: {predicted_class}')
        st.write(f'Confidence: {confidence}%')
