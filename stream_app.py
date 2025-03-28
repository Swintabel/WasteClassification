import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
import cv2
import numpy as np

# Define the labels
labels = {0: 'Cardboard',
          1: 'Food Organics',
          2: 'Glass',
          3: 'Metal',
          4: 'Miscellaneous Trash',
          5: 'Paper',
          6: 'Plastic',
          7: 'Textile Trash',
          8: 'Vegetation'}

# Load the models
cnn = load_model("cnnModel.keras")
#hyb = load_model("HYBModel.keras")

# Streamlit app
st.title("Trash Classification with CNN Model (EfficientNetV2B0)")

# Add a description about the application
st.write("""
Welcome to the Trash Classification App! This application utilizes a Convolutional Neural Network (CNN) model based on the EfficientNetV2B0 architecture to classify different types of trash. 
Upload an image of trash, and the model will predict the type of material, helping you to sort your waste more efficiently. Nine classes of waste can
be tested with the app.
""")

# File uploader
uploaded_file = st.sidebar.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Save the uploaded file temporarily
    with open("img.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load and preprocess the image
    image = cv2.imread("img.jpg")
    image = cv2.resize(image, (224, 224))
    image_array = np.array(image)
    image_array = preprocess_input(image_array)
    img_array = np.expand_dims(image_array, axis=0)

    # Predictions
    pred = cnn.predict(img_array)
    #pred2 = hyb.predict(img_array)

    predicted_class = np.argmax(pred, axis=1)
    #predicted_class2 = np.argmax(pred2, axis=1)

    pred_label = labels[predicted_class[0]]
    #pred_label2 = labels[predicted_class2[0]]

    # Results
    results = pred_label
        

    # Display results
    st.image("img.jpg", caption="Uploaded Image", use_container_width=True)
    st.markdown(f"## Prediction is {results}")




# Add an image 
st.image("classes.jpg", caption="Welcome to the Trash Classification App", use_container_width=True)


