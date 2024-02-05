import streamlit as st
import numpy as np
import os
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2

# Load the pre-trained model
loaded_model = load_model('mobileNet.h5')  # Provide the path to your saved model

# Function to preprocess an image for prediction
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0  # Normalize pixel values to the range [0, 1]

# Function to preprocess a video frame for prediction
def preprocess_frame(frame):
    frame = cv2.resize(frame, (224, 224))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_array = np.expand_dims(frame, axis=0)
    return frame_array / 255.0  # Normalize pixel values to the range [0, 1]

# Function to make predictions on an image
def predict_image(model, img_path):
    preprocessed_img = preprocess_image(img_path)
    prediction = model.predict(preprocessed_img)
    return prediction[0][0]  # Assuming binary classification, adjust if needed

# Function to make predictions on a video frame
def predict_video_frame(model, frame):
    preprocessed_frame = preprocess_frame(frame)
    prediction = model.predict(preprocessed_frame)
    return prediction[0][0]  # Assuming binary classification, adjust if needed

# Streamlit app
st.title("Pothole Detection App 🏴‍☠️🚧")

# Create a navigation bar
page = st.sidebar.selectbox("Select a Page", ["Image Upload", "Video Upload"])
st.sidebar.image('Welcome to POT-HOLE DETECTION.png')
if page == "Image Upload":
    # Image upload section
    st.header("Image Upload")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg"])

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, use_column_width=True)

        # Make prediction when the user clicks the 'Predict' button
        if st.button("Predict"):
            try:
                img_path = "temp_image.jpg"
                with open(img_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Make prediction
                prediction = predict_image(loaded_model, img_path)

                # Display the prediction result
                result = "Pothole" if prediction >= 0.5 else "Normal"
                st.success(f"Prediction: {result}")

            except Exception as e:
                st.error(f"Error: {e}")
            finally:
                # Remove the temporary image file
                if os.path.exists(img_path):
                    os.remove(img_path)

elif page == "Video Upload":
    # Video upload section
    st.header("Video Upload")
    uploaded_file = st.file_uploader("Choose a video...", type=["mp4"])

    if uploaded_file is not None:
        # Save the uploaded video locally
        video_path = "temp_video.mp4"
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Display the uploaded video
        st.video(video_path)

        # Make prediction when the user clicks the 'Predict' button
        if st.button("Predict"):
            try:
                # OpenCV VideoCapture using the file path
                video_cap = cv2.VideoCapture(video_path)

                # Get the frames per second (fps) of the input video
                fps = video_cap.get(cv2.CAP_PROP_FPS)

                # Read the first frame
                success, frame = video_cap.read()

                # Make prediction on the first frame
                prediction = predict_video_frame(loaded_model, frame)

                # Display the prediction result
                result = "Pothole 🚧🚩" if prediction >= 0.5 else "Normal 🏳🏁"
                st.success(f"Prediction: {result}")

            except Exception as e:
                st.error(f"Error: {e}")
            finally:
                # Release the video capture object
                video_cap.release()
                # Remove the temporary video file
                if os.path.exists(video_path):
                    os.remove(video_path)
