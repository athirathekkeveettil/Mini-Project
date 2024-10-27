import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Custom CSS Styling
st.markdown(
    """
    <style>
    .title {
        text-align: center;
        color: #FF5733;  
        font-size: 40px;
        font-family: 'Times New Roman';
        margin: 20px 0;
    }
    .header {
        color: #4CAF50; 
    }
    .prediction {
        font-size: 24px;
        color: #FF5733;  
        font-weight: bold;
    }
    
    button {
        transition: background-color 0.3s, transform 0.3s; /* Animation for buttons */
    }
    button:hover {
        background-color: #FF5733; 
        transform: scale(1.05); 
    }
    </style>
    """, unsafe_allow_html=True
)

def model_prediction(test_image):
    # Load the pre-trained model
    model = tf.keras.models.load_model("yogamodel2.keras")
    
    # Open the image
    image = Image.open(test_image)
    
    # Resize the image to (224, 224) as expected by VGG16
    image = image.resize((224, 224))
    
    # Convert the image to an array
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    
    # Expand dimensions to make it (1, 224, 224, 3)
    input_arr = np.expand_dims(input_arr, axis=0)
    
    # Normalize the image array (VGG16 expects input values scaled between 0 and 255)
    input_arr = input_arr / 255.0
    
    # Get predictions
    predictions = model.predict(input_arr)
    
    # Return the index of the highest probability class
    return np.argmax(predictions)

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About Project", "Prediction"])

# Main Page
if app_mode == "Home":
    st.markdown("<div class='title'>YOGA POSE DETECTION AND CLASSIFICATION</div>", unsafe_allow_html=True)
    image_path = "MainPage.jpg"
    st.image(image_path)

# About Project
elif app_mode == "About Project":
    st.markdown("<div class='header'>About the Project</div>", unsafe_allow_html=True)
    st.subheader("Dataset Overview")
    st.text("This project classifies images of 10 Yoga poses classes:")
    st.code("Camel Pose or Ustrasana, Chair Pose, Lord of the Dance Pose or Natarajasana, Sitting pose 1 (normal),Split pose, Standing Forward Bend pose or Uttanasana, Supta Virasana Vajrasana, Tree Pose or Vrksasana, Upward Bow (Wheel) Pose or Urdhva Dhanurasana, Virasana or Vajrasana")
   
    st.subheader("Dataset Details")
    st.text("The dataset used is the Yoga-82 dataset.")
    st.text("1. Training Set: 2800 images in total, with 244x224x3 resolution for each yoga class.")
    st.text("2. Test Set: 400 images, balanced across all classes.")
    st.text("3. Validation Set: 800 images, balanced across all classes.")

    st.subheader("Project Goal")
    st.text("The goal is to classify the images of yoga poses using a pretrained VGG16 model.")
    
# classification Page
elif app_mode == "Prediction":
    st.markdown("<div class='title'>Model Classification</div>", unsafe_allow_html=True)
    
    test_image = st.file_uploader("Choose an Image:", type=["jpg", "jpeg", "png"])
    
    # Display the image when uploaded
    if test_image is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Show Image"):
                st.image(test_image, width=400, use_column_width=True)

        with col2:
            
            prediction_placeholder = st.empty()
           
            if st.button("Predict"):
            
                prediction_placeholder.markdown("<div style='background-color:#FF5733; padding:10px; color:white;'>Predicting...</div>", unsafe_allow_html=True)

                # Make the prediction using the model
                result_index = model_prediction(test_image)

                # Reading Labels
                with open("labels.txt") as f:
                    label = [line.strip() for line in f if line.strip()]  # Read and strip empty lines

                # Check if the prediction index is within bounds and if the label is known
                if result_index < len(label):
                    predicted_pose = label[result_index]

                    if predicted_pose != "Unknown":
                        prediction_placeholder.empty()  # Clear the "Predicting..." message
                        st.markdown(f"<div class='prediction'>The predicted pose is {predicted_pose}</div>", unsafe_allow_html=True)
                    else:
                        prediction_placeholder.empty()
                        st.markdown("<div class='prediction'>Pose not detected</div>", unsafe_allow_html=True)
                else:
                    prediction_placeholder.empty()
                    st.error("Pose not detected or prediction index out of bounds.")
