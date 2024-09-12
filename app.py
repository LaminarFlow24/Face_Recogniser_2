import io
import joblib
import pandas as pd
from PIL import Image
import streamlit as st
import boto3
from face_recognition import preprocessing
# from dotenv import load_dotenv
import os

# # Load .env file
# load_dotenv()

# Access AWS credentials
aws_access_key_id = st.secrets["AWS_ACCESS_KEY_ID"]
aws_secret_access_key = st.secrets["AWS_SECRET_ACCESS_KEY"]
aws_region = st.secrets["AWS_REGION"]

# Initialize the S3 client using credentials stored in your .env file
s3 = boto3.client('s3', 
                  aws_access_key_id=aws_access_key_id,
                  aws_secret_access_key=aws_secret_access_key,
                  region_name=aws_region)

bucket_name = "yashasbucket247"
attendance_file_key = "attendance-data/E2_attendance_september.xlsx"  # S3 key for attendance file
model_file_key = "trained_models/SE2.pkl"  # S3 key for the face recognition model

# Function to download attendance file from S3
@st.cache_data
def load_attendance_data_from_s3():
    obj = s3.get_object(Bucket=bucket_name, Key=attendance_file_key)
    return pd.read_excel(io.BytesIO(obj['Body'].read()))

# Function to download and load face recognizer model from S3
@st.cache_resource
def load_face_recogniser_model_from_s3():
    model_obj = s3.get_object(Bucket=bucket_name, Key=model_file_key)
    return joblib.load(io.BytesIO(model_obj['Body'].read()))

divs = ['SE1', 'SE2', 'SE3', 'SE4']
div = st.selectbox("Choose a division", divs)

if div == 'SE2':
    # Load the face recognizer model from S3
    face_recogniser = load_face_recogniser_model_from_s3()
else:
    st.write("Model not trained for this division")
    st.stop()

# Initialize session state for attendance DataFrame
if 'df_attendance' not in st.session_state:
    st.session_state.df_attendance = load_attendance_data_from_s3()

# Initialize session state for tracking date changes
if 'previous_date' not in st.session_state:
    st.session_state.previous_date = None

# Initialize session state for resetting the uploader key
if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = 0

# Initialize session state for tracking reupload button click
if 'reupload_clicked' not in st.session_state:
    st.session_state.reupload_clicked = False

preprocess = preprocessing.ExifOrientationNormalize()

# Streamlit interface
st.title("Face Recognition Attendance Application")

# Date and name of the lecture
lecture_date = st.date_input("Select the date of the lecture")
lecture_date = str(lecture_date)

# Check if the date has changed
if st.session_state.previous_date != lecture_date:
    st.session_state.previous_date = lecture_date
    st.session_state.uploader_key += 1  # Increment key to reset the file uploader

# Prompt for lecture name
lecture_name = st.text_input("Enter the name of the lecture")

# Checkbox for including all predictions
include_predictions = st.checkbox("Include all face recognition predictions", value=False)

# File uploader for the image
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"], key=f"uploader_{st.session_state.uploader_key}")

# If an image is uploaded
if uploaded_file is not None:
    # Read the image
    img = Image.open(io.BytesIO(uploaded_file.read()))
    
    # Preprocess the image
    img = preprocess(img)
    
    # Convert image to RGB (stripping alpha channel if exists)
    img = img.convert('RGB')
    
    # Perform face recognition
    faces = face_recogniser(img)
    
    # Display the uploaded image
    st.image(img, caption='Uploaded Image', use_column_width=True)
    
    # Prepare data for Excel
    recognized_names = []

    # Display results
    st.subheader("Recognition Results")
    if faces:
        for idx, face in enumerate(faces):
            st.markdown(f"### Face {idx + 1}")
            st.markdown(f"**Top Prediction:** {face.top_prediction.label} (Confidence: {face.top_prediction.confidence:.2f})")
            st.markdown(f"**Bounding Box:** Left: {face.bb.left}, Top: {face.bb.top}, Right: {face.bb.right}, Bottom: {face.bb.bottom}")
            
            # Add name to recognized names list
            recognized_names.append(face.top_prediction.label)
            
            if include_predictions:
                st.markdown("**All Predictions:**")
                for pred in face.all_predictions:
                    st.markdown(f"- {pred.label}: {pred.confidence:.2f}")
    else:
        st.warning("No faces detected.")
    
    # Update attendance in the Excel sheet stored in session state
    if recognized_names:
        df_attendance = st.session_state.df_attendance  # Use session state
        
        if lecture_date not in df_attendance.columns:
            # Create a new column for this date
            df_attendance[lecture_date] = ""
        
        # Mark 'P' for recognized names
        for name in recognized_names:
            df_attendance.loc[df_attendance['students'] == name, lecture_date] = 'P'
        
        # Update the DataFrame in session state
        st.session_state.df_attendance = df_attendance
        
        st.success(f"Attendance marked for {len(recognized_names)} recognized students.")
        
# Option to stop and download the updated sheet
if st.button('Stop and Download Attendance Sheet'):
    # Save the updated Excel file to a BytesIO object
    output = io.BytesIO()
    st.session_state.df_attendance.to_excel(output, index=False)
    output.seek(0)

    # Provide download option
    st.download_button(label="Download Updated Attendance Sheet", 
                       data=output, 
                       file_name='updated_attendance.xlsx', 
                       mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

    # Reupload the modified file to S3 without causing a script re-run
    if st.button('Reupload to S3') and not st.session_state.reupload_clicked:
        # Set session state to prevent re-execution
        st.session_state.reupload_clicked = True
        s3.put_object(Bucket=bucket_name, Key=attendance_file_key, Body=output.getvalue())
        st.success("File successfully uploaded to S3.")
