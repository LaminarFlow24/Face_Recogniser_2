import io
import joblib
import pandas as pd
from PIL import Image
import streamlit as st
from face_recognition import preprocessing

# Cache the attendance data to prevent reloading
@st.cache_data
def load_attendance_data():
    return pd.read_excel('students_attendance_september.xlsx')

# Cache the model so it's loaded only once
@st.cache_resource
def load_face_recogniser_model():
    return joblib.load('model/face_recogniserL.pkl')

# Initialize session state for attendance DataFrame
if 'df_attendance' not in st.session_state:
    st.session_state.df_attendance = load_attendance_data()

# Initialize session state for tracking date changes
if 'previous_date' not in st.session_state:
    st.session_state.previous_date = None

# Initialize session state for resetting the uploader key
if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = 0

# Load the face recognizer model
face_recogniser = load_face_recogniser_model()
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
    # Save the updated Excel file
    output = io.BytesIO()
    st.session_state.df_attendance.to_excel(output, index=False)
    output.seek(0)

    # Provide download option
    st.download_button(label="Download Updated Attendance Sheet", 
                       data=output, 
                       file_name='updated_attendance.xlsx', 
                       mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
