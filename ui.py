import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
from model import predict_from_array
import tensorflow as tf
from model import CRNN

# Your existing code here (all the imports and functions)
# ... [paste all your existing code here] ...

# Streamlit UI
st.set_page_config(page_title="Text Extraction from Images", layout="wide")

st.title("Text Extraction from Images")
st.write("Upload an image to extract text using CRNN model")

# Model selection
model_option = st.sidebar.radio(
    "Select Model Version",
    ("60 Epochs", "20 Epochs"),
    index=0
)

# Load the appropriate model
if model_option == "60 Epochs":
    model = tf.keras.models.load_model('crnn_model_60.keras', custom_objects={"CRNN": CRNN})
else:
    model = tf.keras.models.load_model('crnn_model.keras', custom_objects={"CRNN": CRNN})

# File upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        image.save(tmp_file.name)
        image_path = tmp_file.name
    
    # Process buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Extract Full Text"):
            with st.spinner('Processing full image...'):
                try:
                    predicted_text = predict_from_array(model, cv2.imread(image_path))
                    final_text = predicted_text.strip().replace('[UNK]', '')
                    st.success("Extracted Text:")
                    st.text_area("Result", final_text, height=100)
                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")
    
    with col2:
        if st.button("Extract Words Separately"):
            with st.spinner('Processing words...'):
                try:
                    # Create a container for results
                    result_container = st.container()
                    
                    # Process the image
                    img = cv2.imread(image_path)
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
                    dilated = cv2.dilate(thresh, kernel, iterations=1)
                    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    bounding_boxes = [cv2.boundingRect(c) for c in contours]
                    
                    # Group into lines
                    lines = []
                    tolerance = 20
                    for box in bounding_boxes:
                        x, y, w, h = box
                        cy = y + h // 2
                        added = False
                        for line in lines:
                            ly = line[0][1] + line[0][3] // 2
                            if abs(ly - cy) < tolerance:
                                line.append(box)
                                added = True
                                break
                        if not added:
                            lines.append([box])
                    
                    for line in lines:
                        line.sort(key=lambda b: b[0])
                    lines.sort(key=lambda line: min(b[1] for b in line))
                    
                    sorted_boxes = [box for line in lines for box in line]
                    
                    final_text = ""
                    word_images = []
                    
                    for x, y, w, h in sorted_boxes:
                        if w > 50 and h > 30:
                            word_img = img[y:y+h, x:x+w]
                            word_images.append((x, y, w, h, word_img))
                    
                    # Display results
                    result_container.write("Detected Words:")
                    cols = st.columns(4)
                    col_idx = 0
                    
                    full_text = ""
                    for i, (x, y, w, h, word_img) in enumerate(word_images):
                        predicted_text = predict_from_array(model, word_img)
                        clean_text = predicted_text.strip().replace('[UNK]', '')
                        full_text += clean_text + " "
                        
                        with cols[col_idx]:
                            st.image(word_img, caption=f"Word {i+1}: {clean_text}", width=150)
                            col_idx = (col_idx + 1) % 4
                    
                    st.success("Combined Text:")
                    st.text_area("Full Text", full_text.strip(), height=100)
                    
                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")
    
    # Clean up
    os.unlink(image_path)

# Instructions
st.sidebar.markdown("""
### Instructions:
1. Upload an image containing text
2. Choose to either:
   - Extract text from the whole image at once
   - Extract words separately (better for multi-line text)
3. View the results

### Model Options:
- **60 Epochs**: More accurate but slower
- **20 Epochs**: Faster but less accurate
""")