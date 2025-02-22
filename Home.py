import streamlit as st
from PIL import Image  # Recommended for handling local images

# Page configuration
st.set_page_config(page_title="Medical App", page_icon="🩺", layout="wide")

# Custom CSS for styling (optional)
st.markdown(
    """
    <style>
    .block-container {
        background-color: rgba(255, 255, 255, 0.9);
        padding: 2rem;
        border-radius: 10px;
        margin-top: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Display a local image stored in your project folder
# Make sure "my_local_image.png" exists in the same directory as your app, or provide the relative path.
local_image = Image.open("pages/img.webp")

# Create three columns; the middle column will center the image
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image(local_image, width=500)

# Page title and description
st.title("Welcome to the Medical Dashboard 🩺")
st.write("""
Welcome to our Medical Dashboard! Use the sidebar menu to navigate between pages (Home, Dashboard, Patients, etc.).
This project provides an example of analyzing and managing no-show data in the healthcare sector.
""")

# A meaningful quote related to healthcare and appointments
st.markdown(
    """
    <div style="text-align:center; margin-top: 2rem;">
        <h2 style="color:#007bff;">"Healing is a matter of time, but it is sometimes also a matter of opportunity."</h2>
        <p style="font-size:1.2rem;">- Hippocrates</p>
    </div>
    """,
    unsafe_allow_html=True
)
