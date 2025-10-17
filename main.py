import cv2
import numpy as np
import streamlit as st
import pandas as pd
import time
import base64
import io
from PIL import Image


def resize_image(image, target_width=600):
    # Get original dimensions
    height, width = image.shape[:2]
    # Calculate the aspect ratio and new height
    aspect_ratio = height / width
    target_height = int(target_width * aspect_ratio)
    # Resize the image
    resized_image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)

    return resized_image


def download_image(image, channels, filename="filtered_image.jpg"):
    # Convert NumPy array to PIL Image
    image = resize_image(image)
    if channels == "BGR":
        image = image[:, :, ::-1]  # BGR to RGB
    image = Image.fromarray(image)

    # Convert PIL Image to bytes
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:image/jpeg;base64,{img_str}" download="{filename}">Download Image</a>'
    # Display download button/link
    st.markdown(href, unsafe_allow_html=True)



def pencil_sketch(img, k_size=3, sigma_s = 10):
    # Apply a Gaussian blur
    blurred_image = cv2.GaussianBlur(img, (k_size, k_size), 0, 0)
    img_sketch_bw, img_sketch_color = cv2.pencilSketch(blurred_image, sigma_s=sigma_s)
    return img_sketch_bw, img_sketch_color

def embossed_edges(img):
    kernel = np.array([[0, -3, -3],
                       [3, 0, -3],
                       [3, 3, 0]])

    img_emboss = cv2.filter2D(img, -1, kernel=kernel)

    return img_emboss

st.sidebar.title('Select Image')
img = st.sidebar.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
tab1, tab2, tab3 = st.tabs(["Filter Samples", "Pencil Sketch", "Emboss Edges"])

with tab1:
#    with st.container(border=True):
#        st.image(image)
#        download_image(image, channels="RGB", filename="original.jpg")

    with st.container(border=True):
        st.subheader("Pencil Sketch Filter")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image("pencil_1.jpg")
        with col2:
            st.image("pencil_2.jpg")
        with col3:
            st.image("pencil_3.jpg")
    with st.container(border=True):
        st.subheader("Emboss Edge Filter")
        col4, col5 = st.columns(2)
        with col4:
            st.image("emboss_1.jpg")
        with col5:
            st.image("emboss_2.jpg")

if img is not None:

    file_bytes = np.asarray(bytearray(img.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Convert to grayscale
    #gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    with tab2:
        with st.container(border=True):
            st.image(image, channels="BGR", use_column_width=True)
        with st.container(border=True):
            st.subheader("Pencil Sketch")
            filter_11, filter_12 = st.columns(2)
            with filter_11:
               k_size = st.slider("Blur Kernel", 3, 11, 5, 2)
            with filter_12:
                sigma_s = st.slider("Smoothness", 0, 200, 10)




            img_sketch_bw, img_sketch_color = pencil_sketch(image, k_size, sigma_s)
            st.image(img_sketch_bw, channels="GRAY", use_column_width=True)
            download_image(img_sketch_bw, channels = "GRAY", filename= "bw_sketch.jpg")
            st.image(img_sketch_color, channels="BGR", use_column_width=True)
            download_image(img_sketch_color, channels = "BGR", filename= "color_pencil.jpg")
    with tab3:
        img_emboss = embossed_edges(image)
        st.image(img_emboss, channels="BGR", use_column_width=True)
        download_image(img_emboss, channels="BGR", filename="emboss.jpg")

