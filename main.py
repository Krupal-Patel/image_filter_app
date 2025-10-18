import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from image_processing import (resize_image, brighten_image, download_image,
                             pencil_sketch, embossed_edges, sepia,
                             vignette, outline, stylization, generate_histogram)

# Sidebar for image upload and histogram toggle
st.sidebar.title('Select Image')
img = st.sidebar.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
show_histogram = st.sidebar.checkbox("Show Histogram", value=False)

# Tabs for different functionalities
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Filter Samples", "Basic Editing",
                                                     "Pencil Sketch", "Emboss Edges",
                                                     "Basic Filters", "Stylization", "Outline"])

# Tab 1: Filter Samples (unchanged)
with tab1:
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

# Process image only if uploaded
if img is not None:
    original_filename = img.name  # Store the original filename
    file_bytes = np.asarray(bytearray(img.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    #image = resize_image(image)  # Resize once to optimize performance

    # Tab 2: Basic Editing
    with tab2:
        with st.container(border=True):
            st.subheader("Basic Editing")
            bright = st.slider("Adjust Brightness", -255, 255, 0, 1)
            contrast = st.slider("Adjust Contrast", 0.0, 3.0, 1.0, 0.1)
            col1, col2, col3 = st.columns([2, 2, 1])
            with col1:
                st.image(image[:, :, ::-1], caption="Original")
            with col2:
                with st.spinner("Processing..."):
                    img_bright = brighten_image(image, bright=bright, contrast=contrast)
                st.image(img_bright[:, :, ::-1], caption="Edited")
                st.markdown(download_image(img_bright, channels="BGR",
                                          original_filename=original_filename, filter_name="edited"),
                            unsafe_allow_html=True)
            with col3:
                if show_histogram:
                    fig = generate_histogram(img_bright)
                    st.pyplot(fig)
                    plt.close(fig)
                # Check for clipping
                total_pixels = img_bright.size // 3
                threshold = total_pixels * 0.01
                hist_b = cv2.calcHist([img_bright], [0], None, [256], [0, 256])
                hist_g = cv2.calcHist([img_bright], [1], None, [256], [0, 256])
                hist_r = cv2.calcHist([img_bright], [2], None, [256], [0, 256])
                clipping_detected = any(hist[0] > threshold or hist[255] > threshold
                                       for hist in [hist_b, hist_g, hist_r])
                if clipping_detected:
                    st.warning("Warning: Image details may be lost!")

    # Tab 3: Pencil Sketch
    with tab3:
        with st.container(border=True):
            st.subheader("Pencil Sketch")
            filter_11, filter_12 = st.columns(2)
            with filter_11:
                k_size = st.slider("Blur Kernel", 3, 11, 5, 2)
            with filter_12:
                sigma_s = st.slider("Smoothness", 0, 200, 10)
            with st.spinner("Processing..."):
                img_sketch_bw, img_sketch_color = pencil_sketch(image, k_size, sigma_s)
            col1, col2, col3 = st.columns([2, 2, 1])
            with col1:
                st.image(img_sketch_bw, channels="GRAY", caption="Black & White Sketch")
                st.markdown(download_image(img_sketch_bw, channels="GRAY",
                                          original_filename=original_filename, filter_name="bw_sketch"),
                            unsafe_allow_html=True)
            with col2:
                st.image(img_sketch_color, channels="BGR", caption="Color Sketch")
                st.markdown(download_image(img_sketch_color, channels="BGR",
                                          original_filename=original_filename, filter_name="color_pencil"),
                            unsafe_allow_html=True)
            with col3:
                if show_histogram:
                    fig = generate_histogram(img_sketch_color)
                    st.pyplot(fig)
                    plt.close(fig)

    # Tab 4: Emboss Edges
    with tab4:
        with st.container(border=True):
            st.subheader("Emboss Edges")
            col1, col2 = st.columns([2, 1])
            with col1:
                with st.spinner("Processing..."):
                    img_emboss = embossed_edges(image)
                st.image(img_emboss, channels="BGR", caption="Embossed")
                st.markdown(download_image(img_emboss, channels="BGR",
                                          original_filename=original_filename, filter_name="emboss"),
                            unsafe_allow_html=True)
            with col2:
                if show_histogram:
                    fig = generate_histogram(img_emboss)
                    st.pyplot(fig)
                    plt.close(fig)

    # Tab 5: Basic Filters
    with tab5:
        with st.container(border=True):
            st.subheader("Basic Filters")
            level = st.slider("Vignette Level", 1, 10, 1, 1)
            with st.spinner("Processing..."):
                img_sepia = sepia(image)
                img_vignette = vignette(image, level=level)
                sepia_vignette = vignette(img_sepia, level=level)
            # Display each filter with its histogram
            for filter_name, img_filtered, channels, filename_suffix in [
                ("Original", image, "BGR", "original"),
                ("Sepia", img_sepia, "BGR", "sepia"),
                ("Vignette", img_vignette, "BGR", "vignette"),
                ("Sepia + Vignette", sepia_vignette, "BGR", "sepia_vignette")
            ]:
                st.subheader(filter_name)
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.image(img_filtered, channels=channels, caption=filter_name)
                    st.markdown(download_image(img_filtered, channels=channels,
                                              original_filename=original_filename,
                                              filter_name=filename_suffix),
                                unsafe_allow_html=True)
                with col2:
                    if show_histogram:
                        fig = generate_histogram(img_filtered, channels=channels)
                        st.pyplot(fig)
                        plt.close(fig)

    # Tab 6: Stylization
    with tab6:
        with st.container(border=True):
            st.subheader("Stylization")
            # User-defined stylization
            st.write("User-Defined Settings")
            k_size = st.slider("Blur Kernel", 3, 11, 3, 2)
            sigma_s = st.slider("sigma_s", 0, 200, 40, 1)
            sigma_r = st.slider("sigma_r", 0.0, 1.0, 0.1, 0.1)
            with st.spinner("Processing User-Defined Stylization..."):
                img_style_user = stylization(image, k_size=k_size, sigma_s=sigma_s, sigma_r=sigma_r)
            col1, col2 = st.columns([2, 1])
            with col1:
                st.image(img_style_user, channels="BGR", caption="User-Defined Stylization")
                st.markdown(download_image(img_style_user, channels="BGR",
                                          original_filename=original_filename, filter_name="style_user"),
                            unsafe_allow_html=True)
            with col2:
                if show_histogram:
                    fig = generate_histogram(img_style_user)
                    st.pyplot(fig)
                    plt.close(fig)

            # Preprogrammed stylizations
            st.write("Preprogrammed Settings")
            presets = [
                ("Subtle Stylization", 3, 20, 0.05, "style_subtle"),
                ("Bold Stylization", 5, 80, 0.2, "style_bold"),
                ("Smooth Cartoon", 7, 40, 0.1, "style_cartoon"),
                ("Detailed Edges", 3, 60, 0.3, "style_edges")
            ]
            for preset_name, k_size, sigma_s, sigma_r, filename_suffix in presets:
                st.subheader(preset_name)
                with st.spinner(f"Processing {preset_name}..."):
                    img_style = stylization(image, k_size=k_size, sigma_s=sigma_s, sigma_r=sigma_r)
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.image(img_style, channels="BGR", caption=preset_name)
                    st.markdown(download_image(img_style, channels="BGR",
                                              original_filename=original_filename,
                                              filter_name=filename_suffix),
                                unsafe_allow_html=True)
                with col2:
                    if show_histogram:
                        fig = generate_histogram(img_style)
                        st.pyplot(fig)
                        plt.close(fig)

    # Tab 7: Outline
    with tab7:
        with st.container(border=True):
            st.subheader("Outline")
            k = st.slider("Outline K", 9, 12, 9, 1)
            with st.spinner("Processing..."):
                img_outline = outline(image, k=k)
            col1, col2 = st.columns([2, 1])
            with col1:
                st.image(img_outline, channels="BGR", caption="Outline")
                st.markdown(download_image(img_outline, channels="BGR",
                                          original_filename=original_filename, filter_name="outline"),
                            unsafe_allow_html=True)
            with col2:
                if show_histogram:
                    fig = generate_histogram(img_outline)
                    st.pyplot(fig)
                    plt.close(fig)
