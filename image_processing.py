import cv2
import numpy as np
from PIL import Image
import io
import base64
import matplotlib.pyplot as plt
import os

def resize_image(image, target_width=600):
    """Resize image while maintaining aspect ratio."""
    height, width = image.shape[:2]
    aspect_ratio = height / width
    target_height = int(target_width * aspect_ratio)
    return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)

def brighten_image(image, bright=0, contrast=1.0):
    """Adjust brightness and contrast of the image."""
    img_float = image.astype(np.float32)
    adjusted = contrast * (img_float - 128) + 128 + bright
    return np.clip(adjusted, 0, 255).astype(np.uint8)

def download_image(image, channels, original_filename, filter_name):
    """Generate a download link for the image with filename as original_filename_filter_name.jpg."""
    if channels == "BGR":
        image = image[:, :, ::-1]  # Convert BGR to RGB
    image = Image.fromarray(image)
    # Create filename: remove original extension and append filter name
    base_name = os.path.splitext(original_filename)[0]
    filename = f"{base_name}_{filter_name}.jpg"
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f'<a href="data:image/jpeg;base64,{img_str}" download="{filename}">Download Image</a>'

def pencil_sketch(img, k_size=3, sigma_s=10):
    """Apply pencil sketch effect."""
    blurred_image = cv2.GaussianBlur(img, (k_size, k_size), 0, 0)
    img_sketch_bw, img_sketch_color = cv2.pencilSketch(blurred_image, sigma_s=sigma_s)
    return img_sketch_bw, img_sketch_color

def embossed_edges(img):
    """Apply embossed edges effect."""
    kernel = np.array([[0, -3, -3],
                       [3, 0, -3],
                       [3, 3, 0]])
    return cv2.filter2D(img, -1, kernel=kernel)

def sepia(img):
    """Apply sepia tone effect."""
    img_sepia = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float64)
    img_sepia = cv2.transform(img_sepia, np.array([[0.393, 0.769, 0.189],
                                                   [0.349, 0.686, 0.168],
                                                   [0.272, 0.534, 0.131]]))
    img_sepia = np.clip(img_sepia, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img_sepia, cv2.COLOR_RGB2BGR)

def vignette(img, level=2):
    """Apply vignette effect."""
    height, width = img.shape[:2]
    X_resultant_kernel = cv2.getGaussianKernel(width, width / level)
    Y_resultant_kernel = cv2.getGaussianKernel(height, height / level)
    kernel = Y_resultant_kernel * X_resultant_kernel.T
    mask = kernel / kernel.max()
    img_vignette = img.copy()
    for i in range(3):
        img_vignette[:, :, i] = img_vignette[:, :, i] * mask
    return img_vignette

def outline(img, k=9):
    """Apply outline effect."""
    k = max(k, 9)
    kernel = np.array([[-1, -1, -1],
                       [-1, k, -1],
                       [-1, -1, -1]])
    return cv2.filter2D(img, ddepth=-1, kernel=kernel)

def stylization(img,k_size, sigma_s=40, sigma_r=0.1):
    """Apply stylization effect."""
    img_blur = cv2.GaussianBlur(img, (k_size, k_size), 0, 0)
    img_style = cv2.stylization(img_blur, sigma_s=sigma_s, sigma_r=sigma_r)
    return img_style

def generate_histogram(image, channels="BGR"):
    """Generate histogram plot for the image."""
    fig = plt.figure()
    if channels == "GRAY":
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        plt.plot(hist, color='black')
    else:
        hist_b = cv2.calcHist([image], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([image], [1], None, [256], [0, 256])
        hist_r = cv2.calcHist([image], [2], None, [256], [0, 256])
        plt.plot(hist_r, color='red')
        plt.plot(hist_g, color='green')
        plt.plot(hist_b, color='blue')
    plt.gca().get_yaxis().set_visible(False)
    return fig
