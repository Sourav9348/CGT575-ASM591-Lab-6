import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import keras
import folium
from exif import Image as exif
import geopandas as gpd
import plotly.express as px
import matplotlib.pyplot as plt
from streamlit_folium import folium_static
from PIL import Image

from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras.utils import load_img, img_to_array

# NEW IMPORTS FOR OBJECT DETECTION
import cv2


st.title('CGT575 / ASM591 Lab 6')
st.write('Deploying Deep Learning Models on Web and Smartphone Applications Using Streamlit API')

# ✅ Added "Object Detection" here
task = st.sidebar.selectbox('Select Page', ['Homepage', 'Deep Learning', 'Mapping', 'Homework', 'Object Detection'])


# Helper functions for extracting geocoordinates from EXIF data
def decimal_coords(coords, ref):
    decimal_degrees = coords[0] + coords[1] / 60 + coords[2] / 3600
    if ref in ('S', 'W'):
        decimal_degrees = -decimal_degrees
    return decimal_degrees


def extract_coordinates(uploaded_file):
    img_exif = exif(uploaded_file)
    if img_exif.has_exif:
        lat = decimal_coords(img_exif.gps_latitude, img_exif.gps_latitude_ref)
        lon = decimal_coords(img_exif.gps_longitude, img_exif.gps_longitude_ref)
        return lat, lon
    return None, None


# ✅ Simple object detection function (MobileNet SSD using OpenCV)
def detect_objects(image_path):
    net = cv2.dnn.readNetFromCaffe(
        'MobileNetSSD_deploy.prototxt.txt',
        'MobileNetSSD_deploy.caffemodel'
    )

    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant",
               "sheep", "sofa", "train", "tvmonitor"]

    image = cv2.imread(image_path)
    (h, w) = image.shape[:2]

    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)),
                                 0.007843, (300, 300), 127.5)

    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.4:
            idx = int(detections[0, 0, i, 1])
            label = CLASSES[idx]

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            cv2.rectangle(image, (startX, startY), (endX, endY),
                          (0, 255, 0), 2)
            cv2.putText(image, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image


def pages():

    # ── HOMEPAGE ──
    if task == 'Homepage':
        st.header('Homepage')
        m = folium.Map(location=[40.422989, -86.921776], zoom_start=15)
        folium_static(m)

    # ── DEEP LEARNING ──
    elif task == 'Deep Learning':
        st.header('Deep Learning')

        model = VGG16(weights='imagenet')

        img_path = 'car.jpg'

        img = load_img(img_path, color_mode='rgb', target_size=(224, 224))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        features = model.predict(x)
        p = decode_predictions(features)

        st.image(img_path)
        for item in p[0]:
            st.write(item[1], ' : ', str(item[2] * 100) + '%')

        st.subheader('Try your own image')
        uploaded_file = st.file_uploader('Upload Image', type=['jpg', 'jpeg', 'png'])

        use_example = st.button('Or use example image (motorbike.jpg)')

        if uploaded_file is not None:
            pil_img = Image.open(uploaded_file)
        elif use_example:
            pil_img = Image.open('motorbike.jpg')
        else:
            pil_img = None

        if pil_img is not None:
            st.image(pil_img)

            pil_img_resized = pil_img.resize((224, 224))
            x_upload = img_to_array(pil_img_resized)
            x_upload = np.expand_dims(x_upload, axis=0)
            x_upload = preprocess_input(x_upload)

            features_upload = model.predict(x_upload)
            p_upload = decode_predictions(features_upload)

            for item in p_upload[0]:
                st.write(item[1], ' : ', str(item[2] * 100) + '%')

    # ── MAPPING ──
    elif task == 'Mapping':
        st.header('Mapping')

        latitude = 40.422989
        longitude = -86.921776

        m = folium.Map(location=[latitude, longitude], zoom_start=15)
        folium.Marker(
            [latitude, longitude],
            popup='Lab6',
            tooltip='Lab6'
        ).add_to(m)
        folium_static(m)

    # ── HOMEWORK ──
    elif task == 'Homework':
        st.header('Homework')

        uploaded_file = st.file_uploader('Upload Image')

        use_example_hw = st.button('Or use example image (img.JPG)')

        if uploaded_file is not None:
            pil_img = Image.open(uploaded_file)
            st.image(pil_img)
            uploaded_file.seek(0)
            lat, lon = extract_coordinates(uploaded_file)
        elif use_example_hw:
            pil_img = Image.open('img.JPG')
            st.image(pil_img)
            with open('img.JPG', 'rb') as f:
                lat, lon = extract_coordinates(f)
        else:
            pil_img = None
            lat, lon = None, None

        if lat is not None and lon is not None:
            st.write(f'Latitude: {lat}')
            st.write(f'Longitude: {lon}')

            m = folium.Map(location=[lat, lon], zoom_start=16)
            folium.Marker(
                [lat, lon],
                popup='Image Location',
                tooltip='Image Location'
            ).add_to(m)
            folium_static(m)
        elif pil_img is not None:
            st.error('No EXIF GPS data found in this image.')

    # ── OBJECT DETECTION ──  NEW PAGE
    elif task == 'Object Detection':
        st.header('Object Detection')

        st.write('Displaying object detection results on 5 images')

        image_paths = [
            'image1.jpg',
            'image2.jpg',
            'image3.jpg',
            'image4.jpg',
            'image5.jpg'
        ]

        for img_path in image_paths:
            st.subheader(img_path)

            detected_img = detect_objects(img_path)

            # Convert BGR (OpenCV) → RGB
            detected_img = cv2.cvtColor(detected_img, cv2.COLOR_BGR2RGB)

            st.image(detected_img, caption='Detected Objects', use_column_width=True)


pages()
