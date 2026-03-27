import streamlit as st
import pandas as pd
import numpy as np
import keras
import folium
from exif import Image as exif
import geopandas as gpd
import plotly.express as px
import matplotlib.pyplot as plt
from streamlit_folium import folium_static
from PIL import Image

from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.utils import load_img, img_to_array


st.title('CGT575 / ASM591 Lab 6')
st.write('Deploying Deep Learning Models on Web and Smartphone Applications Using Streamlit API')

task = st.sidebar.selectbox('Select Page', ['Homepage', 'Deep Learning', 'Mapping', 'Homework'])


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

        # Allow users to upload images
        uploaded_file = st.file_uploader('Upload Image', type=['jpg', 'jpeg', 'png'])

        if uploaded_file is not None:
            pil_img = Image.open(uploaded_file)
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

        if uploaded_file is not None:
            pil_img = Image.open(uploaded_file)
            st.image(pil_img)

            uploaded_file.seek(0)
            lat, lon = extract_coordinates(uploaded_file)

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
            else:
                st.error('No EXIF GPS data found in this image.')


pages()
