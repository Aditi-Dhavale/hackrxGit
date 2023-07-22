# To explore mobilenetv2
# To explore VGG16
# import os#all below imported packeages are important

import cv2
from PIL import Image
import numpy as np
import streamlit as st
from skiimage import boundingBox

# import pandas as pd # pip install pandas
# from matplotlib import pyplot as plt # pip install matplotlib
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'#Used to avoid printing of Warnings
st.title("Deep Learning based Visual Testing Tool for Website Design Validation Developed by PixelProphets")
# get the path/directory
folder_dir = "C://aditi//competitions//hackerx4//test"

# streamlite commands

st.set_option('deprecation.showfileUploaderEncoding', False)
#st.title('Website Design Assessment Tool using Deep Learning')
st.text(
    'Upload the Image from the listed category.\n[good design or bad design]')

uploaded_file1 = st.file_uploader("Choose the first photo", type=["jpg", "jpeg", "png"])
""" if uploaded_file1 is not None:
    image1 = Image.open(uploaded_file1)
    st.image(image1, caption='First Photo', use_column_width=True) """

uploaded_file2 = st.file_uploader("Choose the second photo", type=["jpg", "jpeg", "png"])
""" if uploaded_file2 is not None:
    image2 = Image.open(uploaded_file2)
    st.image(image2, caption='Second Photo', use_column_width=True)
 """
if uploaded_file1 and uploaded_file2 is not None:
    image1 = Image.open(uploaded_file1)
    st.image(image1, caption='First Photo', use_column_width=True)
    image2 = Image.open(uploaded_file2)
    st.image(image2, caption='Second photo')


    if st.button('COMPARE'):        

        st.write('Result.....')
        flat_data = []
        imge = np.array(image1)
        print(imge.shape)

        # y = np.expand_dims(norm_image, axis=0)
        # print(y.shape)

        difference = boundingBox(uploaded_file1,uploaded_file2) 
        #y_out4="ADEQUATE NUMBER OF COLORS USED IN WEBSITE DESIGN"

        #y_out5 = rate_text_amount(uploaded_file)

        #st.title(f' PREDICTED OUTPUT: {y_out1}')
        #st.title(f' PREDICTED OUTPUT: {difference}')
        st.image(difference, caption='Result photo')

        # q = saved_model.predict_proba(y)
        # for index, item in enumerate(Categories):
        #   st.write(f'{item} : {q[0][index]*100}%')

st.text("")
st.text('Made by PixelProphets')
