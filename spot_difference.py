import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt

def boundingBox(path1, path2):
    # ... (the complete function as provided)

def main():
    st.title("Image Comparison App")
    uploaded_file1 = st.file_uploader("Upload the first image", type=["jpg", "jpeg", "png"])
    uploaded_file2 = st.file_uploader("Upload the second image", type=["jpg", "jpeg", "png"])

    if uploaded_file1 and uploaded_file2:
        image1 = Image.open(uploaded_file1)
        st.image(image1, caption='First Photo', use_column_width=True)
        image2 = Image.open(uploaded_file2)
        st.image(image2, caption='Second Photo', use_column_width=True)

        if st.button('COMPARE'):
            st.write('Result.....')
            with st.spinner("Comparing images..."):
                difference = boundingBox(uploaded_file1, uploaded_file2)

            st.title(f'Predicted Output: {difference}')

            # Display the final output image with bounding boxes
            # Load the images again, as the boundingBox function modifies the images
            image1 = Image.open(uploaded_file1)
            image2 = Image.open(uploaded_file2)

            st.subheader("First Image with Bounding Box")
            st.image(image1, caption="First Image with Bounding Box", use_column_width=True)

            st.subheader("Second Image with Bounding Box")
            st.image(image2, caption="Second Image with Bounding Box", use_column_width=True)

if __name__ == "__main__":
    main()

