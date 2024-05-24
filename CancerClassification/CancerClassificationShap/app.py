import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import shap

classes = {
    "Brain Cancer": {
        0: "Glioma", 1: "Meningioma", 2: "Pituitary Tumor"
    },
    "Breast Cancer": {
        0: "Benign", 1: "Malignant"
    },
    "Cervical Cancer": {
        0: "Dyskeratotic", 1: "Koilocytotic", 2: "Metaplastic", 3: "Parabasal", 4: "Superficial-Intermediat"
    },
    "Kidney Cancer": {
        0: "Normal", 1: "Tumor"
    },
    "Lung and Colon Cancer": {
        0: "Colon Adenocarcinoma", 1: "Colon Benign Tissue", 2: "Lung Adenocarcinoma", 3: "Lung Benign Tissue", 4: "Lung Squamous Cell Carcinoma"
    },
    "Lymphoma": {
        0: "Chronic Lymphocytic Leukemia", 1: "Follicular Lymphoma", 2: "Mantle Cell Lymphoma"
    },
    "Oral Cancer": {
        0: "Normal", 1: "Oral Squamous Cell Carcinoma"
    }
}

def load_background_batch():

    test_dir = './test'
    batch_data = []

    for dirc in os.listdir(test_dir):
        dir_path = os.path.join(test_dir, dirc)
        image_files = os.listdir(dir_path)

        background_data = []

        for img_file in image_files:
            img_path = os.path.join(dir_path, img_file)
            img = image.load_img(img_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)  
            background_data.append(img_array)

        background_batch = np.vstack(background_data)

        batch_data.append(background_batch)


    return batch_data

def load_all_models():
    models_list = []

    for each_model in os.listdir('./models'):
        model = load_model(f'./models/{each_model}', compile=False)
        models_list.append(model)

    return models_list


def predict_class(img, model):
    img = Image.open(img)

    img = img.resize((224, 224))  
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)

    predictions = model.predict(img)
    predicted_class_idx = np.argmax(predictions, axis=1)[0]  # Get the index of the max predicted class
    return predictions, img, predicted_class_idx


def shap_explanation(model, img_array, background):
    explainer = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(img_array)
    return shap_values


def show_shap(shap_values, img_array, predicted_class_idx, class_names):
    
    if len(img_array.shape) == 3:
        img_array = np.expand_dims(img_array, axis=0)

    # Get the SHAP values for the predicted class
    shap_values_for_predicted_class = shap_values[predicted_class_idx]

    # Plotting
    plt.figure()
    shap.image_plot(shap_values_for_predicted_class, img_array)
    plt.show()
    
    # Print the predicted class
    predicted_class_name = class_names[predicted_class_idx]
    st.warning(f"Model predicted: {predicted_class_name}")

    st.write('Inference for the Prediction: Plot of SHAP values')
    st.pyplot(plt.gcf())


def main():

    models_list = load_all_models()
    background_batch_data = load_background_batch()

    cancer_classes = list(classes.keys())

    # model = load_model(f'./models/brain_model.h5', compile=False)


    st.title('Image Classification App')
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    choice = st.selectbox('Select Cancer Type', options=list(classes.keys()))
    
    if uploaded_file is not None and st.button('Predict'):
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        
        model = models_list[cancer_classes.index(choice)]
        background_batch = background_batch_data[cancer_classes.index(choice)]

        predictions, img, predicted_class_idx = predict_class(uploaded_file, model)
        print(np.argmax(predictions))

        shap_values = shap_explanation(model, img, background_batch)
        
        show_shap(shap_values, img, predicted_class_idx, classes[choice])


main()
