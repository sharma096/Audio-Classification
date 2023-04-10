import streamlit as st
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import  Input, Dense, Dropout,Activation, Flatten, Embedding, LSTM
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.optimizers import Adam
import tqdm
import time
import numpy as np
# !pip install librosa
import librosa
import os
import zipfile
import matplotlib.pyplot as plt
import IPython
import pickle

st.image('https://img.securityinfowatch.com/files/base/cygnus/siw/image/2021/03/GettyImages_497748804.603d37fa3061a.png?auto=format,compress&fit=fill&fill=blur&w=1200&h=630')

os.chdir('C:/Users/TanishSharma/OneDrive - TheMathCompany Private Limited/Desktop/Audio Classifications')

# load the saved model
with open("ANN_Model.pickle", "rb") as f:
    ANN_Model = pickle.load(f)

st.title('Audio Classification system')
uploaded_file=st.file_uploader("Choose an Audio file",type=[".wav","wave",".flac",".mp3"], accept_multiple_files=False)

###################### Method to Save Uploaded Image into local############################


def Save_audio(upload_audio):
    try:
        if not os.path.exists("uploads"):
            os.makedirs("uploads")
        save_path = os.path.join(os.getcwd(), "uploads", upload_audio.name)
        with open(save_path, 'wb') as f:
            f.write(upload_audio.getbuffer())
        return save_path
    except Exception as e:
        print("Error saving file:", e)
        return None
def extract_feature(file):
    data, sample_rates=librosa.load(file)
    mfcc_features=librosa.feature.mfcc(y=data,sr=sample_rates,n_mfcc=40)
    mfcc_scaled_feature=np.mean(mfcc_features.T,axis=0)
    return mfcc_scaled_feature
extract_features=[]

if uploaded_file is not None:
    if Save_audio(uploaded_file):
        audio_bytes = uploaded_file.read()
        st.audio(audio_bytes, format="audio/wav")
        extract_features.append(extract_feature(os.path.join("uploads",uploaded_file.name)))
        progress_text = "Hold on! Result will shown below."
        my_bar = st.progress(0, text=progress_text)
        for percent_complete in range(100):
            time.sleep(0.02)
            my_bar.progress(percent_complete + 1, text=progress_text) ## to add progress bar untill feature got extracted
        

    # use the loaded model for prediction
    predictions = ANN_Model.predict(np.array(extract_features))
    pred_class = np.argmax(predictions)

    with open("Categories.pickle", "rb") as f:
        Categories = pickle.load(f)
    class_cat=Categories[Categories['Class_ID']==pred_class]['Category']

    # st.markdown("""This Uploaded sound clip is""" :,""" and bold""")

    # highlighted_text = f"<span style='background-color: yellow'>{np.array(class_cat)[0]}</span>"
    bold_text = f"<t>{np.array(class_cat)[0]}</t>"
    st.write(f'<span style="font-size:20px;">This Uploaded sound clip is {bold_text}</span>', unsafe_allow_html=True)