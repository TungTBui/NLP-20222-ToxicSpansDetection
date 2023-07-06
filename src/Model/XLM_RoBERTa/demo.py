
import streamlit as st
import torch 
import pandas as pd
import numpy as np
import re

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

# import sys
# sys.path.append("/content/drive/MyDrive/NLP 20222/src/Model/XLM_RoBERTa")

# from predict import predict

auth.authenticate_user()

gauth = GoogleAuth()

gauth.credentials = GoogleCredentials.get_application_default()

drive = GoogleDrive(gauth)

fileDownloaded = drive.CreateFile({"id":"1GlzauYbU-m5iBKs5S3hmiW6Aa4i8_mma"})

fileDownloaded.GetContentFile("checkpoint_xlm_base.pth.tar")

st.set_page_config(page_title="Named Entity Recognition Tagger", page_icon="📘")
st.title("📘 Named Entity Recognition Tagger")

@st.cache(allow_output_mutation=True)
def load_model():

    checkpoint = torch.load("checkpoint_xlm_base.pth.tar")
    start_epoch = checkpoint['epoch'] + 1
    print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
    model = checkpoint['model']

    return model

with st.form(key='my_form'):

    input_text = st.text_input(label='Enter a sentence:', max_chars=250)
    submit_button = st.form_submit_button(label='🏷️ Create tags')

if submit_button:
    if re.sub('\s+','',input_text)=='':
        st.error('Please enter a non-empty sentence.')

#    elif re.match(r'\A\s*\w+\s*\Z', input_text):
#        st.error("Please enter a sentence with at least one word")
    
    else:
        st.markdown("### Tagged Sentence")
        st.header("")

        model = load_model()

        t, ibo_, temp = predict(model, input_text)

        toxic_spans = temp

        st.write(toxic_spans) 
        