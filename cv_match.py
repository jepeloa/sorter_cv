# Import libraries
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from numpy.linalg import norm
from termcolor import colored
import pandas as pd
import numpy as np
import requests
import PyPDF2
import re
import plotly.graph_objects as go
import nltk
nltk.download('punkt')
from googletrans import Translator
import os
import streamlit as st
import sqlite3
import http.server
import socketserver
import threading
from PIL import Image
import openai
openai.api_key = "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"
import plotly.express as px
#import slate3k as slate
#!pip install slate3k



image = Image.open('logo.png')

st.sidebar.image(image, caption=' ', width=200)

conn = sqlite3.connect('pdf_database.db')
cursor = conn.cursor()

cursor.execute('''
CREATE TABLE IF NOT EXISTS mis_documentos (
    id INTEGER PRIMARY KEY,
    Filename TEXT
)
''')


def chatgpt_filter(df, path_folder='./CV'):
    """Imprime el contenido de los primeros 5 archivos PDF en el DataFrame.

    Args:
    - df (pd.DataFrame): DataFrame con los nombres de los archivos.
    - path_folder (str): Ruta del directorio donde están los archivos, por defecto es el directorio actual.

    Returns:
    None.
    """
    # Selecciona los primeros 5 nombres de archivos
    filenames = df['Filename'].head(5)
    
    for filename in filenames:
        full_path = f"{path_folder}/{filename}"
        try:
            with open(full_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page_num in range(reader.numPages):
                    page = reader.getPage(page_num)
                    print(page.extractText())
        except Exception as e:
            print(f"Error al abrir el archivo {filename}. Error: {e}")




path_to_folder = './CV/'
translator = Translator()     #Funcion para traducir al español

def translate_text(text,src,dest):
    if text == "" or text is None:
        pass
    else:
        return translator.translate(text, src=src, dest=dest).text
    

def delete_table_contents():
    try:
        conn = sqlite3.connect('pdf_database.db')
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM pdf_list')
        conn.commit()
        conn.close()
        
        return "Contenido de la tabla borrado con éxito."
    except Exception as e:
        return f"Error al borrar el contenido de la tabla: {e}"




def preprocess_text(text):
    # Convert the text to lowercase
    text = text.lower()
    
    # Remove punctuation from the text
    text = re.sub('[^a-z]', ' ', text)
    
    # Remove numerical values from the text
    text = re.sub(r'\d+', '', text)
    
    # Remove extra whitespaces
    text = ' '.join(text.split())
    
    return text

def evaluate_candidate(input_CV,input_JD,model):
    v1 = model.infer_vector(input_CV.split())
    v2 = model.infer_vector(input_JD.split())
    similarity = 100*(np.dot(np.array(v1), np.array(v2))) / (norm(np.array(v1)) * norm(np.array(v2)))
    return round(similarity, 2)






def process_JD_and_get_matches(jd_en,model):
    all_files = os.listdir(path_to_folder)
    df = pd.DataFrame(columns=['Filename', 'MatchValue'])
    # Filtra solo los archivos PDF
    pdf_files = [file for file in all_files if file.endswith('.pdf')]
    j=0 #contador de archivos
    for pdf_file in pdf_files:
        pdf_path = os.path.join(path_to_folder, pdf_file)
    
        with open(pdf_path, 'rb') as f:
            pdf = PyPDF2.PdfReader(f)
            resume_en = ''
            for i in range(len(pdf.pages)):
                pageObj = pdf.pages[i]
                text_to_translate=pageObj.extract_text()
                print("longitud " + str(len(text_to_translate)))
                if len(text_to_translate)<5000 and len(text_to_translate)>0:
                    resume_en += translate_text(text_to_translate,'es','en')
                if len(text_to_translate)>5000:
                    resume_en += translate_text(text_to_translate[:5000],'es','en')

        #print("longitud" + str(len(resume_en)))
        #resume_en=translate_text(resume,'es','en')
        input_CV = preprocess_text(resume_en)
        input_JD = preprocess_text(jd_en)
        match=evaluate_candidate(input_CV,input_JD, model)
        print(match)
        df.loc[j] = {'Filename': pdf_file, 'MatchValue': match}
        j=j+1
        print("="*50)
    df['MatchValue'] = df['MatchValue'].astype(float)
    df_sorted = df.sort_values(by='MatchValue', ascending=False)
    return df_sorted


def store_to_sqlite(df):
    conn = sqlite3.connect('pdf_database.db')
    df.to_sql('pdf_list', conn, if_exists='replace', index=False)  # Guarda el dataframe en la tabla 'pdf_list'
    conn.close()


def read_from_sqlite():
    conn = sqlite3.connect('pdf_database.db')
    df = pd.read_sql('SELECT * FROM pdf_list', conn)
    conn.close()
    return df

def init_db():
    conn = sqlite3.connect('pdf_database.db')
    cursor = conn.cursor()
    
    # Crear la tabla "pdf_list" si no existe
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS pdf_list (
            Filename TEXT
        )
    ''')
    
    conn.commit()
    conn.close()

def start_pdf_server(port=8080):
    handler = http.server.SimpleHTTPRequestHandler
    httpd = socketserver.TCPServer(("", port), handler)
    thread = threading.Thread(target=httpd.serve_forever)
    thread.start()
    print(f"PDF server started at port {port}")

# Inicia el servidor al comienzo del script


def main():
    selected_pdf=pd.DataFrame()
    init_db()
    try:
        start_pdf_server()
    except:
        pass
    

    df_sorted=pd.DataFrame
    model = Doc2Vec.load('cv_job_maching.model')
    st.title("CV Sorter")
    st.write("Insert the job description and get the matching CVs.")

    # Text area for the user to input the job description
    jd = st.text_area("Job Description", "")

    if st.button("Process"):
        if jd:
            df_sorted = process_JD_and_get_matches(jd,model)
            chatgpt_filter(df_sorted)
            store_to_sqlite(df_sorted)
            #st.write(df_sorted)
        else:
            st.write("Please enter a job description to process.")
    if st.button('Borrar contenido de la tabla'):
        message = delete_table_contents()
        st.write(message)
    df_sorted_from_db = read_from_sqlite()
    if not df_sorted_from_db.empty:
        selected_pdf = st.selectbox('Elige un PDF:', df_sorted_from_db['Filename'].tolist())
        pdf_url = f"http://localhost:8080/CV/{selected_pdf}"
        st.markdown(f'<iframe src="{pdf_url}" width="700" height="400"></iframe>', unsafe_allow_html=True)
        fig = px.scatter(df_sorted_from_db, x="Filename", y="MatchValue", title="Match Values por Filename", height=1000)
        fig.update_traces(mode="lines+markers")
        st.plotly_chart(fig)
        st.write(df_sorted_from_db)
if __name__ == "__main__":
    main()