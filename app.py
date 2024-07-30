from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from google.generativeai.types.safety_types import HarmBlockThreshold, HarmCategory

from langchain_core.runnables import RunnablePassthrough
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyPDFLoader
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_chroma import Chroma

from flask import Flask, request, redirect, url_for, render_template, jsonify, session
from IPython.display import Markdown
from model import *
import tempfile
import textwrap
import pickle
import os


GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001", google_api_key=GOOGLE_API_KEY)


Gemini = chat_model(GOOGLE_API_KEY)
question_refiner = question_model(GOOGLE_API_KEY)

question_refine_chain = question_chain(question_refiner)

model_objects = {}

# def to_markdown(text):
#   text = text.replace('•', '  *')
#   return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/user_upload')
def user_upload():
    return render_template('upload.html')


'''@app.route('/process_upload', methods=['POST'])
def process_upload():
    file = request.files['file']
    #file.get
    
    if file and file.filename.endswith('.pdf'):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
            tmpfile.write(file.read())  # Use file.read() to get the content
            tmpfile.flush()  # Ensure the data is written to disk
            pdf_loader = PyPDFLoader(tmpfile.name)
            pages = pdf_loader.load_and_split()
            print(pages[3])

        DB = Chroma.from_documents(pages, embeddings)
        retriever = DB.as_retriever(search_type="similarity", search_kwargs={"k": 3})

        model_response_chain = response_chain(retriever, Gemini=Gemini)
        memory = ConversationBufferMemory(memory_key="chat_history", input_key="question")
        
        return jsonify({'message': 'File processed successfully'})
    
    else:
        return jsonify({'error': 'Invalid file type'}), 400
'''

@app.route('/process_upload', methods=['POST'])
def process_upload():
    file = request.files['file']
    if file and file.filename.endswith('.pdf'):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
            tmpfile.write(file.read())
            tmpfile_path = tmpfile.name
        
        # Initialize model here
        pdf_loader = PyPDFLoader(tmpfile_path)
        pages = pdf_loader.load_and_split()
        DB = Chroma.from_documents(pages, embeddings)
        retriever = DB.as_retriever(search_type="similarity", search_kwargs={"k": 3})

        memory = ConversationBufferMemory(memory_key="chat_history", input_key="question")
        model_response_chain = response_chain(retriever, Gemini=Gemini)

        # Store the model response chain and other objects in the global variable
        model_objects['model_response_chain'] = model_response_chain
        model_objects['retriever'] = retriever
        model_objects['memory'] = memory

        return jsonify({'message': 'File processed successfully'})
    else:
        return jsonify({'error': 'Invalid file type'}), 400

@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

@app.route('/model_response', methods=['POST'])
def model_response():
    user_input = request.json.get('user_input')
    if not user_input:
        return jsonify({'error': 'No input provided'}), 400
    
    memory = model_objects.get('memory')

    model_question = question_refine_chain.invoke({"chat_history":memory.buffer, "question": user_input})
    model_response_chain = model_objects.get('model_response_chain')

    if not model_response_chain:
        return jsonify({'error': 'Model not initialized'}), 500

    
    response = model_response_chain.invoke(model_question)
    if response.startswith("answer not available in context"):
        response = model_response_chain.invoke(user_input)
        memory.save_context({"question": user_input}, {"response": response})
        return jsonify({'response': response})
    
    else:
        memory.save_context({"question": user_input}, {"response": response})
        return jsonify({'response': response})
    

if __name__ == '__main__':
    app.run(debug=True)