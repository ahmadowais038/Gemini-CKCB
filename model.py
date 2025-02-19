import os

#from langchain_google_vertexai import VertexAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from google.generativeai.types.safety_types import HarmBlockThreshold, HarmCategory

from langchain_core.runnables import RunnablePassthrough
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.llms import Replicate
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_chroma import Chroma

GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001", google_api_key=GOOGLE_API_KEY)

safety_settings = {
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
}

"""
   Gemini = VertexAI(
      model="gemini-1.5-pro",
      project="llama-ckcb",
      location="us-central1",
      temperature=0.7,
      top_p=0.9,
      #top_k=55,
      max_output_tokens=5000
    )
"""

def chat_model(API_KEY):
    Gemini = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        google_api_key=API_KEY,
        safety_settings=safety_settings,
        temperature=0.7,
        top_p=0.9,
        top_k=55,
        max_output_tokens=5000
    )
    return Gemini

def question_model(API_KEY):
    question_model = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        google_api_key=GOOGLE_API_KEY,
        safety_settings=safety_settings,
        temperature=0.5,
        top_p=0.7,
        top_k=40,
        max_output_tokens=250
    )

    """question_model = VertexAI(
    model="gemini-1.5-flash",
    project="llama-ckcb",
    location="us-central1",
    temperature=0.5,  # Lowered temperature for more focused outputs
    top_p=0.7,        # Slightly reduced for more precise output
    #top_k=40,         # Slightly reduced for more precise output
    max_output_tokens=250  # Reduced to limit the length of the output
    )"""

    return question_model

def question_chain(question_model):
    question_generator_template = """
    chat history: {chat_history}
    Question: {question}
    Review the provided chat history and the follow-up question.
    If the follow-up question builds upon the chat history,
    reformulate it into a clear, standalone question that
    incorporates necessary context. If the follow-up question
    is already clear and self-contained, leave it unchanged.
    Your goal is to ensure the question is understandable without
    needing to refer back to the chat history.
    """

    question_gen_prompt = ChatPromptTemplate.from_template(
        question_generator_template)

    question_gen_chain = (
        {"chat_history": RunnablePassthrough(), "question": RunnablePassthrough()}
        | question_gen_prompt
        | question_model
        | StrOutputParser()
    )

    return question_gen_chain

def retriever(pages, embed_model):
    DB = Chroma.from_documents(pages, embed_model)
    retriever = DB.as_retriever(search_type="similarity", search_kwargs={"k": 3})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def response_chain(retriever, LLM_model):
    prompt_template = """Answer the question as precise as possible using the provided context. 
        If the answer is not contained in the context, say "answer not available in context" \n\n
        Context: \n {context}?\n
        question: {question}\n
        Answer:
        """

    model_prompt = ChatPromptTemplate.from_template(prompt_template)

    model_response_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | model_prompt
        | LLM_model
        | StrOutputParser()
    )

    return model_response_chain
