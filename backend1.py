from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_cohere import CohereEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain import hub
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
import re
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain import hub
from groq import Groq
def response_query(user_query):
    loader1=PyPDFLoader("disaster_management_in_india.pdf")
    docs1=loader1.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs1)
    cohere_api_key = "leTByPB6J9FNbFIup99z08dhPaFwiquAlRqScvJv"
    embeddings = CohereEmbeddings(model="embed-english-light-v3.0",cohere_api_key=cohere_api_key)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    groq_api_key="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"
    chat = ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name="llama3-70b-8192")
    template = """Use the following pieces of context to answer the question at the end.
            Say that you don't know when asked a question you don't know, donot make up an answer. Be precise and concise in your answer.
            dont say based on the given context. you are a disaster management helper
            you need to answer the user queries and help them and comfort them
            if the situation demands
            {context}

            Question: {input}

            Helpful Answer:"""
    prompt = PromptTemplate.from_template(template)
    from langchain.chains.combine_documents import create_stuff_documents_chain
    document_chain=create_stuff_documents_chain(chat,prompt)
    retriever=vectorstore.as_retriever()
    from langchain.chains import create_retrieval_chain
    retrieval_chain=create_retrieval_chain(retriever,document_chain)
    #user_query="Who is the HOD of CS department"

    res=retrieval_chain.invoke({"input":user_query})

    return res['answer']


def response_selector(user_query):
    client = Groq(
             api_key='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',
)
    chat_completion = client.chat.completions.create(
        messages=[
        {
            "role": "user",
            "content": f'''classify the following message '{user_query}' into category 0 or 1 or 2 or 3 or 4
                          if the messgae  is about current weather then category 0
                          if the message is  about situations like floods,wildfire,etc then category 1 
                          if the message is about daily news or local news about a place like kollam then category 2
                          if the message is about flood prediction for the year then category 3
                          if the message is about nearest relief camps or something like that then category 4
                          if the message is about  anything else category 5
                          return the category number only''',
        }
         ],
        model="llama3-8b-8192",
     )
    response_message = chat_completion.choices[0].message.content
    return response_message    

def palce_finder(user_query):
    matches = re.findall(r"\b[A-Z][a-z]+\s[A-Z][a-z]+\b|\b[A-Z][a-z]+\b",user_query)
    return matches    


import os
def response_tavily(user_query):
    os.environ["TAVILY_API_KEY"] = "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"
    from langchain_community.retrievers import TavilySearchAPIRetriever

    retriever = TavilySearchAPIRetriever(k=3)

    groq_api_key="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"
    chat = ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name="llama3-70b-8192")
    prompt = ChatPromptTemplate.from_template(
        """Answer the question based only on the context provided.
            dont say according to context. reply like a human
    Context: {context}

    Question: {question}"""
    )
    chain = (
        RunnablePassthrough.assign(context=(lambda x: x["question"]) | retriever)
        | prompt
        | chat
        | StrOutputParser()
    )
    return chain.invoke({"question": user_query})


def response_from_news(user_query):
    urls = [
    "https://www.manoramaonline.com/district-news/thiruvananthapuram.html",
    "https://www.manoramaonline.com/district-news/kollam.html",
    "https://www.manoramaonline.com/district-news/pathanamthitta.html",
    "https://www.manoramaonline.com/district-news/alappuzha.html",
    "https://www.manoramaonline.com/district-news/kottayam.html",
    "https://www.manoramaonline.com/district-news/idukki.html",
    "https://www.manoramaonline.com/district-news/ernakulam.html",
    "https://www.manoramaonline.com/district-news/thrissur.html",
    "https://www.manoramaonline.com/district-news/palakkad.html",
    "https://www.manoramaonline.com/district-news/kozhikode.html",
    "https://www.manoramaonline.com/district-news/wayanad.html",
    "https://www.manoramaonline.com/district-news/malappuram.html",
    "https://www.manoramaonline.com/district-news/kasargod.html",
    "https://www.manoramaonline.com/district-news/kannur.html"
]
    loader1=WebBaseLoader(urls)
    docs1=loader1.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs1)
    cohere_api_key = "leTByPB6J9FNbFIup99z08dhPaFwiquAlRqScvJv"
    embeddings = CohereEmbeddings(model="embed-english-light-v3.0",cohere_api_key=cohere_api_key)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    groq_api_key="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"
    chat = ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name="llama3-70b-8192")
    template = """Use the following pieces of context to answer the question at the end.
            Say that you don't know when asked a question you don't know, donot make up an answer. Be precise and concise in your answer.
            dont say based on the given context. you are a chatbot and u have all the data about all the districts in kerela
            ans all queries from user especiallly about disasters and weather and dont say here are some news. just answer your query like a human
            {context}

            Question: {input}

            Helpful Answer:"""
    prompt = PromptTemplate.from_template(template)
    from langchain.chains.combine_documents import create_stuff_documents_chain
    document_chain=create_stuff_documents_chain(chat,prompt)
    retriever=vectorstore.as_retriever()
    from langchain.chains import create_retrieval_chain
    retrieval_chain=create_retrieval_chain(retriever,document_chain)
    #user_query="Who is the HOD of CS department"

    res=retrieval_chain.invoke({"input":user_query})

    return res['answer']

from streamlit_folium import st_folium
import folium
import streamlit as st
def plot_map(user_query):
    print("user quyery is ")
    print(user_query)
    start=-1
    end=-1
    for i in range(len(user_query)):
        if i=="(":
            start=i
        elif i==")":
            end=i
    cordinates=user_query[start:end+1]
    print(cordinates)
    location_str=cordinates.strip("()")
    #location=[float(cord) for cord in cordinates.split(",")]
    location=[9.0833,76.6113]
    map=folium.Map(location,zoom_start=9)
    folium.Marker(location)
    st.header("nearest camps")
    st_folium(map,width=700)