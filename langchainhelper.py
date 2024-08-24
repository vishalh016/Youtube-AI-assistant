from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
import os
import openai

# load_dotenv()

apiKey = "YOUR API KEY"
openai.api_key = apiKey

os.environ['OPENAI_API_KEY'] = apiKey

embeddings = OpenAIEmbeddings()

video_url ="https://youtu.be/lG7Uxts9SXs?si=NcRWzHGJWEArP8Rx"

def create_vectorDB_from_youtube_url(video_url: str)-> FAISS:
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    docs = text_splitter.split_documents(transcript)

    db = FAISS.from_documents(docs,embeddings)
    return db

# print(create_vectorDB_from_youtube_url(video_url))

def get_response_from_query(db,query,k=4):
    docs = db.similarity_search(query,k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    llm = OpenAI(model="text-davinci-003")
    # llm = OpenAI(temperature=0.8)

    prompt = PromptTemplate(
        input_variables = ["question", "docs"],
        template=f"""
        You are a helpful Youtube assistant that can answer questions about 
        videos based on the video's transcript.
        
        Answer the following question : {{question}}
        By searching the following video transcript : {{docs}}
        
        Only use the factual information from the transcript to answer the question,
        If you feel like you don't have enough information to answer the question, just say "I don't know".
        
        Your answers should be detailed.
        """

    )

    chain = LLMChain(llm=llm, prompt=prompt)
    response =chain.run(question = query, docs = docs_page_content)
    response = response.replace("\n","")
    return response,docs_page_content
