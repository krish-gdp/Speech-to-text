from openai import OpenAI
from dotenv import load_dotenv
from langchain_core.messages import AIMessage,HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import streamlit as st
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode
from PyPDF2 import PdfReader
import soundfile as sf
import threading
import time



client = OpenAI(api_key='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')

audio_file= open("C:\Projects\Genai\Speech_to_text\Recordings\Recording.m4a", "rb")
transcription = client.audio.transcriptions.create(
  model="whisper-1", 
  file=audio_file
)
print(transcription.text)



def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(raw_text):
    text_Splitter = CharacterTextSplitter(separator="\n",chunk_size = 1000,chunk_overlap= 200,length_function = len)
    chunks = text_Splitter.split_text(raw_text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name = "hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts = text_chunks,embedding = embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key = 'chat_history',return_messages =True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm = llm,
                                                               retriever = vectorstore.as_retriever(),
                                                                memory = memory )
    return conversation_chain
def audio_to_text(audio_file):
    transcription = client.audio.transcriptions.create(
    model="whisper-1", 
    file=audio_file
    )
    return transcription.text

def generate_questions(document_content):
    question = lc.generate(
        prompt="Generate questions from the given document:\n{}".format(
             document_content
        ),
        max_tokens=50,
        num_outputs=1,
        stop_sequences=["\n"],
    )
    return question

def get_response(user_query: str, chat_history: list):

  template = """
    You are a interviewer taking the interview at a company. You are interacting with a potential employee and replying to the answer given by the employee and correcting him.
    Conversation History: {chat_history}
    Employee_Answer{user_query}
    Interviewr Question : {question}
    Interviewr Response: {response}"""
  
  prompt = ChatPromptTemplate.from_template(template)
  
  llm = ChatOpenAI(model="gpt-4-0125-preview")
#   llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0)
  
  chain = (
    RunnablePassthrough.assign(query=sql_chain).assign(
      schema=lambda _: db.get_table_info(),
      response=lambda vars: db.run(vars["query"]),
    )
    | prompt
    | llm
    | StrOutputParser()
  )
  
  return chain.invoke({
    "question": user_query,
    "chat_history": chat_history,
  })


if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
      AIMessage(content="Hello! I'm the Interview Bot. Lets Introduce about yourself."),
    ]

load_dotenv()
st.set_page_config(page_title="Interview Preparation Bot")
st.title("Interview Bot")
with st.sidebar:
    st.subheader("Interview Bot")
    st.write("Sample bot application to Prepare for the Interviews. Upload the resume and take  practice interviews.")
    pdf_docs = st.file_uploader("Upload PDF File and click on Process",accept_multiple_files=True)
    if st.button("Upload"):
        with st.spinner("uploading....."):
            # get pdf text
            raw_text = get_pdf_text(pdf_docs)
            st.write(raw_text)

            # get the text chunks
            text_chunks = get_text_chunks(raw_text)
            st.write(text_chunks)

            # create vector store
            vectorstore = get_vectorstore(text_chunks)
            # create conversation
            st.session_state.conversation = get_conversation_chain(vectorstore)
for message in st.session_state.chat_history:
    if isinstance(message,AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content)
    elif isinstance(message,HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)
audio_uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "ogg",".m4a"])
if audio_uploaded_file is not None:
    st.audio(audio_uploaded_file, format="audio/m4a")

user_query = audio_to_text(audio_uploaded_file)
print(user_query)
if user_query is not None and user_query.strip() != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    
    with st.chat_message("Human"):
        st.markdown(user_query)
        
    with st.chat_message("AI"):
        # response = get_response(user_query, st.session_state.db, st.session_state.chat_history)
        response = "Dont Know"
        st.markdown(response)
        
    st.session_state.chat_history.append(AIMessage(content=response))