
"""def load_doc(file):
    from langchain_community.document_loaders import PyPDFLoader
    #from pypdf import PdfReader
    print(f"file name:{file}")
    file_data=PyPDFLoader(file).load()
    return file_data"""
    
def load_doc(file):
    from langchain_community.document_loaders import TextLoader
    
    print(f"file name: {file}")
    file_data = TextLoader(file).load()
    return file_data

def chunking(file_data,chunk_size=100):
   # from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter=RecursiveCharacterTextSplitter(chunk_size=100,chunk_overlap=0)
    chunks=splitter.split_documents(file_data)
    return chunks


def embedding_cost(file_data):
    import tiktoken
    enc=tiktoken.encoding_for_model("text-embedding-ada-002")
    total_token=sum([len(enc.encode(page.page_content)) for page in file_data])
    print(f"total tokens -->{total_token}")
    print(f"cost for embeddings-->{(total_token/1000)*0.0004}")
    embedd_cost=(total_token/1000)*0.0004
    return embedd_cost

def embeddings(chunk_data):
    #from langchain.vectorstores import FAISS
    #from langchain.embeddings import HuggingFaceEmbeddings
    from langchain_huggingface import HuggingFaceEmbeddings
    
    embeddings=HuggingFaceEmbeddings(model_name = 
                                     "sentence-transformers/all-mpnet-base-v2")
    #vector_store=FAISS.from_documents(ch,embedding=embeddings)
    from langchain_community.vectorstores import Chroma
    vector_store=Chroma.from_documents(documents=chunk_data,
                        embedding=embeddings,persist_directory="./chromnew.db")
    vector_store.persist()
    return vector_store
    
def get_ans_from_convchain(query,vector_store,api_key,k):
    from langchain.chat_models import ChatOpenAI
    #from langchain.vectorstores import Chroma
    from langchain_community.vectorstores import Chroma
    from langchain.chains import ConversationalRetrievalChain
    from langchain.memory import ConversationBufferMemory
    #from langchain_openai import ChatOpenAI
    from langchain.prompts import SystemMessagePromptTemplate,ChatPromptTemplate,HumanMessagePromptTemplate
           
            
    print(vector_store._collection.count())    
    llm=ChatOpenAI(temperature=0.5,model_name="gpt-4o-mini",api_key=api_key)
    retriever=vector_store.as_retriever(search_kwargs={"k": 20},search_type="similarity")
    memory=ConversationBufferMemory(memory_key="chat_history",
                                   return_messages=True)
        #docs = retriever.get_relevant_documents("how to get sampels")
    system_template=''' you need to checj the
        below context and provide the short answer
        context:{context}'''
    user_template="question: {question}, chat history:{chat_history}"
    messages=[
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(user_template)
        ]
    prmpt=ChatPromptTemplate.from_messages(messages)
        
        
    chain=ConversationalRetrievalChain.from_llm(
            llm=llm,chain_type='stuff',
            retriever=retriever,memory=memory,
            combine_docs_chain_kwargs={"prompt":prmpt},verbose=True)
    ans=chain.invoke(query)
    return ans

    
if(__name__=="__main__"):
    import os  
    import streamlit as st
    st.subheader("T20 match for EXAM preparation??")
    st.text("No worries we are here to help. Just upload your text file and query question we will answer you")
    
    if "vs" not in st.session_state:
        st.session_state.vs=None
    with st.sidebar:
        api_key=st.text_input("open AP key:",type='password')
        if(api_key):
            os.environ["OPENAI_API_KEY"]=api_key
        uploaded_file=st.file_uploader("upload a file",
                                       type=["docx","pdf","txt"])
        chunk_size=st.number_input("chunksize:",
                                           min_value=50,                                                                       max_value=400,value=250)
        k=st.number_input("k NEighbours:",min_value=4,max_value=20,value=4)
        add_data=st.button("add_data")
        
        if(uploaded_file and add_data): # store uploaded file locally
            st.spinner("file uploaded chunking,embedding")
            byte_file=uploaded_file.read()
            file_path=os.path.join("./",uploaded_file.name)
            
            with open(file_path,"wb") as f:
                f.write(byte_file)
            
            file_data=load_doc(file_path) #read uploaded file
            
            chunk_data=chunking(file_data,chunk_size=chunk_size)#chunking  files
            st.write(f"chunks:{chunk_size} chunks:{len(chunk_data)}")
            st.write(f"this is cost of embedding: {embedding_cost(file_data)}")
            vector_store=embeddings(chunk_data)#embedding the chunks
            st.session_state.vs=vector_store
            st.success("file uploaded successfully. Thank you")
            #ans=get_ans_from_convchain("Effect",vector_store,api_key,k)#retriving using coversationretrive chain
            #st.write(ans)
    q=st.text_input("Please enter your question")
    vector_store=st.session_state.vs
    if(q):
     if "vs" in st.session_state:
         ans=get_ans_from_convchain(q,vector_store,api_key,k)#retriving using coversationretrive chain
         st.text_area("LLM answer",value=ans['answer'])
    st.image("noor.jpg")
            
    
