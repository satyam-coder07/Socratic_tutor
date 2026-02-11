import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq
import os

st.set_page_config(page_title="Socratic Study Buddy", layout="centered")
st.title("Socratic Study Buddy (Groq Edition)")


ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2", 
    device="cpu"
)
client = chromadb.PersistentClient(path="./socratic_db")
collection = client.get_or_create_collection(name="study_material", embedding_function=ef)

with st.sidebar:
    st.header("1. Configuration")
    api_key = st.text_input("Enter Groq API Key:", type="password")
    
    st.header("2. Upload Material")
    uploaded_file = st.file_uploader("Upload a study text file (.txt)", type="txt")
    
    if st.button("Index Material"):
        if uploaded_file and api_key:
            content = uploaded_file.read().decode("utf-8")
            
            chunks = [c.strip() for c in content.split("\n\n") if len(c.strip()) > 10]
            
            try:
                client.delete_collection("study_material")
                collection = client.get_or_create_collection(name="study_material", embedding_function=ef)
            except:
                pass

            collection.add(
                documents=chunks,
                metadatas=[{"source": uploaded_file.name}] * len(chunks),
                ids=[f"{uploaded_file.name}_{i}" for i in range(len(chunks))]
            )
            st.success(f"Successfully indexed {len(chunks)} sections!")
        else:
            st.warning("Please provide both API Key and a File.")

st.info("I will guide you to the answer using your notes, but I won't give it to you!")

user_query = st.text_input("What would you like to learn?")

if st.button("Ask My Tutor"):
    if not api_key:
        st.error("Please enter your Groq API Key.")
    elif collection.count() == 0:
        st.error("Please upload and index a file first.")
    elif user_query:
        results = collection.query(query_texts=[user_query], n_results=1)
        
        if results["documents"][0]:
            retrieved_context = results["documents"][0][0]
            
            try:
                groq_client = Groq(api_key=api_key)
                
                system_prompt = f"""
                You are a Socratic Tutor. 
                STRICT RULE: Only use the information provided in the CONTEXT below. 
                If the answer isn't in the context, say "I don't have that information in my notes."

                Your goal: Help the student find the answer themselves by asking a leading question.
                DO NOT give the answer. DO NOT use formulas. 

                CONTEXT:
                {retrieved_context}
                """
                
                completion = groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_query}
                    ],
                    temperature=0.4,
                )
                
                st.subheader("Tutor's Guidance:")
                st.write(completion.choices[0].message.content)
                
                with st.expander("Internal Reasoning (RAG Context)"):
                    st.write(f"**Retrieved Fact:** {retrieved_context}")
            except Exception as e:
                st.error(f"Error: {e}")