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

            collection.add(
                documents=[content],
                metadatas=[{"source": uploaded_file.name}],
                ids=[uploaded_file.name]
            )
            st.success("Material indexed successfully!")
        else:
            st.warning("Please provide both API Key and a File.")


st.info("Ask me anything about your uploaded material. I will guide you to the answer, but I won't give it to you!")

user_query = st.text_input("What would you like to learn?")

if st.button("Ask My Tutor"):
    if not api_key:
        st.error("Please enter your Groq API Key in the sidebar.")
    elif collection.count() == 0:
        st.error("Please upload and index a file first.")
    elif user_query:
        results = collection.query(query_texts=[user_query], n_results=1)
        retrieved_context = results["documents"][0][0]
        
        try:
            groq_client = Groq(api_key=api_key)
            
            system_prompt = f"""
            You are a Socratic Tutor. Use the provided context to guide the student.
            
            RULES:
            1. NEVER provide direct answers, formulas, or definitions.
            2. ALWAYS answer with a leading question that points to a detail in the context.
            3. If the user is in a rush or demands the answer, politely refuse and ask a simpler question.
            
            CONTEXT:
            {retrieved_context}
            """
            
            completion = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query}
                ],
                temperature=0.5,
                max_tokens=1024,
            )
            
            tutor_response = completion.choices[0].message.content
            
            st.subheader("Tutor's Guidance:")
            st.write(tutor_response)
            
            with st.expander("Internal Reasoning (RAG Context)"):
                st.write(f"**Retrieved Fact:** {retrieved_context}")

        except Exception as e:
            st.error(f"An error occurred: {e}")