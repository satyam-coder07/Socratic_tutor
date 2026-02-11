import chromadb
from chromadb.utils import embedding_functions

ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2", 
    device="cpu"
)

client = chromadb.PersistentClient(path="./socratic_db")
collection = client.get_or_create_collection(name="technical_docs", embedding_function=ef)

def seed_data():
    if collection.count() == 0:
        docs = [
            "Photosynthesis is the process where plants use sunlight, water, and CO2 to create oxygen and energy in the form of sugar.",
            "The Pythagorean theorem states that in a right-angled triangle, the square of the hypotenuse is equal to the sum of the squares of the other two sides.",
            "Newton's Second Law of Motion defines force as mass times acceleration (F=ma)."
        ]
        ids = ["bio_01", "math_01", "phys_01"]
        collection.add(documents=docs, ids=ids)
        print("Knowledge base initialized.")

def get_socratic_response(user_input, retrieved_fact):
    pressure_keywords = ["rush", "answer", "tell me", "hurry", "fast"]
    if any(word in user_input.lower() for word in pressure_keywords):
        return "I understand you are in a hurry, but true learning takes time. Look at the data again: what is the relationship between the variables mentioned?"

    if "photo" in user_input.lower():
        return "Think about what a plant needs from the sky and the ground. How do you think it combines those to make its own food?"
    
    if "pythagorean" in user_input.lower() or "triangle" in user_input.lower():
        return "If you have a right triangle and you know the two shorter sides, what happens if you square them and add them together?"

    if "force" in user_input.lower() or "newton" in user_input.lower():
        return "Consider an object's mass. If you want it to speed up (accelerate), what do you need to apply to it?"

    return "That's a deep question. Based on the text, what is the most important component involved in this process?"

def run_tutor():
    seed_data()
    print("\n" + "="*40)
    print("WELCOME TO THE SOCRATIC STUDY BUDDY")
    print("="*40)
    print("I will help you find the answer, but I will never give it to you.")
    
    while True:
        query = input("\nStudent: ")
        
        if query.lower() in ["exit", "quit", "bye"]:
            print("Keep studying! Goodbye.")
            break
            

        results = collection.query(query_texts=[query], n_results=1)
        fact = results["documents"][0][0]
        
        print(f"   [DEBUG LOG: System retrieved factual context: {fact}]")
        
        response = get_socratic_response(query, fact)
        print(f"Tutor: {response}")

if __name__ == "__main__":
    run_tutor()