# Chatbot code for "QA Chatbot over GitHub Repo"
import os, sys, openai
import streamlit as st
from streamlit_chat import message
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake

sys.path.append('..')
from keys import OPENAI_API_KEY, ACTIVELOOP_TOKEN
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["ACTIVELOOP_TOKEN"] = ACTIVELOOP_TOKEN

my_activeloop_org_id = "iryna"
my_activeloop_dataset_name = "chat_over_github"
active_loop_dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"


def load_embeddings_and_database(active_loop_dataset_path):
    """Load Embeddings and DeepLake database"""
    embeddings = OpenAIEmbeddings()
    db = DeepLake(
        dataset_path=active_loop_dataset_path,
        read_only=True,
        embedding_function=embeddings
    )
    return db


def get_text():
    """Create a Streamlit input field and return the user's input."""
    input_text = st.text_input("", key="input")
    return input_text


def search_db(db, query):
    """Search for a response to the query in the DeepLake database."""
    # Create a retriever from the DeepLake instance
    retriever = db.as_retriever()
    # Set the search parameters for the retriever
    retriever.search_kwargs["distance_metric"] = "cos"
    retriever.search_kwargs["fetch_k"] = 100
    retriever.search_kwargs["maximal_marginal_relevance"] = True
    retriever.search_kwargs["k"] = 3
    # Create a ChatOpenAI model instance
    model = ChatOpenAI(model="gpt-3.5-turbo")
    # Create a RetrievalQA instance from the model and retriever
    qa = RetrievalQA.from_llm(model, retriever=retriever)
    # Return the result of the query
    return qa.run(query)

        
# Main function to run the app
def main():
    # Set the title for the Streamlit app
    st.title(f"Chat with GitHub Repository")
    
    # Load embeddings and the DeepLake database
    db = load_embeddings_and_database(active_loop_dataset_path)
    
    # Initialize session state for generated responses and past messages
    if "generated" not in st.session_state:
        st.session_state["generated"] = ["I am ready to help you ser"]
    if "past" not in st.session_state:
        st.session_state["past"] = ["Hey there!"]
    
    # Get the user's input from the text input field
    user_input = get_text()
    
    # Search the databse and add the responses to state
    if user_input:
        output = search_db(db, user_input) #qa.run(user_input)
        st.session_state.past.append(user_input)
        st.session_state.generated.append(output)
    
    # Create the conversational UI using the previous states
    if st.session_state["generated"]:
        for i in range(len(st.session_state["generated"])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
            message(st.session_state["generated"][i], key=str(i))

        
# Run the main function when the script is executed
if __name__ == "__main__":
    main()