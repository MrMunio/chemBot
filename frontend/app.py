import streamlit as st
import requests
from uuid import uuid4

# Constants
API_URL = "http://localhost:8000"
API_URL = "http://host.docker.internal:8000" # for docker deployment

def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid4())

def clear_chat():
    st.session_state.messages = []
    st.session_state.thread_id = str(uuid4())
    requests.post(f"{API_URL}/reset")

def convert_message_for_api(message):
    """Convert a message to the format expected by the API."""
    return {
        "type": "HumanMessage" if message["role"] == "human" else "AIMessage",
        "content": message["content"]
    }

def send_message(message: str):
    # Add user message to chat history
    st.session_state.messages.append({"role": "human", "content": message})
    
    # Prepare messages for API
    api_messages = [
        convert_message_for_api(msg) 
        for msg in st.session_state.messages
    ]
    
    # Send request to backend
    try:
        response = requests.post(
            f"{API_URL}/chat",
            json={
                "messages": api_messages,
                "thread_id": st.session_state.thread_id
            }
        )
        
        if response.status_code == 200:
            response_data = response.json()
            # Get only the last AI message from the response
            all_messages = response_data["messages"]
            if all_messages:
                last_ai_message = next(
                    (msg for msg in reversed(all_messages) 
                     if msg["type"] == "AIMessage"), 
                    None
                )
                if last_ai_message:
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": last_ai_message["content"]
                    })
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
            
    except Exception as e:
        st.error(f"Error communicating with backend: {str(e)}")

def main():
    st.title("ChemBot")
    
    # Initialize session state
    init_session_state()
    
    # Add clear chat button
    st.sidebar.button("Clear Chat", on_click=clear_chat)
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if message := st.chat_input("Ask something about chemistry..."):
        send_message(message)
        st.rerun()

if __name__ == "__main__":
    main()