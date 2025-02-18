from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from .models.chat import ChatRequest, ChatResponse, Message
from .agent.agent import get_agent
from .agent.config import get_llm
from langchain_core.messages import HumanMessage, AIMessage

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def convert_to_langchain_messages(messages: List[Message]):
    """Convert API messages to LangChain messages."""
    message_map = {
        "HumanMessage": HumanMessage,
        "AIMessage": AIMessage
    }
    
    return [message_map[msg.type](content=msg.content) for msg in messages]

def convert_from_langchain_messages(messages: List[HumanMessage | AIMessage]):
    """Convert LangChain messages to API messages."""
    return [
        Message(
            type=msg.__class__.__name__,
            content=msg.content
        )
        for msg in messages
    ]

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        # Convert incoming messages to LangChain format
        langchain_messages = convert_to_langchain_messages(request.messages)
        
        # Get the configured LLM and agent
        llm = get_llm()
        agent = get_agent(llm)
        
        # Process the chat request
        config = {"configurable": {"thread_id": request.thread_id}}
        response = agent.invoke(
            {"messages": langchain_messages},
            config
        )
        
        # Convert response messages to API format
        api_messages = convert_from_langchain_messages(response["messages"])
        
        return ChatResponse(
            messages=api_messages,
            thread_id=request.thread_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reset")
async def reset_chat():
    return {"status": "success"}