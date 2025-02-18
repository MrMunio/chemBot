from pydantic import BaseModel
from typing import List, Dict
from uuid import uuid4

class Message(BaseModel):
    type: str  # "HumanMessage" or "AIMessage"
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    thread_id: str = str(uuid4())

class ChatResponse(BaseModel):
    messages: List[Message]
    thread_id: str