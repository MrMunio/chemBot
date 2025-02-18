import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

def validate_api_key(key_name: str, key_value: str | None) -> None:
    """Validate that required API keys are present."""
    if not key_value:
        raise ValueError(
            f"Missing {key_name} in environment variables. "
            f"Please add it to your .env file or set it as an environment variable."
        )

def get_openai_llm():
    """Initialize OpenAI LLM with proper error handling."""
    api_key = os.getenv("OPENAI_API_KEY")
    validate_api_key("OPENAI_API_KEY", api_key)
    
    return ChatOpenAI(
        api_key=api_key,
        model="gpt-4",
        temperature=0.5
    )

def get_groq_llm():
    """Initialize Groq LLM with proper error handling."""
    api_key = os.getenv("GROQ_API_KEY")
    validate_api_key("GROQ_API_KEY", api_key)
    
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.5,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

def get_llm(provider=None):
    """Get configured LLM based on environment or specified provider."""
    if provider is None:
        provider = os.getenv("DEFAULT_LLM_PROVIDER", "openai").lower()
    
    llm_providers = {
        "openai": get_openai_llm,
        "groq": get_groq_llm,
    }
    
    if provider not in llm_providers:
        raise ValueError(
            f"Unsupported LLM provider: {provider}. "
            f"Available providers: {', '.join(llm_providers.keys())}"
        )
    
    try:
        return llm_providers[provider]()
    except ValueError as e:
        raise ValueError(f"Error initializing {provider} LLM: {str(e)}")