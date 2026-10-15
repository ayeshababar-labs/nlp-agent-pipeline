import requests

BASE_URL = "http://localhost:5001"

# Tool functions that call the Flask endpoints
def get_sentiment(text):
    response = requests.post(f"{BASE_URL}/predict", json={"text": text})
    return response.json()

def extract_keywords(text):
    response = requests.post(f"{BASE_URL}/keywords", json={"text": text})
    return response.json()

def classify_topic(text):
    response = requests.post(f"{BASE_URL}/classify", json={"text": text})
    return response.json()


TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_sentiment",
            "description": "Analyzes the sentiment of text, returns positive or negative with a confidence score.",
            "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "The text to analyze"}
            },
            "required": ["text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "extract_keywords",
            "description": "Extracts the main keywords and phrases from text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The text to extract keywords from"}
                },
                "required": ["text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "classify_topic",
            "description": "Classifies the topic of the given text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Classifies text into a topic category such as technology, sports, politics, business, entertainment, health or science"}
                },
                "required": ["text"]
            }
        }
    }
]

# Map tool names to functions
TOOL_MAP = {
    "get_sentiment": get_sentiment,
    "extract_keywords": extract_keywords,
    "classify_topic": classify_topic
}