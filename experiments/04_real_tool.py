import requests
import json
from groq import Groq

client = Groq()

# This time the tool actually calls Flask API
def get_sentiment(text):
    response = requests.post(
        "http://localhost:5001/predict",
        json={"text": text}
    )
    return response.json()

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_sentiment",
            "description": "Analyzes the sentiment of text using an ML model",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text to analyze"
                    }
                },
                "required": ["text"]
            }
        }
    }
]

messages = [{"role": "user", "content": "Is this review good or bad?: The battery died after 2 hours"}]

response = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=messages,
    tools=tools,
    tool_choice="auto"
)

message = response.choices[0].message

if message.tool_calls:
    tool_call = message.tool_calls[0]
    args = json.loads(tool_call.function.arguments)
    
    # This now hits your real Flask endpoint
    result = get_sentiment(args["text"])
    print(f"Tool result from Flask: {result}")  # so you can see what came back
    
    messages.append(message)
    messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": json.dumps(result)
    })
    
    final = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=messages
    )
    print(f"Agent response: {final.choices[0].message.content}")
else:
    print(message.content)