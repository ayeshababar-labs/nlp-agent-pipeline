import json
from groq import Groq
from app.tools import TOOL_DEFINITIONS, TOOL_MAP

client = Groq()

def run_agent(question: str, text: str) -> dict:
    messages = [
        {
            "role": "system",
            "content": "You are a text analysis assistant. Use the available tools to analyze text and answer questions accurately."
        },
        {
            "role": "user",
            "content": f"{question}\n\nText: {text}"
        }
    ]

    tools_called = []

    # Agent loop — keep going until LLM stops calling tools
    while True:
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=messages,
            tools=TOOL_DEFINITIONS,
            tool_choice="auto"
        )

        message = response.choices[0].message

        # No more tool calls — LLM has final answer
        if not message.tool_calls:
            return {
                "response": message.content,
                "tools_called": tools_called
            }

        # Process each tool call
        messages.append(message)

        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)

            # Call the actual function
            tool_fn = TOOL_MAP[tool_name]
            result = tool_fn(args["text"])
            tools_called.append(tool_name)

            # Send result back to LLM
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result)
            })