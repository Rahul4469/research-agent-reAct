import json
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

client = Anthropic()

# Define tools that the agent can use
tools = [
    {
        "name": "calculator",
        "description": "A simple calculator that can perform basic arithmetic operations. Use this when you need to perform mathematical calculations.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate (e.g., '2 + 2', '10 * 5', '100 / 4')"
                }
            },
            "required": ["expression"]
        }
    },
    {
        "name": "get_weather",
        "description": "Get the current weather for a given location. Use this when the user asks about weather conditions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and country (e.g., 'London, UK' or 'New York, USA')"
                }
            },
            "required": ["location"]
        }
    }
]


def execute_tool(tool_name: str, tool_input: dict) -> str:
    """Execute a tool and return the result."""
    if tool_name == "calculator":
        try:
            # Only allow safe mathematical operations
            allowed_chars = set("0123456789+-*/.() ")
            expression = tool_input["expression"]
            if all(c in allowed_chars for c in expression):
                result = eval(expression)
                return json.dumps({"result": result})
            else:
                return json.dumps({"error": "Invalid characters in expression"})
        except Exception as e:
            return json.dumps({"error": str(e)})

    elif tool_name == "get_weather":
        # Mock weather data - replace with real API call
        location = tool_input["location"]
        return json.dumps({
            "location": location,
            "temperature": "22°C",
            "condition": "Partly cloudy",
            "humidity": "65%",
            "note": "This is mock data. Integrate a real weather API for actual data."
        })

    return json.dumps({"error": f"Unknown tool: {tool_name}"})


def run_agent(user_message: str, conversation_history: list) -> tuple[str, list]:
    """
    Run the agent with a user message and return the response.
    Handles tool calls in a loop until the agent produces a final response.
    """
    conversation_history.append({
        "role": "user",
        "content": user_message
    })

    while True:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            system="You are a helpful AI assistant. You have access to tools that you can use to help answer questions. Use tools when appropriate to provide accurate information.",
            tools=tools,
            messages=conversation_history
        )

        # Check if we need to process tool calls
        if response.stop_reason == "tool_use":
            # Add assistant's response to history
            conversation_history.append({
                "role": "assistant",
                "content": response.content
            })

            # Process each tool call
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    tool_result = execute_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": tool_result
                    })

            # Add tool results to history
            conversation_history.append({
                "role": "user",
                "content": tool_results
            })
        else:
            # No more tool calls, extract final text response
            conversation_history.append({
                "role": "assistant",
                "content": response.content
            })

            # Extract text from response
            final_response = ""
            for block in response.content:
                if hasattr(block, "text"):
                    final_response += block.text

            return final_response, conversation_history


def main():
    """Main function to run the interactive agent."""
    print("AI Agent initialized. Type 'quit' to exit.")
    print("This agent can help with calculations and weather queries.\n")

    conversation_history = []

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break

        if not user_input:
            continue

        try:
            response, conversation_history = run_agent(user_input, conversation_history)
            print(f"\nAssistant: {response}\n")
        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()
