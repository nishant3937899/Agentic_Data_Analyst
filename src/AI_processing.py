from google import genai
from google.genai import types
from .ai_tools import gemini_tools,execute_tool
import os

def load_api_key(api_ky=None):
    if api_ky:
        client = genai.Client(api_key=api_ky)
        return client
    api_key = os.environ.get("GEMINI_API_KEY")
    return genai.Client(api_key=api_key)

#gemini here
def ask_agent(user_question,client):


    charts = []

    conversation = [
        types.Content(
            role="user",
            parts=[types.Part(text=user_question)]
        )
    ]

    conversation = [
        types.Content(
            role="user",
            parts=[
                types.Part(
                    text=user_question
                )
            ]
        )
    ]

    max_itr=15
    for itr in range(max_itr):

        print("\n🤖 Gemini thinking...")

        response = client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=conversation,
            config=types.GenerateContentConfig(
                tools=gemini_tools,
                temperature=0
            )
        )

        candidate = response.candidates[0]

        parts = candidate.content.parts

        #print("\n DEBUG")
        #print("Finish reason:", candidate.finish_reason)
        #print("Parts:", parts)
        #print("Response:", response)

        function_call = None

        for part in parts:

            if part.function_call:
                function_call = part.function_call
                break

        
        # Gemini wants to call a tool
        

        if function_call:

            tool_name = function_call.name

            tool_args = dict(
                function_call.args
            )

            print(
                f"\n🔧 Gemini selected: {tool_name}"
            )

            print(
                f"📦 Arguments: {tool_args}"
            )

            # Add Gemini's tool request
            conversation.append(
                candidate.content
            )

            # Execute actual Python function
            tool_result = execute_tool(
                tool_name,
                tool_args
            )

            print(
                "\n📊 Tool result:"
            )

            print(tool_result)


            # collect generated chart
            if tool_name == "make_chart":
                if tool_result.get("status") == "success":
                    charts.append(tool_result)


            # Send result back to Gemini
            conversation.append(
                types.Content(
                    role="tool",
                    parts=[
                        types.Part(
                            function_response=
                            types.FunctionResponse(
                                name=tool_name,
                                response=tool_result
                            )
                        )
                    ]
                )
            )

            continue

        
        # Gemini produced final answer
        

        final_text = ""

        for part in parts:

            if part.text:
                final_text += part.text
        print(charts)
        print(final_text)
        return {
            "answer": final_text,
            "charts": charts
        }

    