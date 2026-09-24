from dotenv import load_dotenv
import os, anthropic
import time

load_dotenv()

client=anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

def get_streaming_response(conversation):

    full_response=""
    with client.messages.stream(model="claude-haiku-4-5-20251001", max_tokens=1234, messages=conversation) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
            time.sleep(0.01)
            full_response+=text
    return full_response

def main():
    conversation=[]
    while True:
        user_message=input("user message: ")
        conversation.append({"role":"user", "content": user_message})
        reply=get_streaming_response(conversation)
        conversation.append({"role":"assistant", "content":reply})
        if user_message.lower()=="quit":
            break

main()

 