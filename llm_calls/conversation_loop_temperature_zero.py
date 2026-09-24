from dotenv import load_dotenv
import os
import anthropic
load_dotenv()
api_key=os.environ.get("ANTHROPIC_API_KEY")
client=anthropic.Anthropic(api_key=api_key)
def get_response(conversation):
    message= client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1000,
        temperature=0.0,
        messages=conversation
    )
    return message.content[0].text

def main():
    conversation=[]
    while True:
        user_message=input("enter you question: ")
        conversation.append({"role":"user","content":user_message})
        reply=get_response(conversation)
        print(reply)
        conversation.append({"role":"assistant","content":reply})
        if user_message.lower()=="quit":
            break

main()





