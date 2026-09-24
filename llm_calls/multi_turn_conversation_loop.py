from dotenv import load_dotenv
import os
import anthropic

load_dotenv()

api_key=os.environ.get("ANTHROPIC_API_KEY")
print("Key loaded:", api_key is not None)

conversation= []

client=anthropic.Anthropic(api_key=api_key)

while True:
    user_question = input("enter the question: ")
    conversation.append({"role":"user","content":user_question})
    message=client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1234,
        messages=conversation
    )
    conversation.append({"role":"assistant","content":(message.content[0].text)})
    print(message.content[0].text)
    if user_question.lower() =='quit':
        break


