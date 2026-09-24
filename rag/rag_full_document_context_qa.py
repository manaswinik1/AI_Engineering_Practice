from dotenv import load_dotenv
import os
import anthropic


load_dotenv() 

api_key = os.environ.get("ANTHROPIC_API_KEY")
print("Key loaded:", api_key is not None)


with open('../data/economic.txt','r') as file:
          file_content = file.read()
client=anthropic.Anthropic(api_key=api_key)
user_quetsion = input("enter the question about the file: ")
message=client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1234,
        messages=[{"role":"user","content": f"{file_content},{user_quetsion}"}]
    )

print(message.content[0].text)


