from dotenv import load_dotenv
import os
import anthropic

load_dotenv()
system_prompt="""
You are an epxence classifier. 

Classify expences into 4 categories, HOUSING, FOOD, TRANSPORT, HEALTH, ENTERTAINMENT, LIFESTYLE, SAVINGS, INVESTMENTS, OTHER

Example: 
Rent: $875 -> HOUSING
Hair cut: $100 -> LIFESTYLE
RICE: $150 ->FOOD
Roth IRA: $1000 ->INVESTMENTS
Emergecy fund: $300 ->SAVINGS
UBER: $80 -> TRANSPORTATION
SUPPLEMENTS: $30 ->HEALTH

"""
api_key=os.environ.get("ANTHROPIC_API_KEY")
print("Key loaded:", api_key is not None)
client=anthropic.Anthropic(api_key=api_key)

def get_response(conversation):
    message=client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1234,
        system=system_prompt,
        messages=conversation
    )
    return message.content[0].text

def main():
    conversation= []
    Expenses=[]
    while True:
        user_question = input("enter the question: ")
        conversation.append({"role":"user","content":user_question})
        reply=get_response(conversation)
        print(reply)
        conversation.append({"role":"assistant","content":reply})
        

        if user_question.lower() =='quit':
            break
main()

