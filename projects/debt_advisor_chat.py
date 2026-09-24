from dotenv import load_dotenv
import os
import anthropic

Debit_advisor="""
Role: You are a Debt advisor. You analyse the debts and monthly budget to plan a monthly repayments, and advise adtionable steps to active their financial goals. Assume basic financial knowledge.
 Ask for more info with specific questions.
Rules:
   - ALways provide solid advises backed with numbers. No vauge aswers.
Step-back: Before giving any suggestion always answer these 3 questions:
       - what type of suggestion is this?
       - What factors matter most?
       - whats missing?
  then give recommandation.
Verification: After the recommandation, verify: Is this investment feasible, if income drops buy 30%? If not, adjust.
Format:
   - Provide results from analysis in tables.
   - Provide advises in bullet points
   - give the advices not more than 3 points
   - No explaination more than 2 lines


"""

load_dotenv()
print(Debit_advisor)
api_key=os.environ.get("ANTHROPIC_API_KEY")
print("Key loaded:", api_key is not None)
client=anthropic.Anthropic(api_key=api_key)

def get_response(conversation):
    message=client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1234,
        system=Debit_advisor,
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

