from dotenv import load_dotenv
import os, anthropic

load_dotenv()

client=anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

system_prompt = '''
You are a reciept parser
Return only in valid JSON. No other text. No explainations. No markdown. No table Just the json.


Requrired json:
{
store: Store name or Unknown,
date: date or null,
items: [name:item name,price:0.0,quantity:1],
total: 0.0
}
output needs to be the above required json format
'''

message=client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=1234,
    system=system_prompt,
    messages=[{"role":"user","content":"Parse: Chocolates $8.96, Wine $87.99, protien shake $9.99"}]
)


raw_text= message.content[0].text
if raw_text.startswith("```"):
    raw_text=raw_text.strip("`").strip()
    if raw_text.startswith("json"):
        raw_text=raw_text[4:].strip()
print(raw_text)
import json

try:
    data=json.loads(raw_text)
    for item in data["items"]:
        print(f"{item['name']}: {item['price']}")
except json.JSONDecodeError:
    print("Non-JSON Response - check system prompt")