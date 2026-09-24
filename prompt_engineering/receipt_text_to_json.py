from dotenv import load_dotenv
import os, anthropic

load_dotenv()

client=anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

system_prompt = '''
 You are a reciept parser
Return only in valid JSON. No other text. No explainations. No markdown. Just the json.


Requrired json:
{
"store": "Store name or Unknown",
"date": "date or null",
"items":[{"name":"item name","price":0.0,"quantity":1}]
}

'''

message=client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=1234,
    messages=[{"role":"user","content":"Parse: Chocolates $8.96, Wine $87.99, protien shake $9.99"}]
)

import json
try:
    data = json.loads(message.content[0].text)
    for item in data["items"]:
        print(f"{item['name']}: {item['price']}")
except json.JSONDecodeError:
    print("Non-JSON Response - check system prompt")