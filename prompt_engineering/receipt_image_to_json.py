from dotenv import load_dotenv
import os, anthropic
import base64

load_dotenv()
client=anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
def encode_image(path):
    with open(path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")

system_prompt='''
You are a reciept parser. Exract the items form the image and retuen them in a valid JSON format. No explaination. No Markdown. Just JSON
 
Required_structure:
{
"store": "Store_name",
"date":"data if visible",
"items":[{"name": "item_name", "price": 0.00,"quantity":1}],
"total":0.00
}

'''


message=client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=1234,
    messages=[{"role":"user","content":[
        {"type": "image", "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": encode_image("../assets/grocery_list.png")
        }},
        {"type":"text", "text": system_prompt}
    ]}]
)
raw_data=message.content[0].text
if raw_data.startswith("`"):
    raw_data=raw_data.strip("`")
    if raw_data.startswith("json"):
        raw_data=raw_data.strip("json")

import json
try:
    data=json.loads(raw_data)
    for item in data["items"]:
        print(f"{item['name']}: {item['price']}, {item['quantity']}")
except json.JSONDecodeError:
    print("Non-Json Response")

