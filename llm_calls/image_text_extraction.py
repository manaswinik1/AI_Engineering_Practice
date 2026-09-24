from dotenv import load_dotenv
import os, anthropic
import time
import base64
load_dotenv()

client=anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

def encode_image(path):
    with open(path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")

message=client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=1234,
    messages=[{"role":"user","content":[
        {"type":"image","source":{
            "type":"base64",
            "media_type":"image/jpeg",
            "data":encode_image("../assets/quote.jpeg")
        }},
        {"type":"text", "text": "Extract the text from the picture. in one line"}
    ]}]
)
print(message.content[0].text)