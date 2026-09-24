
from dotenv import load_dotenv
import anthropic, os, time, base64, json
load_dotenv()

client=anthropic.Anthropic(api_key=os.environ.get("ANTHROPI_API_KEY"))
system_prompt_json = '''
You are a reciept parser
Return only in valid JSON. No other text. No explainations. No markdown. Just the json.


Requrired json:
{
"store": "Store name or Unknown",
"date": "date or null",
"items":[{"name":"item name","price":0.0,"quantity":1}]
"Total":"total amount"
}

'''
PRICING = {
    "claude-haiku-4-5-20251001": {"input": 1.00, "output": 5.00},
    "claude-sonnet-5":           {"input": 2.00, "output": 10.00},
}

def calculate_cost(model, input_tokens, output_tokens):
    rates = PRICING[model]
    cost = (input_tokens / 1_000_000) * rates["input"] + (output_tokens / 1_000_000) * rates["output"]
    return cost
#3.2
def trim_history(conversation, max_exchnages=10):
    max_messges=max_exchnages*2
    if len(conversation) > max_messges:
        conversation = conversation[-max_messges:]
    return conversation
#3.6
def classify_complexity(question):
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1234,
        messages=[{"role":"user", "content": f"Classify: SIMPLE(factual, simple answer) or COMPLEX( Multi-step, reasoning)\n {question}\n Answer only SIMPLE or COMPLEX"}]
    )
    return response.content[0].text.strip()
def scan(path):
    if path.endswith(".png"):
        media_type="image/png"
    elif path.endswith(".jpeg"):
        media_type="image/jpeg"
    elif path.endswith(".gif"):
            media_type="image/gif"
    elif path.endswith(".webp"):
            media_type="image/webp"
    def encode_image(path):
        with open(path, "rb") as f:
            return base64.standard_b64encode(f.read()).decode("utf-8")
    model="claude-sonnet-5"
    message=client.messages.create(
        model=model,
        max_tokens=1234,
        system=system_prompt_json,
        messages=[{"role":"user","content":[
            {"type":"image","source":{
                "type":"base64",
                "media_type":media_type,
                "data":encode_image(path)
            }},
            {"type":"text", "text": "Extract the contents from the image"}
        ]}]
    )
    if message.content[0].text is None:
        raw_text=message.content[1].text
    else:
        raw_text=message.content[0].text
    if raw_text.startswith("```"):
        raw_text=raw_text.strip("`").strip()
        if raw_text.startswith("json"):
            raw_text=raw_text[4:].strip()
    scan_cost=calculate_cost(model,message.usage.input_tokens,message.usage.output_tokens)
    scan_tokens=(message.usage.input_tokens+message.usage.output_tokens)
    return raw_text,scan_cost,scan_tokens

def ask(model, conversation):
    full_response=" "
    with client.messages.stream(
        model=model,
        max_tokens=4096,
        messages=conversation
        ) as stream:
        start = time.time()
        for text in stream.text_stream:
            print(f"[{time.time()-start:.2f}s] {text!r}")
            print(text, end="", flush=True)
            time.sleep(0.01)
            full_response+=text
        message = stream.get_final_message()
        ask_cost=calculate_cost(model,message.usage.input_tokens,message.usage.output_tokens)
        ask_tokens=(message.usage.input_tokens+message.usage.output_tokens)
    return full_response, ask_cost, ask_tokens
def determine_model(question):
    complexity= classify_complexity(question)
    if complexity=="SIMPLE":
        model = "claude-haiku-4-5-20251001"
        print(f"{complexity}, so going with {model}")
    else:
        model = "claude-sonnet-5"
        print(f"{complexity}, so goint with {model}")
    return model
def get_smart_response():
    conversation=[]
    grocery_list=[]
    cost=0
    total_tokens=0
    while True:
        type=input("type:")
        if type.lower()=="scan":
            path=input("path: ")
            conversation.append({"role":"user","content":path})
            scanned_groceries, scan_cost, scan_tokens=scan(path)
            try:
                grocery_list.append(json.loads(scanned_groceries))
            except json.JSONDecodeError:
                print("Couldn't parse that receipt:", scanned_groceries)
            conversation.append({"role":"assistant","content":scanned_groceries})
            #3.1
            cost+=scan_cost
            total_tokens+=scan_tokens
            print(f"{scanned_groceries},\ntoken_used: {scan_tokens}, estimated_cost: ${scan_cost}")
            print(f"token_used: {total_tokens}, estimated_cost: ${cost}")
        elif type.lower()=="ask":
            question=input("Question: ")
            conversation.append({"role":"user","content":question})
            model=determine_model(question)
            answer, ask_cost, ask_tokens=ask(model,conversation)
            cost+=ask_cost
            total_tokens+=ask_tokens
            print(f"{answer},\ntoken_used: {ask_tokens}, estimated_cost: ${ask_cost}$")
            print(f"token_used: {total_tokens}, estimated_cost: ${cost}")
            conversation.append({"role":"assistant","content":answer})

        elif type.lower()=="list":
            print(grocery_list)
        #3.2
        context_pct=(total_tokens/200000)*100
        print(f"Context usage percantage:{context_pct} ")
        if context_pct > 70:
            print("Warning: Context getting full - trimming history soon")
        if context_pct > 90:
            print("Critical TRIMMING NOW")
            conversation=trim_history(conversation)
        if type.lower()=="quit":
                print(f"Total cost: ${cost}. If opus was ussed for all the question total cost would be ${60*cost} ")
                break

       
    
    

if __name__=="__main__":
    get_smart_response()




