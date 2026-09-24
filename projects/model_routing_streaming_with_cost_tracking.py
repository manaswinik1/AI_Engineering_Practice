
from dotenv import load_dotenv
import anthropic, os, time
load_dotenv()

client=anthropic.Anthropic(api_key=os.environ.get("ANTHROPI_API_KEY"))

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
def get_smart_response():
    conversation=[]
    cost=0
    while True:
        question=input("Question: ")
        conversation.append({"role":"user","content":question})

        complexity= classify_complexity(question)
        if complexity=="SIMPLE":
            model = "claude-haiku-4-5-20251001"
            print(f"{complexity}, so goint with {model}")
        else:
            model = "claude-sonnet-5"
            print(f"{complexity}, so goint with {model}")
        full_response=" "
        with client.messages.stream(
            model=model,
            max_tokens=4096,
            messages=conversation
            ) as stream:
            start = time.time()
            for text in stream.text_stream:
                print(">>> STREAMING BLOCK REACHED <<<")
                print(f"[{time.time()-start:.2f}s] {text!r}")
                print(text, end="", flush=True)
                time.sleep(0.01)
                full_response+=text
            message = stream.get_final_message()
        print()
        #3.1
        cost+=(message.usage.input_tokens+message.usage.output_tokens)
        # if message.content[0].text is None:
        #     answer=full_response
        # else: 
        #     answer=message.content[0].text
        answer=full_response
        print(answer)
        print(cost)
        conversation.append({"role":"assistant","content":f"{answer}, model: {model}"})
        #3.2
        context_pct=(cost/200000)*100
        print(f"Context usage percantage:{context_pct} ")
        if context_pct > 70:
            print("Warning: Context getting full - trimming history soon")
        if context_pct > 90:
            print("Critical TRIMMING NOW")
            conversation=trim_history(conversation)
        if question.lower()=="quit":
            print(f"Total cost: {cost}. If opus was ussed for all the question total cost would be {60*cost} ")
            break
    
    

if __name__=="__main__":
    get_smart_response()




