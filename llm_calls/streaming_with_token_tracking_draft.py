from dotenv import load_dotenv
import os, anthropic


load_dotenv()

client= anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

def get_response_stream(conversation):
    full_response=""
    while message==client.messages.stream(model="claude-haiku-4-5-20251001", max_tokens=1234,
        messages=conversation)
    return message.content[0].text, message.usage.input_tokens, message.usage.output_tokens,message.stop_reason

def main():
    conversation=[]
    while True:
        total_input_tokens=0
        total_output_tokens=0
        user_message=input("user messgae: ")
        conversation.append({"role":"user", "content": user_message})
        reply, input_tokens, output_tokens, stop_reason = get_response(conversation)
        print(f"reply:{reply}")
        conversation.append({"role":"assistant", "content": reply})
        total_input_tokens=+input_tokens
        total_output_tokens=+output_tokens
        print(f"Input_tokens: {input_tokens} tokens")
        print(f"Output_tokens: {output_tokens} tokens")
        print(f"Stop: {stop_reason}")
        pct=((total_input_tokens+total_output_tokens)/200000)*100
        if user_message.lower()=="quit":
            print(pct)
            break

main()