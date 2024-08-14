from transformers import pipeline

# Load a local model (e.g., distilGPT-2 or any other small transformer model)
chatbot = pipeline('text-generation', model='gpt2')

def chat(messages):
    # Example: Concatenate the last human message and generate a response
    last_message = messages[-1]['content'] if messages else "Hello!"
    response = chatbot(last_message, max_length=50, do_sample=True)[0]['generated_text']
    return {"content": response.strip()}

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "human", "content": "Hi, how are you?"}
]

response = chat(messages)
print(response['content'])
