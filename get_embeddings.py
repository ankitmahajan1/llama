from llama_cpp import Llama


model_path = "./model/llama-2-7b-chat.ggmlv3.q4_0.bin"


# pass embedding=True to create embedding
llm = Llama(model_path=model_path, embedding=True)
text = "my name is ankit"
result = llm.create_embedding(input=text)

text_embedding = result['data'][0]['embedding']

print(f"len of embedding: {len(text_embedding)}")
print(f"text_embedding: {text_embedding}")