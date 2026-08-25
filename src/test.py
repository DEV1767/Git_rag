from embedding_model import embedding_model

text = "This is a GitHub repository."

vector = embedding_model.embed_query(text)

print("Embedding created successfully")
print("Dimensions:", len(vector))
print("First 5 values:", vector[:5])
