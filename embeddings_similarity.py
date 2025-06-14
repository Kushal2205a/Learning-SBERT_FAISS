from sentence_transformers import SentenceTransformer 

# 1. Load a pretrained model 
model = SentenceTransformer("all-MiniLM-L6-v2")

#   The sentences to encode
sentences = [
    "The weather is lovely today.",
    "It's so sunny outside!",
    "He drove to the stadium.",
]

# 2. Calculate embeddings 
embeddings = model.encode(sentences)
print(embeddings.shape)

# 3. Calculate embeddings Similarities (Shows how likely similar are two words)
similarities = model.similarity(embeddings,embeddings)

print(similarities)
