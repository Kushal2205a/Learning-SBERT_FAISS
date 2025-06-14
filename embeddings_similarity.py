from sentence_transformers import SentenceTransformer 

# 1. Load a pretrained model 
model = SentenceTransformer("all-MiniLM-L6-v2")

#   The sentences to encode
sentences1 = [
    "The new movie is awesome",
    "The cat sits outside",
    "A man is playing guitar",
]

sentences2 = [
    "The dog plays in the garden",
    "The new movie is so great",
    "A woman watches TV",
]

# 2. Calculate embeddings 
embeddings1 = model.encode(sentences1)
embeddings2 = model.encode(sentences2)


# 3. Calculate embeddings Similarities (Shows how likely similar are two words)
similarities = model.similarity(embeddings1,embeddings2)

for i,sentence1 in enumerate(sentences1):
    print(sentence1)
    for j,sentence2 in enumerate(sentences2):
        print(f"{sentences2[j]}: {similarities[i][j]}")

