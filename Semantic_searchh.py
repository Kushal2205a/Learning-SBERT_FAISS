"""
Semantic Search :
                Embed all corpus into a vector space. At search time the query is embedded into the same vector space as the corpus and the closest embeddings from the 
corpus are found.

Two types = [Symmetric Semantic search : query and the target corpus are of same length]
            [Asymmetric semantic search : Query is usally smaller and the target corpus is a big paragraph explaining the query]


                    
"""
import torch 
from sentence_transformers import SentenceTransformer 


model = SentenceTransformer("all-MiniLM-L6-v2")
corpus = [
    "A man is eating food.",
    "A man is eating a piece of bread.",
    "The girl is carrying a baby.",
    "A man is riding a horse.",
    "A woman is playing violin.",
    "Two men pushed carts through the woods.",
    "A man is riding a white horse on an enclosed ground.",
    "A monkey is playing drums.",
    "A cheetah is running behind its prey.",
]

corpus_embeddings = model.encode(corpus)

queries = [
    "A man is eating pasta.",
    "Someone in a gorilla costume is playing a set of drums.",
    "A cheetah chases prey on across a field.",
]

top_k = min(5, len(corpus))


for query in queries:
    querie_embeddings = model.encode(query)
    simil = model.similarity(querie_embeddings, corpus_embeddings)[0]
    scores, indices = torch.topk(simil, k = top_k)

    print(f"\nQuery : {query}")

    for i,j in zip(scores,indices):
        print(f"{corpus[j]} score : {i}")

