from sentence_transformers import SentenceTransformer 

model = SentenceTransformer(
    "intfloat/multilingual-e5-large",
    prompts={
        "classification": "Classify the following text: ",
        "retrieval": "Retrieve semantically similar text: ",
        "clustering": "Identify the topic or theme based on the text: ",
    },
    default_prompt_name="retrieval",
)

sentences = [
    "The weather is lovely today.",
    "It's so sunny outside!",
    "He drove to the stadium.",
]

prompt = model.prompts["retrieval"]
prompted_sentences = [prompt + sentence for sentence in sentences]
embeddings = model.encode(prompted_sentences)
print(prompted_sentences)