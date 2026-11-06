from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

text = "luxury interior design"

embedding = model.encode(text)

print(len(embedding))