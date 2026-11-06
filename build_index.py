# build_index.py

import pandas as pd
import numpy as np
import pickle

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

df = pd.read_csv("data/references.csv")

embeddings = model.encode(
    df["title"].tolist()
)

with open("embeddings/index.pkl", "wb") as f:
    pickle.dump(
        {
            "titles": df["title"].tolist(),
            "images": df["image"].tolist(),
            "embeddings": embeddings,
        },
        f,
    )

print("Index built!")
