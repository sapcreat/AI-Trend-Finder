# search.py

import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

with open("embeddings/index.pkl", "rb") as f:
    db = pickle.load(f)

query = input("Search: ")

query_embedding = model.encode([query])

scores = cosine_similarity(
    query_embedding,
    db["embeddings"]
)[0]

ranked = sorted(
    zip(
        db["titles"],
        db["images"],
        scores
    ),
    key=lambda x: x[2],
    reverse=True
)

for title, image, score in ranked[:5]:
    print()
    print(title)
    print(image)
    print(score)

st.subheader("🤖 AI Trend Insight")

df = pd.read_csv("data/references.csv")

words = []

for title in df["title"]:
    words.extend(title.lower().split())

counter = Counter(words)

top_keywords = [
    word
    for word, count in counter.most_common(5)
]

insight = f"""
Dominant design trends are centered around:
{', '.join(top_keywords)}.

The dataset indicates growing interest in premium
living, modern architecture, interior design,
property branding, and luxury experiences.
"""

st.info(insight)

st.subheader("🤖 AI Trend Insight")

df = pd.read_csv("data/references.csv")

words = []

ignore = {
    "and",
    "the",
    "of",
    "for",
    "to",
    "a",
    "in"
}

for title in df["title"]:
    words.extend(title.lower().split())

words = [w for w in words if w not in ignore]

counter = Counter(words)

top_keywords = [
    word
    for word, count in counter.most_common(5)
]

insight = f"""
### Trend Summary

Dominant keywords:
**{', '.join(top_keywords)}**

Market observations:

• Luxury positioning remains the strongest direction.

• Modern architecture and interior aesthetics continue to dominate visual references.

• Premium residential branding is a recurring theme across the dataset.

• Lifestyle-focused communication is frequently associated with property marketing.

• Emerging opportunities include sustainable living, wellness-focused design, and tropical modern concepts.
"""

st.markdown(insight)

st.subheader("🤖 AI Trend Insight")

for insight in generate_insight(ranked):

    st.success(insight)
