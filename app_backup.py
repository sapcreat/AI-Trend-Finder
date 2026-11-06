import streamlit as st
import pickle
import pandas as pd

from collections import Counter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ====================================
# CONFIG
# ====================================

st.set_page_config(
    page_title="AI Design Trend Finder",
    layout="wide"
)

# ====================================
# LOAD MODEL
# ====================================

@st.cache_resource
def load_model():
    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

model = load_model()

# ====================================
# LOAD DATA
# ====================================

with open("embeddings/index.pkl", "rb") as f:
    db = pickle.load(f)

df = pd.read_csv("data/references.csv")

# ====================================
# KEYWORD ANALYSIS
# ====================================

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

# ====================================
# SIDEBAR
# ====================================

st.sidebar.title("📈 Top Keywords")

for word, count in counter.most_common(10):
    st.sidebar.write(f"{word} ({count})")

# ====================================
# FUNCTIONS
# ====================================

def generate_insight(results):

    text = " ".join(
        [title for title, _, _ in results[:10]]
    ).lower()

    insights = []

    if "luxury" in text:
        insights.append(
            "Luxury positioning remains the dominant market direction."
        )

    if (
        "property" in text
        or "residence" in text
        or "villa" in text
    ):
        insights.append(
            "Property-focused branding appears frequently across references."
        )

    if "architecture" in text:
        insights.append(
            "Architectural storytelling is becoming an important visual strategy."
        )

    if "residence" in text:
        insights.append(
            "Residential lifestyle positioning is strongly represented."
        )

    if "sustainable" in text:
        insights.append(
            "Sustainability is emerging as a supporting theme."
        )

    return insights


def category_score(results, keywords):

    score = 0

    for title, image, similarity in results[:10]:

        title = title.lower()

        for keyword in keywords:

            if keyword in title:
                score += 1
                break

    return score


# ====================================
# APP HEADER
# ====================================

st.title("🎨 AI Design Trend Finder")

query = st.text_input(
    "Cari trend desain",
    placeholder="quiet luxury property"
)

# ====================================
# TREND SNAPSHOT
# ====================================

st.subheader("🔥 Trend Snapshot")

c1, c2, c3, c4, c5 = st.columns(5)

top5 = counter.most_common(5)

if len(top5) >= 5:
    c1.metric(top5[0][0].title(), top5[0][1])
    c2.metric(top5[1][0].title(), top5[1][1])
    c3.metric(top5[2][0].title(), top5[2][1])
    c4.metric(top5[3][0].title(), top5[3][1])
    c5.metric(top5[4][0].title(), top5[4][1])

# ====================================
# SEARCH
# ====================================

if query:

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

    st.caption(f"Query: {query}")
    st.caption(f"Total Indexed: {len(db['titles'])}")

    # ====================================
    # TOP MATCHES
    # ====================================

    st.subheader("🔍 Top Matches")

    for title, image, score in ranked[:10]:

        with st.container():

            col1, col2 = st.columns([1, 4])

            with col1:
                st.write("🖼️")
               import os

image_path = os.path.join(
    "images",
    image
)

if os.path.exists(image_path):
    st.image(
        image_path,
        use_container_width=True
    )
else:
    st.caption(image)

            with col2:

                st.markdown(
                    f"### {title}"
                )

                st.progress(
                    min(max(float(score), 0), 1)
                )

                st.caption(
                    f"Similarity: {score:.2f}"
                )

            st.divider()

    # ====================================
    # CATEGORY DETECTION
    # ====================================

    luxury_keywords = [
        "luxury",
        "premium",
        "exclusive",
        "high end"
    ]

    property_keywords = [
        "property",
        "real estate",
        "residence",
        "home",
        "housing",
        "villa",
        "apartment"
    ]

    architecture_keywords = [
        "architecture",
        "architectural",
        "facade",
        "building"
    ]

    lifestyle_keywords = [
        "living",
        "lifestyle",
        "wellness"
    ]

    branding_keywords = [
        "branding",
        "marketing",
        "campaign",
        "identity",
        "brochure"
    ]

    luxury_score = category_score(
        ranked,
        luxury_keywords
    )

    property_score = category_score(
        ranked,
        property_keywords
    )

    architecture_score = category_score(
        ranked,
        architecture_keywords
    )

    lifestyle_score = category_score(
        ranked,
        lifestyle_keywords
    )

    branding_score = category_score(
        ranked,
        branding_keywords
    )

    st.subheader("🎯 Trend Category Detection")

    st.write(f"Luxury Living: {luxury_score}/10")
    st.progress(luxury_score / 10)

    st.write(f"Property: {property_score}/10")
    st.progress(property_score / 10)

    st.write(f"Architecture: {architecture_score}/10")
    st.progress(architecture_score / 10)

    st.write(f"Lifestyle: {lifestyle_score}/10")
    st.progress(lifestyle_score / 10)

    st.write(f"Branding: {branding_score}/10")
    st.progress(branding_score / 10)

    # ====================================
    # AI INSIGHT
    # ===============================

import plotly.graph_objects as go

st.subheader("🛰️ Trend Radar")

radar_categories = [
    "Luxury",
    "Modern",
    "Interior",
    "Property",
    "Architecture"
]

radar_values = [
    counter.get("luxury", 0),
    counter.get("modern", 0),
    counter.get("interior", 0),
    counter.get("property", 0),
    counter.get("architecture", 0)
]

fig = go.Figure()

fig.add_trace(
    go.Scatterpolar(
        r=radar_values,
        theta=radar_categories,
        fill='toself',
        name='Design Trend Strength'
    )
)

fig.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True
        )
    ),
    showlegend=False,
    height=550
)

st.plotly_chart(
    fig,
    use_container_width=True
)

if st.button("📌 Import Pinterest Data"):
    st.success("Pinterest import started...")

st.download_button(
    "📥 Export Insight",
    insight_text,
    file_name="trend_report.txt"
)