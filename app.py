import os
import pickle
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from collections import Counter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# =====================================
# CONFIG
# =====================================

st.set_page_config(
    page_title="AI Design Trend Finder",
    page_icon="🎨",
    layout="wide"
)

# =====================================
# LOAD MODEL
# =====================================

@st.cache_resource
def load_model():
    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

model = load_model()

# =====================================
# LOAD DATA
# =====================================

with open("embeddings/index.pkl", "rb") as f:
    db = pickle.load(f)

df = pd.read_csv("data/references.csv")

# =====================================
# ANALYTICS
# =====================================

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
    words.extend(
        title.lower().split()
    )

words = [
    w for w in words
    if w not in ignore
]

counter = Counter(words)

# =====================================
# SIDEBAR
# =====================================

st.sidebar.title("📈 Top Keywords")

for word, count in counter.most_common(10):
    st.sidebar.write(
        f"{word} ({count})"
    )

# =====================================
# FUNCTIONS
# =====================================

def category_score(results, keywords):

    score = 0

    for title, image, similarity in results[:10]:

        title = title.lower()

        for keyword in keywords:

            if keyword in title:
                score += 1
                break

    return score


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
        or "villa" in text
        or "residence" in text
    ):
        insights.append(
            "Property-focused branding appears frequently across references."
        )

    if "architecture" in text:
        insights.append(
            "Architectural storytelling is becoming an important visual strategy."
        )

    if "sustainable" in text:
        insights.append(
            "Sustainability is emerging as a supporting theme."
        )

    return insights

# =====================================
# HEADER
# =====================================

st.title(
    "🎨 AI Design Trend Finder"
)

query = st.text_input(
    "Cari trend desain",
    placeholder="quiet luxury property"
)

# =====================================
# TREND SNAPSHOT
# =====================================

st.subheader(
    "🔥 Trend Snapshot"
)

c1, c2, c3, c4, c5 = st.columns(5)

top5 = counter.most_common(5)

if len(top5) >= 5:

    c1.metric(
        top5[0][0].title(),
        top5[0][1]
    )

    c2.metric(
        top5[1][0].title(),
        top5[1][1]
    )

    c3.metric(
        top5[2][0].title(),
        top5[2][1]
    )

    c4.metric(
        top5[3][0].title(),
        top5[3][1]
    )

    c5.metric(
        top5[4][0].title(),
        top5[4][1]
    )

# =====================================
# TREND RADAR
# =====================================

st.subheader(
    "🛰️ Trend Radar"
)

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
        fill="toself"
    )
)

fig.update_layout(
    showlegend=False,
    height=450
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =====================================
# SEARCH
# =====================================

if query:

    query_vector = model.encode(
        [query]
    )

    scores = cosine_similarity(
        query_vector,
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

    st.caption(
        f"Query: {query}"
    )

    st.caption(
        f"Total Indexed: {len(db['titles'])}"
    )

    st.subheader(
        "🔍 Top Matches"
    )

    for title, image, score in ranked[:10]:

        col1, col2 = st.columns([1, 4])

        with col1:

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

                st.write("🖼️")

                st.caption(image)

        with col2:

            st.markdown(
                f"### {title}"
            )

            st.progress(
                float(score)
            )

            st.caption(
                f"Similarity: {score:.2f}"
            )

        st.divider()

    # CATEGORY

    luxury_score = category_score(
        ranked,
        ["luxury", "premium", "exclusive"]
    )

    property_score = category_score(
        ranked,
        [
            "property",
            "villa",
            "home",
            "residence",
            "apartment"
        ]
    )

    architecture_score = category_score(
        ranked,
        [
            "architecture",
            "architectural",
            "facade"
        ]
    )

    branding_score = category_score(
        ranked,
        [
            "branding",
            "marketing",
            "campaign"
        ]
    )

    st.subheader(
        "🎯 Trend Category Detection"
    )

    st.write(
        f"Luxury Living: {luxury_score}/10"
    )
    st.progress(
        luxury_score / 10
    )

    st.write(
        f"Property: {property_score}/10"
    )
    st.progress(
        property_score / 10
    )

    st.write(
        f"Architecture: {architecture_score}/10"
    )
    st.progress(
        architecture_score / 10
    )

    st.write(
        f"Branding: {branding_score}/10"
    )
    st.progress(
        branding_score / 10
    )

    # AI INSIGHT

    st.subheader(
        "🤖 AI Trend Insight"
    )

    for insight in generate_insight(ranked):

        st.success(
            insight
        )

    # EXPORT

    insight_text = f"""
Query: {query}

Luxury Score: {luxury_score}/10
Property Score: {property_score}/10
Architecture Score: {architecture_score}/10
Branding Score: {branding_score}/10
"""

    st.download_button(
        label="📥 Export Insight",
        data=insight_text,
        file_name="trend_report.txt",
        mime="text/plain"
    )

    if st.button(
        "📌 Import Pinterest Data"
    ):
        st.info(
            "Pinterest integration coming soon."
        )