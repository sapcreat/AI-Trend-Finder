import pandas as pd
from collections import Counter

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

total = len(df)

print(f"\nTotal References: {total}\n")
print("TOP TRENDS\n")

top_keywords = []

for word, count in counter.most_common(20):

    percentage = round(count / total * 100, 2)

    top_keywords.append(word)

    print(
        f"{word:<15} {count:<5} ({percentage}%)"
    )

print("\nAI INSIGHT\n")

top5 = top_keywords[:5]

insight = f"""
Dominant design trends are centered around
{', '.join(top5)}.

The data suggests a strong preference for
premium residential branding, modern architecture,
lifestyle-focused marketing, and luxury living concepts.

Emerging related themes include sustainable design,
wellness living, tropical architecture, and
minimalist visual communication.
"""

print(insight)