import pandas as pd
import string
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
from wordcloud import WordCloud

import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# nltk.download('stopwords')
# nltk.download('punkt')

st.set_option("deprecation.showPyplotGlobalUse", False)

# Load the dataset
df = pd.read_csv("./data/kjv-bible.csv")

# Load the auxiliary dataset
df_auxiliary = pd.read_csv("./data/kjv-bible-books.csv")

# Create enriched DataFrame
df_enriched = df_auxiliary.drop(["Tanakh", "New Jerusalem Version"], axis=1)
# Replace missing values with 0 before converting to integer
df_enriched["King James Version"] = df_enriched["King James Version"].fillna(0)
df_enriched["King James Version"] = df_enriched["King James Version"].astype("int")

# Merge into original DataFrame
df = df.merge(df_enriched, left_on="b", right_on="King James Version")

# Lowercase the text
df["t"] = df["t"].astype("str")
df["t"] = df["t"].str.lower()

# Tokenize the text and remove stopwords and punctuation
sw = stopwords.words("english")
sw.extend(["from", "upon", "away", "even", "unto"])

stop_words = set(sw)
punctuations = set(string.punctuation)
word_tokens = df["t"].apply(word_tokenize)
filtered_words = word_tokens.apply(
    lambda x: [
        word
        for word in x
        if word.isalpha() and word not in stop_words and word not in punctuations
    ]
)

# Flatten the list of lists into a single list
df["filtered_words"] = filtered_words
df["filtered_words"] = df["filtered_words"].apply(lambda x: " ".join(x))

# Create "testament" column based on the book number
df.loc[df["b"] <= 39, "testament"] = "Old"
df.loc[df["b"] > 39, "testament"] = "New"

# Count word frequencies for each testament
word_freq_old_testament = Counter(
    " ".join(df[df["testament"] == "Old"]["filtered_words"]).split()
)
word_freq_new_testament = Counter(
    " ".join(df[df["testament"] == "New"]["filtered_words"]).split()
)

# Get the 20 most common words for each testament
top_20_old_testament_words = dict(word_freq_old_testament.most_common(20))
top_20_new_testament_words = dict(word_freq_new_testament.most_common(20))

# Set seaborn style
sns.set_style("whitegrid")


def plot_bar_chart(data, title):
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(
        x=list(data.keys()),
        y=list(data.values()),
        hue=list(data.keys()),
        palette="viridis",
        dodge=False,
        legend=False,
    )
    ax.set_xticks(range(len(data)))
    ax.set_xticklabels(data.keys(), rotation=45, ha="right")
    plt.xlabel("Word", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.title(title, fontsize=16)
    for p in ax.patches:
        ax.annotate(
            f"{int(p.get_height())}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="center",
            xytext=(0, 10),
            textcoords="offset points",
            fontsize=10,
        )
    sns.despine()
    plt.tight_layout()
    st.pyplot()


# Streamlit UI
st.title("bible-word-frequencies")

st.header("Top 20 Words")
plot_bar_chart(top_20_old_testament_words, "Top 20 Words in the Old Testament")

# st.header("Top 20 Words in the New Testament")
plot_bar_chart(top_20_new_testament_words, "Top 20 Words in the New Testament")

st.header("Word Clouds")
st.subheader("The Old Testament")
old_wordcloud = WordCloud().generate(
    " ".join(df[df["testament"] == "Old"]["filtered_words"])
)
st.image(old_wordcloud.to_array())

st.subheader("The New Testament")
new_wordcloud = WordCloud().generate(
    " ".join(df[df["testament"] == "New"]["filtered_words"])
)
st.image(new_wordcloud.to_array())
