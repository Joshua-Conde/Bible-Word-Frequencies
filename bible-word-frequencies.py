import pandas as pd
import string
import matplotlib.pyplot as plt
from collections import Counter

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# nltk.download('stopwords')
# nltk.download('punkt')

# python3 bible-word-frequencies.py

# Load the dataset
df = pd.read_csv("./kjv.csv")

# Create "testament" column based on the book number
df["t"] = df["t"].astype("str")
df.loc[df["b"] <= 39, "testament"] = "old"
df.loc[df["b"] > 39, "testament"] = "new"

# Lowercase the text
df["t"] = df["t"].str.lower()

# Tokenize the text and remove stopwords and punctuation
stop_words = set(stopwords.words("english"))
punctuations = set(string.punctuation)
word_tokens = df["t"].apply(word_tokenize)
filtered_words = word_tokens.apply(
    lambda x: [
        word
        for word in x
        if word.isalpha() and word not in stop_words and word not in punctuations
    ]
)

# Combine the filtered words with the 'testament' column
df["filtered_words"] = filtered_words

# Flatten the list of lists into a single list
df["filtered_words"] = df["filtered_words"].apply(lambda x: " ".join(x))

# Count word frequencies for each testament
word_freq_old_testament = Counter(
    " ".join(df[df["testament"] == "old"]["filtered_words"]).split()
)
word_freq_new_testament = Counter(
    " ".join(df[df["testament"] == "new"]["filtered_words"]).split()
)

# Get the 20 most common words for each testament
top_20_old_testament_words = dict(word_freq_old_testament.most_common(20))
top_20_new_testament_words = dict(word_freq_new_testament.most_common(20))

# Plot word frequency distribution for the Old Testament
plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
plt.bar(*zip(*top_20_old_testament_words.items()))
plt.xlabel("Word")
plt.ylabel("Frequency")
plt.title("Top 20 Words in the Old Testament")
plt.xticks(rotation=45)

# Plot word frequency distribution for the New Testament
plt.subplot(1, 2, 2)
plt.bar(*zip(*top_20_new_testament_words.items()))
plt.xlabel("Word")
plt.ylabel("Frequency")
plt.title("Top 20 Words in the New Testament")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
