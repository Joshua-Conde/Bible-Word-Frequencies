# bible-word-frequencies

## Overview
This project analyzes word frequencies in the King James Version of the Bible, specifically comparing word frequencies between the Old and New Testaments.

## Dependencies
- matplotlib
- nltk
- numpy
- pandas
- seaborn
- streamlit
- wordcloud

## Contents
- `bible-word-frequencies.ipynb`: Jupyter notebook containing the project.
- `kjv-bible.csv`: Dataset containing the text of the King James Version of the Bible.
- `kjv-bible-books.csv`: Auxiliary dataset containing information about the books of the Bible.

## Data Preprocessing
- Loading the dataset and auxiliary dataset.
- Enriching the dataset by merging with the auxiliary dataset.
- Lowercasing the text and tokenizing it.
- Removing stopwords, punctuation, and non-alphabetic characters.
- Creating a column indicating the testament (Old or New) based on book numbers.

## Data Exploration and Visualization
- Analyzing word frequencies in the Old and New Testaments.
- Plotting the top 20 words in each testament.
- Visualizing the number of verses and number of words per book and time period.
- Creating word clouds for the Old and New Testaments.

## Conclusion
- Significant differences in word frequencies exist between the Old and New Testaments.
- Visualization techniques provide insights into the structure and content of the Bible.
- Further analysis could explore linguistic patterns and theological implications.
