# Used to create the .csv files from .txt files to use for paranormal.py.

import nltk
import re
import pandas as pd
from nltk import sent_tokenize

# Load text
# filename = "./harry_potter_1.txt"
filename = "./harry_potter_2.txt"
file = open(filename, "r", encoding="utf8")
text = file.read()
file.close()

# Replace all \n\n with \n
text = re.sub(r"(?:(\S)[ \t]*)(\n){1}(?:(\S)[ \t]*)", r"\1 \3", text)
text = text.replace("\n\n", "\n")

# Split text into paragraphs
paragraphs = text.split("\n")
paragraphs = [p for p in paragraphs if p]  # Ignore all whitespace

# Split paragraphs into sentences
sentences = []
for p in paragraphs:
    s = sent_tokenize(p)
    # s = [sent for sent in s if sent]  # Ignore all whitespace
    if len(s) == 1:
        s[0] = s[0] + "\n"
    else:
        s[-1] = s[-1] + "\n"
    sentences.extend(s)

# Classifiers
df = pd.DataFrame(
    {
        "sentence": [],
        "sent_length": [],
        "end_of_par": [],
        "final_punct": [],
        "final_word": [],
    }
)
df["sentence"] = sentences
df = df.astype({"final_punct": "string", "final_word": "string"})

# Determine classifier values
for i, s in df.iterrows():
    curr_sent = s["sentence"]

    # Sentence length
    df.at[i, "sent_length"] = len(curr_sent)

    # Last sentence in a paragraph
    if str(curr_sent[-1]) == "\n":
        df.at[i, "end_of_par"] = 1
    else:
        df.at[i, "end_of_par"] = 0

    # Punctuation at the end of sentence
    df.at[i, "final_punct"] = (
        curr_sent[-2] if (curr_sent[-1] == "\n") else curr_sent[-1]
    )

    # Final word in the sentence
    words = curr_sent.split(" ")
    df.at[i, "final_word"] = words[-1]

# df.to_csv("train.csv")
df.to_csv("test.csv")
