import pandas as pd
import tiktoken
import openai
from openai.embeddings_utils import get_embedding


def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

openai.api_key = open_file('openaiapikey.txt')

# embedding model parameters
embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
max_tokens = 8000  # the maximum for text-embedding-ada-002 is 8191


# load & inspect dataset
input_datapath = "data/pbd-inventory-published-edited.csv"  # to save space, we provide a pre-filtered dataset
df = pd.read_csv(input_datapath, index_col=0)
df = df.reset_index()

df = df[["product_name","stock_status","regular_price","images","product_category","product_brand","product_caliber","product_link"]]
df = df.dropna()

# print(df.head(2))


encoding = tiktoken.get_encoding(embedding_encoding)

# # omit reviews that are too long to embed

df["combined"] = df["product_name"] + " " + df["stock_status"] + " " + df["regular_price"].astype(str) + " " + df["images"] + " " + df["product_category"] + " " + df["product_brand"] + " " + df["product_caliber"] + " " + df["product_link"]
df["n_tokens"] = df["combined"].apply(lambda x: len(encoding.encode(x)))

# print(df.head(5))
# print(len(df))

# This may take a few minutes
df["embedding"] = df.apply(lambda row: get_embedding(row["combined"], engine=embedding_model), axis=1)
df.to_csv("data/pbd-inventory-published-embeded.csv")
