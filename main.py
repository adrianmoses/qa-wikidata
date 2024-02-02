from datasets import load_dataset
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from tqdm.auto import tqdm

import ast
import os
import pandas as pd

from dotenv import load_dotenv

load_dotenv()

openai_client = OpenAI()


def setup_pinecone():
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    existing_indices = [index.name for index in pc.list_indexes()]
    print(f"Existing indices {existing_indices}")
    index_name = "qa-wikidata"
    if index_name in existing_indices:
        pc.delete_index(index_name)

    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-west-2"
        )
    )

    index = pc.Index(index_name)
    return index


def read_data():
    max_articles_num = 500
    df = pd.read_csv('./wiki.csv', nrows=max_articles_num)
    return df


def prepare_embeddings(df, index):
    prepped = []

    for i, row in tqdm(df.iterrows(), total=df.shape[0]):
        meta = ast.literal_eval(row['metadata'])
        prepped.append({
            'id': row['id'],
            'values': ast.literal_eval(row['values']),
            'metadata': meta
        })

        if len(prepped) >= 250:
            index.upsert(prepped)
            prepped = []

    print(index.describe_index_stats())


def get_embeddings(articles, model="text-embedding-ada-002"):
    return openai_client.embeddings.create(input=articles, model=model)


def main():
    print("Setting up Pinecone and fetching embeddings")
    index = setup_pinecone()
    df = read_data()
    prepare_embeddings(df, index)

    query = "what is the berlin wall?"
    embed = get_embeddings([query])

    res = index.query(vector=embed.data[0].embedding, top_k=3, include_metadata=True)
    text = [r['metadata']['text'] for r in res['matches']]
    print("\n".join(text))


    print("Building Prompt...")
    query = "write an article titled: what is the berlin wall?"
    embed = get_embeddings([query])
    res = index.query(vector=embed.data[0].embedding, top_k=3, include_metadata=True)

    contexts = [
        x['metadata']['text'] for x in res['matches']
    ]

    prompt_start = (
        "Answer the question based on the context below.\n\n"+
        "Context:\n"
    )

    prompt_end = (
        f"\n\nQuestion: {query}\nAnswer:"
    )

    prompt = (
        prompt_start + "\n\n---\n\n".join(contexts) +
        prompt_end
    )

    print(prompt)

    print("Getting Summary")
    res = openai_client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        temperature=0,
        max_tokens=636,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )
    print('-' * 80)
    print(res.choices[0].text)


if __name__ == "__main__":
    main()

