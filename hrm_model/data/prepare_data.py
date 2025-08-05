import os
from tqdm import tqdm
import tiktoken
import numpy as np
from datasets import load_dataset

def main():
    # --- Configuration ---
    # in Colab use cwd (or replace with a Drive path)
    data_dir = os.getcwd()
    dataset_name = "timaeus/dsir-pile-1m-filtered-no-github-or-dm_mathematics"
    text_column = "contents"

    print(f"Loading dataset '{dataset_name}' in streaming mode...")
    dataset = load_dataset(dataset_name, split="train", streaming=True)

    print("Setting up tokenizer...")
    enc = tiktoken.get_encoding("gpt2")
    def tokenize_function(examples):
        return {"tokens": enc.encode_batch(examples[text_column], disallowed_special=())}

    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=list(dataset.features.keys())  # uses feature names instead of __file__
    )

    train_fp = os.path.join(data_dir, 'train.bin')
    val_fp   = os.path.join(data_dir, 'val.bin')
    for fp in (train_fp, val_fp):
        if os.path.exists(fp):
            os.remove(fp)

    print(f"Writing tokens to {train_fp} and {val_fp}...")
    total_tokens = 0
    pbar = tqdm(total=1_000_000, desc="Rows")

    with open(train_fp, 'ab') as tf, open(val_fp, 'ab') as vf:
        for ex in tokenized:
            toks = ex['tokens']
            if not toks:
                pbar.update(1)
                continue
            split = int(len(toks) * 0.9)
            tf.write(np.array(toks[:split], dtype=np.uint16).tobytes())
            vf.write(np.array(toks[split:], dtype=np.uint16).tobytes())
            total_tokens += len(toks)
            pbar.update(1)

    pbar.close()
    print("Done.")
    print(f"Tokens: {total_tokens:,}")
    print(f"Train: {train_fp}")
    print(f"Val:   {val_fp}")

if __name__ == '__main__':
    main()
