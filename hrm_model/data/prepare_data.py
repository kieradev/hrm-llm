import os
import requests
from tqdm import tqdm
import tiktoken
import numpy as np

def download_file(url, fname, chunk_size=1024):
    """Helper function to download a file with a progress bar."""
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get('content-length', 0))
    with open(fname, 'wb') as file, tqdm(
        desc=fname,
        total=total,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)

def main():
    """
    Downloads a sample dataset and preprocesses it.
    """
    # --- 1. Download and Prepare the Data ---
    data_dir = os.path.dirname(os.path.abspath(__file__))
    input_file_path = os.path.join(data_dir, 'input.txt')
    
    # Download a small sample dataset (Shakespeare's Hamlet)
    if not os.path.exists(input_file_path):
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        print(f"Downloading data from {url}...")
        download_file(url, input_file_path)
    else:
        print(f"{input_file_path} already exists. Skipping download.")
        
    with open(input_file_path, 'r') as f:
        data = f.read()
    n = len(data)
    print(f"Dataset has {n} characters.")

    # --- 2. Tokenize the Data ---
    # According to the plan, we'll use the GPT-2 tokenizer.
    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode_ordinary(data)
    print(f"Dataset has {len(tokens)} tokens.")
    
    # --- 3. Create Training and Validation Splits ---
    # We'll use a 90/10 split.
    train_tokens = tokens[:int(n*0.9)]
    val_tokens = tokens[int(n*0.9):]

    # Export to bin files
    train_ids = np.array(train_tokens, dtype=np.uint16)
    val_ids = np.array(val_tokens, dtype=np.uint16)
    
    train_file = os.path.join(data_dir, 'train.bin')
    val_file = os.path.join(data_dir, 'val.bin')
    
    train_ids.tofile(train_file)
    val_ids.tofile(val_file)

    print(f"Saved training data to {train_file}")
    print(f"Saved validation data to {val_file}")
    
if __name__ == '__main__':
    main()