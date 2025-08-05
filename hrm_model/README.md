# HRM Model Implementation

This directory contains the implementation of the Hierarchical Reasoning Model (HRM) based on the paper "Hierarchical Reasoning Model".

## Setup

1.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate # on Windows, use `venv\Scripts\activate`
    ```

2.  **Install dependencies:**
    ```bash
    pip install torch numpy tqdm tiktoken requests
    ```

## Data Preparation

The `data/prepare_data.py` script is provided to download a sample dataset and prepare it for training. For full-scale training, you will need to replace the sample data with your own large-scale corpus.

1.  **Run the data preparation script:**
    ```bash
    python data/prepare_data.py
    ```
    This will download the sample data and create `train.bin` and `val.bin` in the `data` directory.

## Training

The `train.py` script contains the main training loop for the HRM model.

1.  **Run the training script:**
    ```bash
    python train.py
    ```
    This will start the training process using the prepared data. You can modify the training configurations at the top of the `train.py` file to suit your needs.

## Next Steps

Once the code is transferred to your dedicated machine, you can:

1.  **Replace the sample data:** Update the `data/prepare_data.py` script to use your large-scale English web and code dataset.
2.  **Configure for distributed training:** The current `train.py` script is set up for single-device training. You will need to add support for distributed data parallel to train the 1.5B parameter model efficiently across multiple GPUs.
3.  **Run full-scale training:** Adjust the `max_iters` and other hyperparameters in `train.py` for your full training run.