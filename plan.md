# Project Plan: GPT-2-Scale Hierarchical Reasoning Model

This document outlines the plan to design, train, and evaluate a GPT-2-scale language model based on the Hierarchical Reasoning Model (HRM) architecture.

## Phase 1: Model Design and Architecture

**Goal:** Define the specific architecture of the 1.5B parameter HRM model.

*   **1.1. Finalize Model Architecture:**
    *   **Total Parameters:** ~1.5 Billion
    *   **Model Dimension (`d_model`):** 3072
    *   **Attention Heads (`n_head`):** 12
    *   **Head Dimension (`d_head`):** 256
    *   **Total Transformer Layers:** 12. This will be split between the two identical recurrent modules:
        *   **Low-Level Module (`f_L`):** 6 Transformer layers
        *   **High-Level Module (`f_H`):** 6 Transformer layers
    *   **Feed-Forward Dimension (`d_ffn`):** 8192 (with SwiGLU)
    *   **Vocabulary Size (`vocab_size`):** 50,257 (GPT-2 standard)
*   **1.2. Define Training Hyperparameters:**
    *   **High-Level Cycles (N):** To be determined during initial testing, likely in the range of 2-4.
    *   **Low-Level Timesteps (T):** To be determined during initial testing, likely in the range of 2-4.
    *   **ACT Parameters:** `M_max` will be determined based on task complexity during evaluation, `epsilon` will be set to a small value (e.g., 0.1) to encourage exploration.
*   **1.3. Confirm Implementation Details:**
    *   The one-step gradient approximation will be implemented as described in the paper, detaching hidden states between deep supervision segments.
    *   The model will use a Post-Norm architecture with RMSNorm and no bias terms.
    *   Weights will be initialized with LeCun Normal initialization.

## Phase 2: Data Preparation and Preprocessing

**Goal:** Prepare a high-quality, large-scale dataset for training.

*   **2.1. Dataset Sourcing:**
    *   Acquire a large English web and code dataset (e.g., a subset of The Pile, C4, or a similar corpus) to reach the target of approximately 10 billion tokens.
*   **2.2. Preprocessing Pipeline:**
    *   Develop a robust preprocessing pipeline for cleaning, and tokenizing the data.
    *   Utilize a BPE tokenizer similar to GPT-2.
    *   Format the data for the sequence-to-sequence architecture used by HRM.

## Phase 3: Implementation and Training Infrastructure

**Goal:** Implement the HRM model and set up the training environment.

*   **3.1. Model Implementation:**
    *   Implement the HRM architecture in a suitable deep learning framework (e.g., PyTorch or JAX).
    *   Implement the H and L recurrent modules using Transformer blocks.
    *   Integrate modern LLM enhancements (Rotary Positional Encoding, GLU, RMSNorm) as described in the paper.
    *   Implement the one-step gradient approximation and the deep supervision loop.
    *   Implement the Adaptive Computation Time (ACT) mechanism with the Q-learning head.
*   **3.2. Training Infrastructure:**
    *   Set up a distributed training environment capable of handling a 1.5B parameter model.
    *   Implement data parallelism to distribute the training process across multiple GPUs.
    *   Choose an appropriate optimizer (Adam-atan2 as per the paper, or a similar scale-invariant Adam variant).

## Phase 4: Model Training

**Goal:** Train the 1.5B parameter HRM model to convergence.

*   **4.1. Initial Training and Stability Checks:**
    *   Begin training on a smaller subset of the data to ensure training stability and that the loss is decreasing as expected.
    *   Monitor the forward residuals to confirm the hierarchical convergence mechanism is functioning correctly.
*   **4.2. Full-Scale Training:**
    *   Launch the full training run on the entire ~10B token dataset.
    *   Continuously monitor key metrics: training loss, perplexity, and gradient norms.
    *   Adjust the learning rate and other hyperparameters as needed based on training dynamics.
    *   Log all training metrics and save model checkpoints periodically.

## Phase 5: Evaluation and Analysis

**Goal:** Evaluate the model's performance and analyze its properties.

*   **5.1. Standard Language Modeling Evaluation:**
    *   Evaluate the model on standard language modeling benchmarks (e.g., perplexity on a held-out test set).
*   **5.2. Reasoning Benchmark Evaluation:**
    *   Evaluate the trained model on the reasoning benchmarks from the paper: ARC-AGI, Sudoku-Extreme, and Maze-Hard. This will be a zero-shot or few-shot evaluation, without specific fine-tuning on these tasks unless necessary.
*   **5.3. Architectural Analysis:**
    *   Analyze the emergent properties of the scaled-up model, replicating the dimensionality analysis (Participation Ratio) from the paper to see if the hierarchical dimensionality organization holds at a larger scale.
    *   Visualize intermediate timesteps on reasoning tasks to understand the model's problem-solving strategies.
*   **5.4. Final Report:**
    *   Compile all findings into a final report, comparing the performance of the GPT-2-scale HRM with the original Transformer-based GPT-2 and other relevant models.