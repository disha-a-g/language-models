# Fine-tuning SmolLM2 for multi-step reasoning: CoT, LoRA SFT, and RFT

I taught a small language model (SmolLM2-360M-Instruct) to solve unit-conversion word problems ("How many grams are there per 6 kg?") through three escalating approaches: in-context chain-of-thought prompting, supervised fine-tuning with LoRA adapters, and rejection-sampling fine-tuning (RFT, Yuan et al. 2023). The generation pipeline is implemented from scratch on top of raw Hugging Face `model.generate` calls, including batched decoding, rather than using off-the-shelf pipelines.

## What's implemented

- **Batched generation from scratch** ([src/base_llm.py](src/base_llm.py)). Left-padded tokenization with attention masks so all sequences align at the generation boundary, micro-batching to bound GPU memory, temperature-based sampling with multiple return sequences per prompt, and decoding that strips each prompt's tokens so only the generated text is returned.
- **Chain-of-thought prompting** ([src/cot.py](src/cot.py)). Builds a chat dialogue with a system instruction and five worked examples (length, mass, geometry, time, percentage), rendered through the tokenizer's chat template. The model is steered to reason step by step and wrap its final number in `<answer>` tags, which a parser extracts as a float.
- **LoRA supervised fine-tuning** ([src/sft.py](src/sft.py)). Fine-tunes the model to complete a question directly with `<answer>{value}</answer>`, no chat template. Uses a rank-4 LoRA adapter on all linear layers (8.3 MB, kept small under the adapter size budget), with loss masked to the answer tokens only, gradient checkpointing, and TensorBoard logging via the Hugging Face `Trainer`.
- **RFT data generation** ([src/datagen.py](src/datagen.py)). For each training question, first tries a greedy rollout from the CoT model; if that answer is wrong, samples 10 diverse completions at temperature 0.6 and keeps the first one whose parsed answer matches the ground truth. Questions with no correct rollout are dropped. The surviving question / answer / reasoning triples are written to `data/rft.json`.
- **RFT training** ([src/rft.py](src/rft.py)). Fine-tunes on the self-generated reasoning traces mixed with the original direct-answer examples, using a rank-8 LoRA adapter (17 MB). The result is a model that both reasons in steps and answers in the expected format.
- **Evaluation harness** ([src/data.py](src/data.py)). Benchmarks a model on validation questions and reports accuracy (parsed answer within 5% relative tolerance of ground truth) and answer rate (fraction of outputs containing a parseable answer).

The dataset is 1,000 training and 1,000 validation unit-conversion questions with float answers.

## Results

Measured by running the committed adapters on the first 100 validation questions with greedy decoding. A prediction counts as correct if it is within 5% relative tolerance of the ground truth; answer rate is the fraction of outputs containing a parseable numeric answer.

| Approach | Accuracy | Answer rate |
|---|---|---|
| Chain-of-thought prompting (no training) | 0.50 | 0.85 |
| LoRA SFT | 0.67 | 1.00 |
| RFT | 0.77 | 0.97 |

Each stage earns its keep: fine-tuning teaches the model to always produce a well-formed answer and lifts accuracy well past prompting alone, and training on self-generated reasoning traces takes it further still. Getting the LoRA recipe right also mattered: raising the adapter alpha and learning rate over my first configuration was worth nearly 20 points of validation accuracy during development. The RFT data generation step recovered correct reasoning traces for 404 of the 1,000 training questions, all of them via temperature sampling rather than greedy decoding, which is exactly the diversity effect rejection sampling relies on. The RFT model performed well enough to earn extra credit in the course evaluation.

Both trained adapters are committed (`sft_model/`, `rft_model/`), so these numbers can be reproduced without retraining.

## Repository structure

```
src/
  base_llm.py    Model wrapper with generate and batched_generate
  cot.py         Chain-of-thought prompt construction
  sft.py         LoRA supervised fine-tuning and evaluation
  datagen.py     RFT dataset generation via rejection sampling
  rft.py         RFT training
  data.py        Dataset loading and benchmark utilities
data/
  train.json     1,000 training questions
  valid.json     1,000 validation questions
sft_model/       Trained LoRA adapter (rank 4)
rft_model/       Trained LoRA adapter (rank 8)
```

## Setup

Python 3.12 with [PyTorch](https://pytorch.org/get-started/locally/) installed, then:

```bash
pip install -r requirements.txt
```

The code picks the best available device automatically (CUDA, then Apple MPS, then CPU). Training is practical on a single GPU; the base model is only 360M parameters.

## Usage

Sanity-check generation and batched generation:

```bash
python -m src.base_llm test
```

Benchmark the chain-of-thought model (no training required):

```bash
python -m src.cot test
```

Train and evaluate the SFT model:

```bash
python -m src.sft train sft_model
python -m src.sft test sft_model
```

Generate the RFT dataset, then train and evaluate the RFT model:

```bash
python -m src.datagen data/rft.json
python -m src.rft train rft_model
python -m src.rft test rft_model
```

`datagen` accepts `--oversample` (completions sampled per question, default 10) and `--temperature` (default 0.6).

## Provenance

Built as coursework for Advances in Deep Learning at UT Austin. The dataset and the data loading and benchmark scaffolding came with the course; the generation, prompting, fine-tuning, and data generation implementations in this repo are my own.
