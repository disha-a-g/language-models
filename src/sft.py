from .base_llm import BaseLLM
from .data import Dataset, benchmark


def load() -> BaseLLM:
    from pathlib import Path

    from peft import PeftModel

    model_name = "sft_model"
    model_path = Path(__file__).parent.parent / model_name

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    return llm


def tokenize(tokenizer, question: str, answer: str):
    """
    Tokenize a data element.
    We first append the <EOS> token to the question / answer pair.
    Then we tokenize and construct the ground truth `labels`.
    `labels[i] == -100` for the question or masked out parts, since we only want to supervise
    the answer.
    """
    full_text = f"{question} {answer}{tokenizer.eos_token}"

    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    full = tokenizer(full_text, padding="max_length", truncation=True, max_length=128)

    input_ids = full["input_ids"]
    question_len = len(tokenizer(question)["input_ids"])

    # Create labels: mask out the prompt part
    labels = [-100] * question_len + input_ids[question_len:]

    for i in range(len(labels)):
        if full["attention_mask"][i] == 0:
            labels[i] = -100

    full["labels"] = labels
    return full


def format_example(prompt: str, answer: str) -> dict[str, str]:
    """
    Construct a question / answer pair with the answer wrapped in <answer> tags.
    """
    return {
        "question": prompt,
        "answer": f"<answer>{answer}</answer>",
    }


class TokenizedDataset:
    def __init__(self, tokenizer, data: Dataset, format_fn):
        """
        Use the
        - BaseLLM.tokenizer
        - Dataset
        - format_fn which converts a data element into a dict with entries
          - question: str
          - answer: str
        """
        self.format_fn = format_fn
        self.tokenizer = tokenizer
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        formated_data = self.format_fn(*self.data[idx])
        return tokenize(self.tokenizer, **formated_data)


def train_model(
    output_dir: str,
    **kwargs,
):
    # tuning note: raising lora_alpha and the learning rate over the first
    # configuration improved validation accuracy from 0.52 to 0.71
    from pathlib import Path
    import torch
    from transformers import Trainer, TrainingArguments
    from peft import LoraConfig, get_peft_model

    # lora adapter model
    def build_lora_model(base_model):
        cfg = LoraConfig(
            r=4,
            lora_alpha=32,
            target_modules="all-linear",
            bias="none",
            task_type="CAUSAL_LM",
        )
        lm = get_peft_model(base_model, cfg)
        lm.print_trainable_parameters()
        lm.enable_input_require_grads()
        if torch.cuda.is_available():
            lm = lm.to(next(base_model.parameters()).device)
        return lm

    # dataset
    def get_train_dataset(tokenizer):
        raw = Dataset("train")
        return TokenizedDataset(tokenizer, raw, format_example)

    # trainer
    def make_trainer(model, dataset, **params):
        default_args = dict(
            output_dir=output_dir,
            logging_dir=Path(output_dir) / "logs",
            per_device_train_batch_size=32,
            num_train_epochs=5,
            learning_rate=1e-3,
            warmup_steps=30,
            gradient_checkpointing=True,
            report_to="tensorboard",
            save_strategy="no",
        )
        default_args.update(params)      
        train_args = TrainingArguments(**default_args)
        return Trainer(model=model, args=train_args, train_dataset=dataset)

    # orchestrate
    llm = BaseLLM()
    model = build_lora_model(llm.model)
    train_data = get_train_dataset(llm.tokenizer)
    trainer = make_trainer(model, train_data, **kwargs)

    trainer.train()

    # saving + quick checks 
    model.save_pretrained(output_dir)
    test_model(output_dir)
    

def test_model(ckpt_path: str):
    testset = Dataset("valid")
    llm = BaseLLM()

    # Load the model with LoRA adapters
    from peft import PeftModel

    llm.model = PeftModel.from_pretrained(llm.model, ckpt_path).to(llm.device)

    benchmark_result = benchmark(llm, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})
