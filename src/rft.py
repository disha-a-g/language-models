from .base_llm import BaseLLM
from .sft import test_model


def load() -> BaseLLM:
    from pathlib import Path

    from peft import PeftModel

    model_name = "rft_model"
    model_path = Path(__file__).parent.parent / model_name

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    return llm


def train_model(
    output_dir: str,
    **kwargs,
):
    from pathlib import Path
    import json, torch
    from transformers import Trainer, TrainingArguments
    from peft import LoraConfig, get_peft_model
    from .data import Dataset as Raw

    # RFT roll‑out data
    json_path = Path(__file__).parent.parent / "data" / "rft.json"
    with open(json_path) as f:
        rft_data = json.load(f)

    # mix in the original SFT examples
    sft_raw = Raw("train")
    sft_entries = [
        [q, a, f"<answer>{a}</answer>"]
        for q, a in sft_raw
    ]
    # combined list
    all_data = sft_entries + rft_data   
    llm = BaseLLM()

    # LoRA adapter
    def build_lora_model(base_model):
        cfg = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules="all-linear",
            bias="none",
            task_type="CAUSAL_LM",
        )
        m = get_peft_model(base_model, cfg)
        m.enable_input_require_grads()
        return m.to(llm.device)
    
    model = build_lora_model(llm.model)

    # now combined RFT+SFT dataset
    class RFTDataset(torch.utils.data.Dataset):
        def __init__(self, tokenizer, data, max_length=128):
            self.tokenizer = tokenizer
            self.data = data
            self.max_length = max_length

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            question, _, reasoning = self.data[idx]
            eos = self.tokenizer.eos_token
            full = f"{question} {reasoning}{eos}"
            self.tokenizer.padding_side = "right"
            self.tokenizer.pad_token = eos
            enc = self.tokenizer(
                full,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
            )

            # mask out question tokens
            qlen = len(self.tokenizer(question)["input_ids"])
            labels = [-100] * qlen + enc["input_ids"][qlen:]
            for i, m in enumerate(enc["attention_mask"]):
                if m == 0:
                    labels[i] = -100
            enc["labels"] = labels
            return enc

    train_dataset = RFTDataset(llm.tokenizer, all_data)  # ← use all_data here

    # trainer 
    default_args = dict(
        output_dir=output_dir,
        logging_dir=Path(output_dir) / "logs",
        per_device_train_batch_size=16,
        num_train_epochs=10,
        learning_rate=2e-3,
        warmup_steps=30,
        gradient_checkpointing=True,
        report_to="tensorboard",
        save_strategy="no",
    )
    default_args.update(kwargs)
    train_args = TrainingArguments(**default_args)
    trainer = Trainer(model=model, args=train_args, train_dataset=train_dataset)

    trainer.train()
    model.save_pretrained(output_dir)
    test_model(output_dir)



if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})
