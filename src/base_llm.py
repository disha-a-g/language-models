from typing import overload

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "HuggingFaceTB/SmolLM2-360M-Instruct"

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


class BaseLLM:
    def __init__(self, checkpoint=checkpoint):
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
        self.device = device

    def format_prompt(self, question: str) -> str:
        """
        Convert a question into a model prompt. The base model takes the question as-is;
        subclasses override this to add prompting structure (see CoTModel).
        """
        return question

    def parse_answer(self, answer: str) -> float:
        """
        Parse the <answer></answer> tag and return a float.
        This function is somewhat robust to output errors (e.g. missing </answer> tags).
        """
        try:
            return float(answer.split("<answer>")[1].split("</answer>")[0])
        except (IndexError, ValueError):
            return float("nan")

    def generate(self, prompt: str) -> str:
        """
        Generate a single completion. Convenience wrapper around batched_generate.
        """
        return self.batched_generate([prompt])[0]

    @overload
    def batched_generate(
        self, prompts: list[str], num_return_sequences: None = None, temperature: float = 0
    ) -> list[str]:
        """
        Batched version of `generate` method.
        This version returns a single generation for each prompt.
        """

    @overload
    def batched_generate(
        self, prompts: list[str], num_return_sequences: int, temperature: float = 0
    ) -> list[list[str]]:
        """
        Batched version of `generate` method.
        This version returns a list of generation for each prompt.
        """

    def batched_generate(
        self, prompts: list[str], num_return_sequences: int | None = None, temperature: float = 0
    ) -> list[str] | list[list[str]]:
        """
        Batched version of `generate`.

        Prompts are tokenized with left padding so every sequence is aligned on the right,
        where generation starts, and the attention mask tells the model to ignore the pads.
        Only the newly generated tokens are decoded; each prompt is stripped from its output.
        With temperature > 0 the model samples; at 0 it decodes greedily. When
        num_return_sequences is set, returns a list of generations per prompt.
        """
        from tqdm import tqdm

        # split large batches into micro batches to bound GPU memory
        micro_batch_size = 32
        if len(prompts) > micro_batch_size:
            return [
                r
                for idx in tqdm(
                    range(0, len(prompts), micro_batch_size), desc=f"LLM Running on Micro Batches {micro_batch_size}"
                )
                for r in self.batched_generate(prompts[idx : idx + micro_batch_size], num_return_sequences, temperature)
            ]

        # padding on the left so generation tokens are at the end
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # tokenize and move to device
        batch = self.tokenizer(prompts, padding=True, return_tensors="pt")
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)

        # set up gen args
        gen_kwargs = {
            "max_new_tokens": 50,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        # sample for temperature > 0 (used for RFT rollouts), greedy otherwise
        if temperature > 0:
          gen_kwargs.update({
            "do_sample": True,
            "temperature": temperature,
          })

        if num_return_sequences is not None:
            gen_kwargs["num_return_sequences"] = num_return_sequences

        # perform the actual generation
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **gen_kwargs,
        )

        # figure out how long each prompt was (excluding pads)
        pad_id = self.tokenizer.pad_token_id
        prompt_lengths = (input_ids != pad_id).sum(dim=1).tolist()
        variants = num_return_sequences or 1

        # strip off the prompt from each generation and decode
        flat_results: list[str] = []
        for i, seq in enumerate(outputs):
            prompt_idx = i // variants
            start_pos = prompt_lengths[prompt_idx]
            gen_ids = seq[start_pos:]
            text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            flat_results.append(text)

        # if multiple sequences per prompt, regroup
        if variants > 1:
            return [
                flat_results[i * variants : (i + 1) * variants]
                for i in range(len(prompts))
            ]

        return flat_results


    def answer(self, *questions) -> list[float]:
        """
        Answer questions given as individual string arguments.
        """
        # Convert each question
        prompts = [self.format_prompt(q) for q in questions]
        generations = self.batched_generate(prompts)
        return [self.parse_answer(g) for g in generations]


def test_model():
    # smoke test: the base model just needs to complete text without crashing
    testset = ["The cat went up", "The dog went down"]
    model = BaseLLM()
    for t in testset:
        print("testing generate function")
        print("input", t)
        answer = model.generate(t)
        print("output", answer)
    answers = model.batched_generate(testset)
    print(answers)


if __name__ == "__main__":
    from fire import Fire

    Fire({"test": test_model})
