from .base_llm import BaseLLM


class CoTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        """
        Take a question and convert it into a chat template. The LLM will likely answer much
        better if you provide a chat template. self.tokenizer.apply_chat_template can help here
        """

        system_msg = (
            "You are an assistant specialized in unit conversions and quantitative reasoning. "
            "Always think through the calculation step by step before giving the answer, then wrap only the final numeric result in <answer> tags. "
            "Double-check your arithmetic and unit consistency."
        )

        messages = [
            {"role": "system",    "content": system_msg},

            # length conv 
            {"role": "user",      "content": "Convert 7 meters to feet."},
            {"role": "assistant", "content": (
                "Let's think: 1 m = 3.28084 ft\n"
                "7 × 3.28084 = 22.96588\n"
                "<answer>22.96588</answer> ft"
            )},

            # mass conv
            {"role": "user",      "content": "How many grams are there in 6 kilograms?"},
            {"role": "assistant", "content": (
                "Let's think: 1 kg = 1000 g\n"
                "6 × 1000 = 6000\n"
                "<answer>6000</answer> g"
            )},

            # geometry calc
            {"role": "user",      "content": "What is the area of a rectangle with length 8 and width 3?"},
            {"role": "assistant", "content": (
                "Let's think: area = length × width = 8 × 3 = 24\n"
                "<answer>24</answer>"
            )},

            # time calc
            {"role": "user",      "content": "How many seconds are in 3 minutes?"},
            {"role": "assistant", "content": (
                "Let's think: 1 min = 60 s\n"
                "3 × 60 = 180\n"
                "<answer>180</answer> s"
            )},

            # % calc
            {"role": "user",      "content": "What is 15% of 200?"},
            {"role": "assistant", "content": (
                "Let's think: 15% = 0.15\n"
                "200 × 0.15 = 30\n"
                "<answer>30</answer>"
            )},

            {"role": "user",      "content": question},
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        return prompt


def load() -> CoTModel:
    return CoTModel()


def test_model():
    from .data import Dataset, benchmark

    testset = Dataset("valid")
    model = CoTModel()
    benchmark_result = benchmark(model, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"test": test_model, "load": load})
