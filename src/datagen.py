import json, math
from .cot import CoTModel
from .data import Dataset

def generate_dataset(
    output_json: str,
    oversample: int = 10,
    temperature: float = 0.6,
):
    # Two-pass rollout per question: try a cheap greedy generation first; if it misses,
    # sample several chain-of-thought completions and keep the first correct one.
    # The surviving (question, answer, reasoning) triples become the RFT training set.
    model = CoTModel()
    raw = Dataset("train")
    out = []
    stats = {"pass1": 0, "pass2": 0}

    for question, true_answer in raw:
        prompt = model.format_prompt(question)

        # pass 1: greedy
        gen0 = model.batched_generate(
            [prompt],
            num_return_sequences=1,
            temperature=0.0,
        )[0][0]   
        pred0 = model.parse_answer(gen0)
        if (not math.isnan(pred0)) and abs(pred0 - true_answer) < 1e-3:
            out.append([question, true_answer, gen0])
            stats["pass1"] += 1
            continue

        # pass 2: sampling
        gens = model.batched_generate(
            [prompt],
            num_return_sequences=oversample,
            temperature=temperature,
        )[0]
        chosen = None
        for gen in gens:
            pred = model.parse_answer(gen)
            if (not math.isnan(pred)) and abs(pred - true_answer) < 1e-3:
                chosen = gen
                stats["pass2"] += 1
                break

        if chosen:
            out.append([question, true_answer, chosen])

    # last recorded run: 404 examples (0 from pass1, 404 from pass2)
    with open(output_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {len(out)} examples ({stats['pass1']} from pass1, {stats['pass2']} from pass2) to {output_json}")

if __name__ == "__main__":
    from fire import Fire
    Fire(generate_dataset)

