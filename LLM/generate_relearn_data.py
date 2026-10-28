"""Generate the benign relearning corpus with any OpenAI-compatible chat endpoint.

Passages are encyclopedia-style texts on general topics from the same broad area as the
forget set (computing for WMDP-Cyber, people/places/works for RWKU). Every passage that
mentions an RWKU target or shares a long n-gram with the forget corpus is dropped, so the
relearning data stays disjoint from the forget set.

    export OPENAI_API_KEY=...            # never written to disk by this script
    python generate_relearn_data.py --forget-dataset wmdp --num-texts 200 --api-model gpt-4o-mini
    python generate_relearn_data.py --forget-dataset rwku --base-url http://localhost:8000/v1 --api-model Qwen2.5-72B-Instruct
"""

import argparse
import json
import os
import random
import re
from pathlib import Path

from pleu_llm.data import DEFAULT_DATA_ROOT, load_rwku_forget_corpus, load_wmdp_corpus, relearn_data_path, rwku_dir

TOPICS = {
    "wmdp": [
        "relational database normalization", "the history of the Unix operating system", "TCP congestion control",
        "compilers and intermediate representations", "garbage collection in managed runtimes",
        "version control with Git", "the design of the HTTP protocol", "cache coherence in multicore CPUs",
        "functional programming languages", "distributed consensus algorithms", "unit testing practices",
        "the evolution of graphical user interfaces", "file systems and journaling", "virtual memory and paging",
        "open-source software licensing", "the history of the World Wide Web", "data compression algorithms",
        "cloud computing service models", "software project management", "accessibility in web design",
    ],
    "rwku": [
        "a 19th-century European painter", "a classical composer", "a famous bridge", "a river in South America",
        "a medieval castle", "a scientific instrument", "an ancient trade route", "a traditional cuisine",
        "a national park", "a Renaissance architect", "a historic sailing ship", "a mountain range in Asia",
        "an early aviation pioneer", "a well-known museum", "a folk music tradition", "a botanical garden",
        "a historic university", "an island nation", "a classic novel from the 1800s", "a famous observatory",
    ],
}
PROMPT = ("Write an informative, encyclopedia-style passage of about {words} words about {topic}. "
          "Pick a specific subject, use plain prose without headings or lists.")


def ngrams(text, n):
    tokens = re.findall(r"\w+", text.lower())
    return {" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def rwku_target_names(data_root):
    names = set()
    for path in (rwku_dir(data_root) / "wiki_forget_set").glob("*.json"):
        names.add(path.stem.replace("_", " ").lower())
        for row in json.loads(path.read_text(encoding="utf-8")):
            if isinstance(row, dict) and row.get("target"):
                names.add(str(row["target"]).lower())
    return names


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--forget-dataset", default="wmdp", choices=["wmdp", "rwku"])
    parser.add_argument("--wmdp-subset", default="cyber")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--num-texts", type=int, default=200)
    parser.add_argument("--words", type=int, default=250)
    parser.add_argument("--api-model", default="gpt-4o-mini")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--ngram", type=int, default=8, help="Drop texts sharing an n-gram with the forget corpus")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from openai import OpenAI

    client = OpenAI(base_url=args.base_url) if args.base_url else OpenAI()
    output = args.output or relearn_data_path(args.data_root, args.forget_dataset)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.forget_dataset == "wmdp":
        forget = load_wmdp_corpus(f"{args.wmdp_subset}-forget-corpus", 0)
        banned_names = set()
    else:
        forget = load_rwku_forget_corpus(args.data_root, 0)
        banned_names = rwku_target_names(args.data_root)
    forget_ngrams = set().union(*(ngrams(text, args.ngram) for text in forget))

    rng, kept, dropped = random.Random(args.seed), 0, 0
    with open(output, "w", encoding="utf-8") as handle:
        while kept < args.num_texts:
            if dropped > 10 * args.num_texts:
                raise RuntimeError("Too many generated texts overlap the forget set; adjust topics or --ngram.")
            topic = rng.choice(TOPICS[args.forget_dataset])
            reply = client.chat.completions.create(
                model=args.api_model, temperature=args.temperature,
                messages=[{"role": "user", "content": PROMPT.format(words=args.words, topic=topic)}])
            text = (reply.choices[0].message.content or "").strip()
            lowered = text.lower()
            if not text or any(name in lowered for name in banned_names) or ngrams(text, args.ngram) & forget_ngrams:
                dropped += 1
                continue
            handle.write(json.dumps({"text": text, "topic": topic}) + "\n")
            kept += 1
    print(f"wrote {kept} texts to {output} (dropped {dropped} overlapping)")


if __name__ == "__main__":
    main()
