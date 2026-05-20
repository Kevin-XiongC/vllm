# SPDX-License-Identifier: Apache-2.0
"""Compare batched and individual prompt logprobs under MLA chunked prefill."""

import argparse
from itertools import islice

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from vllm import LLM, SamplingParams


def calc_mse(a: list[float], b: list[float]) -> float:
    return (
        (torch.tensor(a, dtype=torch.float32) - torch.tensor(b, dtype=torch.float32))
        .pow(2)
        .mean()
        .item()
    )


def get_prompt_logprobs(llm: LLM, input_ids: list[int]) -> list[float]:
    outputs = llm.generate(
        [{"prompt_token_ids": input_ids}],
        SamplingParams(
            max_tokens=1,
            temperature=0.0,
            prompt_logprobs=1,
            detokenize=False,
        ),
        use_tqdm=False,
    )
    prompt_logprobs = outputs[0].prompt_logprobs
    assert prompt_logprobs is not None
    return [
        prompt_logprobs[i][token_id].logprob
        for i, token_id in enumerate(input_ids[1:], start=1)
    ]


def get_batched_prompt_logprobs(
    llm: LLM,
    all_input_ids: list[list[int]],
) -> list[list[float]]:
    outputs = llm.generate(
        [{"prompt_token_ids": input_ids} for input_ids in all_input_ids],
        [
            SamplingParams(
                max_tokens=1,
                temperature=0.0,
                prompt_logprobs=1,
                detokenize=False,
            )
            for _ in all_input_ids
        ],
        use_tqdm=False,
    )
    all_logprobs: list[list[float]] = []
    for output, input_ids in zip(outputs, all_input_ids):
        prompt_logprobs = output.prompt_logprobs
        assert prompt_logprobs is not None
        all_logprobs.append(
            [
                prompt_logprobs[i][token_id].logprob
                for i, token_id in enumerate(input_ids[1:], start=1)
            ]
        )
    return all_logprobs


def make_synthetic_tokens(
    model: str,
    num_docs: int,
    max_tokens: int,
    chunked_prefill_size: int,
) -> list[list[int]]:
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    base_text = (
        "The history of distributed inference systems includes scheduling, "
        "prefix caching, attention kernels, and network transport. "
    )
    all_tokens = []
    for i in range(num_docs):
        target_len = min(max_tokens + 1, chunked_prefill_size * (i % 3 + 1) + 257)
        text = base_text * max(1, target_len // 20)
        token_ids = tokenizer(text)["input_ids"][:target_len]
        all_tokens.append(token_ids)
    return all_tokens


def load_wikipedia_tokens(
    model: str,
    num_docs: int,
    max_tokens: int,
) -> list[list[int]]:
    print(f"Loading {num_docs} wikipedia docs...")
    dataset = load_dataset(
        "wikimedia/wikipedia",
        "20231101.en",
        split="train",
        streaming=True,
    )
    docs = list(islice(dataset, num_docs))

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    all_tokens = []
    for row in docs:
        token_ids = tokenizer(row["text"])["input_ids"]
        if len(token_ids) > max_tokens + 1:
            token_ids = token_ids[: max_tokens + 1]
        all_tokens.append(token_ids)
    return all_tokens


def make_llm(args: argparse.Namespace) -> LLM:
    attention_config = {"mla_prefill_backend": args.mla_prefill_backend}
    if args.attention_backend is not None:
        attention_config["backend"] = args.attention_backend

    return LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        trust_remote_code=True,
        enforce_eager=args.enforce_eager,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.chunked_prefill_size,
        enable_chunked_prefill=True,
        kv_cache_dtype=args.kv_cache_dtype,
        attention_config=attention_config,
        compilation_config={"cudagraph_mode": "NONE"},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--num-docs", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=20000)
    parser.add_argument("--chunked-prefill-size", type=int, default=8192)
    parser.add_argument(
        "--dataset",
        choices=("wikipedia", "synthetic"),
        default="wikipedia",
    )
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--kv-cache-dtype", default="fp8_e4m3")
    parser.add_argument("--attention-backend", default="FLASHINFER_MLA")
    parser.add_argument("--mla-prefill-backend", default="TRTLLM_RAGGED")
    parser.add_argument("--enforce-eager", action="store_true")
    args = parser.parse_args()

    if args.dataset == "wikipedia":
        all_tokens = load_wikipedia_tokens(args.model, args.num_docs, args.max_tokens)
    else:
        all_tokens = make_synthetic_tokens(
            args.model,
            args.num_docs,
            args.max_tokens,
            args.chunked_prefill_size,
        )
    print(f"Token lengths: {[len(t) for t in all_tokens]}")

    llm = make_llm(args)

    print("\nRunning individual requests...")
    individual_lps = []
    for tokens in all_tokens:
        individual_lps.append(get_prompt_logprobs(llm, tokens))

    print("Running batched request...")
    batched_lps = get_batched_prompt_logprobs(llm, all_tokens)

    print(
        f"\n=== {args.model} | "
        f"mla_prefill_backend={args.mla_prefill_backend} | "
        f"chunked_prefill_size={args.chunked_prefill_size} ==="
    )
    max_mse = 0.0
    for i in range(args.num_docs):
        mse = calc_mse(individual_lps[i], batched_lps[i])
        max_mse = max(max_mse, mse)
        print(f"  Doc {i} ({len(all_tokens[i]):>6} tokens): MSE = {mse:.6f}")
    print(f"Max MSE: {max_mse:.6f}")

    if hasattr(llm, "shutdown"):
        llm.shutdown()


if __name__ == "__main__":
    main()
