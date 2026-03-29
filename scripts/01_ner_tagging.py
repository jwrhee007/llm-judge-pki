#!/usr/bin/env python3
"""
Step 1: NER 태깅 + 층화 추출.

1. evidence_present 데이터에 NER 태깅 (GPT-4o, Lee et al. Figure 6 프롬프트)
2. NER 태그별 층화 추출 (20개/태그 상한)
3. 결과를 data/processed/triviaqa_rc_sampled.jsonl로 저장

Usage:
    python scripts/01_ner_tagging.py
    python scripts/01_ner_tagging.py --use-batch    # Batch API 사용
    python scripts/01_ner_tagging.py --input data/processed/triviaqa_rc_evidence_present.jsonl
"""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.api.openai_client import OpenAIClient
from src.data.sampler import (
    random_sample,
    stratified_sample,
    posthoc_ner_summary,
    assign_analysis_tag,
    save_sampled_data,
)
from src.data.triviaqa_loader import load_processed_data
from src.prompts.ner_prompt import build_ner_messages, parse_ner_response
from src.utils.logger import setup_logger

logger = setup_logger("ner_tagging")


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_ner_sync(
    client: OpenAIClient,
    data: list[dict],
    model: str,
    temperature: float = 0,
    max_tokens: int = 16,
) -> list[dict]:
    """Synchronously tag all items with NER."""
    tagged = []
    for item in tqdm(data, desc="NER Tagging"):
        messages = build_ner_messages(
            question=item["question"],
            answer=item["answer_value"],
        )
        try:
            response = client.chat_completion(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            ner_tag = parse_ner_response(response)
        except Exception as e:
            logger.warning(f"NER failed for {item['question_id']}: {e}")
            ner_tag = "UNKNOWN"

        item["ner_tag"] = ner_tag
        tagged.append(item)

    return tagged


def run_ner_batch(
    client: OpenAIClient,
    data: list[dict],
    model: str,
    temperature: float = 0,
    max_tokens: int = 16,
    batch_output_dir: str = "data/batch_outputs",
) -> list[dict]:
    """Run NER tagging via Batch API."""
    # Prepare batch requests
    requests = []
    for data_idx, item in enumerate(data):
        messages = build_ner_messages(
            question=item["question"],
            answer=item["answer_value"],
        )
        requests.append({
            "custom_id": f"ner_idx{data_idx}",
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        })

    # Write batch file
    batch_file = client.create_batch_file(
        requests=requests,
        output_path=f"{batch_output_dir}/ner_batch_input.jsonl",
    )

    # Submit and poll
    batch_id = client.submit_batch(batch_file, description="NER Tagging")
    batch = client.poll_batch(batch_id, poll_interval=15)

    if batch.status != "completed":
        logger.error(f"Batch failed: {batch.status}")
        raise RuntimeError(f"Batch failed with status: {batch.status}")

    # Download results
    results = client.download_batch_results(
        batch_id=batch_id,
        output_path=f"{batch_output_dir}/ner_batch_output.jsonl",
    )

    # Parse results
    import re
    pattern = re.compile(r"^ner_idx(\d+)$")
    for result in results:
        m = pattern.match(result["custom_id"])
        if not m:
            continue
        data_idx = int(m.group(1))
        if data_idx >= len(data):
            continue

        response_body = result.get("response", {}).get("body", {})
        choices = response_body.get("choices", [])
        if choices:
            response_text = choices[0].get("message", {}).get("content", "")
            ner_tag = parse_ner_response(response_text)
        else:
            ner_tag = "UNKNOWN"

        data[data_idx]["ner_tag"] = ner_tag

    # Fill any missing
    for item in data:
        if "ner_tag" not in item:
            item["ner_tag"] = "UNKNOWN"

    return data


def main():
    parser = argparse.ArgumentParser(description="NER tagging and sampling")
    parser.add_argument(
        "--input",
        default="data/processed/triviaqa_rc_curated.jsonl",
        help="Input data file",
    )
    parser.add_argument(
        "--output",
        default="data/processed/triviaqa_rc_sampled.jsonl",
        help="Output sampled data file",
    )
    parser.add_argument(
        "--ner-output",
        default="data/processed/triviaqa_rc_ner_tagged.jsonl",
        help="NER-tagged (pre-sampling) output file",
    )
    parser.add_argument(
        "--use-batch",
        action="store_true",
        help="Use OpenAI Batch API instead of sync calls",
    )
    parser.add_argument(
        "--skip-ner",
        action="store_true",
        help="Skip NER tagging (use pre-tagged data from --input)",
    )
    parser.add_argument(
        "--sampling-strategy",
        choices=["random", "stratified"],
        default=None,
        help="Sampling strategy (default: from config)",
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Config file path",
    )
    args = parser.parse_args()

    load_dotenv()
    config = load_config(args.config)
    strategy = args.sampling_strategy or config["sampling"]["strategy"]

    logger.info("=" * 60)
    logger.info(f"Step 1: NER Tagging + Sampling (strategy={strategy})")
    logger.info("=" * 60)

    # Load data
    data = load_processed_data(args.input)

    # NER tagging (skip if --skip-ner or input is already NER-tagged)
    if args.skip_ner:
        logger.info("Skipping NER tagging (--skip-ner)")
        tagged_data = data
    else:
        # Initialize client
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.error("OPENAI_API_KEY not found in environment")
            sys.exit(1)

        client = OpenAIClient(
            api_key=api_key,
            max_retries=config["api"]["max_retries"],
            requests_per_minute=config["api"]["requests_per_minute"],
        )

        ner_model = config["model"]["ner_tagger"]
        logger.info(f"NER model: {ner_model}")

        if args.use_batch:
            logger.info("Using Batch API for NER tagging")
            tagged_data = run_ner_batch(
                client=client,
                data=data,
                model=ner_model,
                batch_output_dir=config["api"]["batch_output_dir"],
            )
        else:
            logger.info("Using synchronous API for NER tagging")
            tagged_data = run_ner_sync(
                client=client,
                data=data,
                model=ner_model,
            )

        # Save NER-tagged data (before sampling)
        output_path = Path(args.ner_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for item in tagged_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        logger.info(f"NER-tagged data saved to {args.ner_output}")

    # NER tag distribution
    from collections import Counter
    tag_dist = Counter(item.get("ner_tag", "UNKNOWN") for item in tagged_data)
    logger.info("NER tag distribution (full dataset):")
    for tag, count in tag_dist.most_common():
        logger.info(f"  {tag}: {count}")

    # Sampling
    if strategy == "random":
        logger.info(f"Using random sampling (target={config['sampling']['target_total']})")
        sampled = random_sample(
            data=tagged_data,
            target_total=config["sampling"]["target_total"],
            seed=config["sampling"]["random_seed"],
        )

        # Post-hoc NER analysis: assign analysis tags
        min_count = config["sampling"].get("min_per_ner_tag_for_analysis", 15)
        ner_summary = posthoc_ner_summary(sampled, min_count=min_count)
        sampled = assign_analysis_tag(sampled, ner_summary["tag_mapping"])

        # Save post-hoc NER summary
        summary_path = Path(args.output).parent / "posthoc_ner_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(ner_summary, f, indent=2, ensure_ascii=False)
        logger.info(f"Post-hoc NER summary saved to {summary_path}")

    elif strategy == "stratified":
        logger.info(f"Using stratified sampling (max_per_tag={config['sampling']['max_per_ner_tag']})")
        sampled = stratified_sample(
            data=tagged_data,
            ner_tag_key="ner_tag",
            max_per_tag=config["sampling"]["max_per_ner_tag"],
            target_total=config["sampling"]["target_total"],
            seed=config["sampling"]["random_seed"],
            target_tags=config["sampling"]["target_ner_tags"],
        )
    else:
        logger.error(f"Unknown sampling strategy: {strategy}")
        sys.exit(1)

    save_sampled_data(sampled, args.output)

    logger.info(f"Done! {len(sampled)} sampled items saved to {args.output}")


if __name__ == "__main__":
    main()