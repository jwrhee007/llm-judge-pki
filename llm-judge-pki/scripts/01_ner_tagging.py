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
from src.data.sampler import stratified_sample, save_sampled_data
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
    for item in data:
        messages = build_ner_messages(
            question=item["question"],
            answer=item["answer_value"],
        )
        requests.append({
            "custom_id": f"ner_{item['question_id']}",
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
    id_to_tag = {}
    for result in results:
        custom_id = result["custom_id"]
        question_id = custom_id.replace("ner_", "", 1)

        response_body = result.get("response", {}).get("body", {})
        choices = response_body.get("choices", [])
        if choices:
            response_text = choices[0].get("message", {}).get("content", "")
            ner_tag = parse_ner_response(response_text)
        else:
            ner_tag = "UNKNOWN"

        id_to_tag[question_id] = ner_tag

    # Merge back
    for item in data:
        item["ner_tag"] = id_to_tag.get(item["question_id"], "UNKNOWN")

    return data


def main():
    parser = argparse.ArgumentParser(description="NER tagging and stratified sampling")
    parser.add_argument(
        "--input",
        default="data/processed/triviaqa_rc_evidence_present.jsonl",
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
        "--config",
        default="configs/config.yaml",
        help="Config file path",
    )
    args = parser.parse_args()

    load_dotenv()
    config = load_config(args.config)

    logger.info("=" * 60)
    logger.info("Step 1: NER Tagging + Stratified Sampling")
    logger.info("=" * 60)

    # Load data
    data = load_processed_data(args.input)

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

    # NER tagging
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
    tag_dist = Counter(item["ner_tag"] for item in tagged_data)
    logger.info("NER tag distribution (full dataset):")
    for tag, count in tag_dist.most_common():
        logger.info(f"  {tag}: {count}")

    # Stratified sampling
    sampled = stratified_sample(
        data=tagged_data,
        ner_tag_key="ner_tag",
        max_per_tag=config["sampling"]["max_per_ner_tag"],
        target_total=config["sampling"]["target_total"],
        seed=config["sampling"]["random_seed"],
        target_tags=config["sampling"]["target_ner_tags"],
    )

    save_sampled_data(sampled, args.output)

    logger.info(f"Done! {len(sampled)} sampled items saved to {args.output}")


if __name__ == "__main__":
    main()
