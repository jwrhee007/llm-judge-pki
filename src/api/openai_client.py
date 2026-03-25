"""OpenAI API client with retry logic and rate limiting."""

import json
import time
from pathlib import Path
from typing import Optional

from openai import OpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class OpenAIClient:
    """Wrapper around OpenAI API with retry, rate limiting, and batch support."""

    def __init__(
        self,
        api_key: str,
        max_retries: int = 5,
        retry_min_wait: float = 1.0,
        retry_max_wait: float = 60.0,
        requests_per_minute: int = 500,
    ):
        self.client = OpenAI(api_key=api_key)
        self.max_retries = max_retries
        self.retry_min_wait = retry_min_wait
        self.retry_max_wait = retry_max_wait
        self.requests_per_minute = requests_per_minute
        self._last_request_time = 0.0
        self._min_interval = 60.0 / requests_per_minute

    def _rate_limit(self):
        """Enforce minimum interval between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_time = time.time()

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        retry=retry_if_exception_type((Exception,)),
        before_sleep=lambda retry_state: logger.warning(
            f"Retry {retry_state.attempt_number} after error: {retry_state.outcome.exception()}"
        ),
    )
    def chat_completion(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0,
        max_tokens: int = 256,
        seed: Optional[int] = None,
    ) -> str:
        """Send a chat completion request and return the response text."""
        self._rate_limit()

        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if seed is not None:
            kwargs["seed"] = seed

        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content.strip()

    # -----------------------------------------------------------------
    # Batch API helpers
    # -----------------------------------------------------------------

    def create_batch_file(
        self,
        requests: list[dict],
        output_path: str,
    ) -> str:
        """
        Write a JSONL file for the OpenAI Batch API.

        Each element in `requests` should have:
          - custom_id: str
          - model: str
          - messages: list[dict]
          - temperature: float (optional)
          - max_tokens: int (optional)
          - seed: int (optional)

        Returns the path to the written JSONL file.
        """
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        with open(output, "w", encoding="utf-8") as f:
            for req in requests:
                body = {
                    "model": req["model"],
                    "messages": req["messages"],
                    "temperature": req.get("temperature", 0),
                    "max_tokens": req.get("max_tokens", 256),
                }
                if "seed" in req and req["seed"] is not None:
                    body["seed"] = req["seed"]

                line = {
                    "custom_id": req["custom_id"],
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": body,
                }
                f.write(json.dumps(line, ensure_ascii=False) + "\n")

        logger.info(f"Batch file written: {output} ({len(requests)} requests)")
        return str(output)

    def submit_batch(self, jsonl_path: str, description: str = "") -> str:
        """Upload a JSONL file and submit a batch job. Returns batch ID."""
        with open(jsonl_path, "rb") as f:
            file_obj = self.client.files.create(file=f, purpose="batch")

        batch = self.client.batches.create(
            input_file_id=file_obj.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": description},
        )
        logger.info(f"Batch submitted: {batch.id} (file: {file_obj.id})")
        return batch.id

    def poll_batch(self, batch_id: str, poll_interval: int = 30) -> dict:
        """Poll a batch job until completion. Returns the batch object."""
        while True:
            batch = self.client.batches.retrieve(batch_id)
            status = batch.status
            logger.info(
                f"Batch {batch_id}: {status} "
                f"(completed: {batch.request_counts.completed}/"
                f"{batch.request_counts.total})"
            )
            if status in ("completed", "failed", "expired", "cancelled"):
                return batch
            time.sleep(poll_interval)

    def download_batch_results(self, batch_id: str, output_path: str) -> list[dict]:
        """Download and parse batch results. Returns list of result dicts."""
        batch = self.client.batches.retrieve(batch_id)
        if batch.status != "completed":
            raise RuntimeError(f"Batch {batch_id} status: {batch.status}")

        content = self.client.files.content(batch.output_file_id)
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(content.content)

        results = []
        with open(output, "r", encoding="utf-8") as f:
            for line in f:
                results.append(json.loads(line))

        logger.info(f"Downloaded {len(results)} results to {output}")
        return results
