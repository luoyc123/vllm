#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request


def request_json(
    url: str,
    *,
    api_key: str | None = None,
    body: dict | None = None,
    timeout: float = 30.0,
) -> dict:
    headers = {"Accept": "application/json"}
    data = None
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if body is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, headers=headers, data=data)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise SystemExit(f"HTTP {error.code}: {detail}") from error


def main() -> None:
    parser = argparse.ArgumentParser(description="Call the single HXinfer API")
    parser.add_argument(
        "--base-url",
        default=os.getenv("HXINFER_API_BASE", "http://127.0.0.1:8000"),
    )
    parser.add_argument("--api-key", default=os.getenv("HXINFER_API_KEY"))
    parser.add_argument("--model", default="Qwen2.5-3B-Instruct")
    parser.add_argument("--prompt", default="请只回答数字：12乘以3等于多少？")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Repeat the request to keep both PP stages active during monitoring",
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    if args.repeat <= 0:
        parser.error("--repeat must be positive")

    base_url = args.base_url.rstrip("/")
    health = request_json(f"{base_url}/health", timeout=args.timeout)
    if health.get("status") != "ok":
        raise SystemExit(f"unexpected health response: {health}")

    print(f"health: {health['status']}")
    for index in range(1, args.repeat + 1):
        response = request_json(
            f"{base_url}/v1/chat/completions",
            api_key=args.api_key,
            timeout=args.timeout,
            body={
                "model": args.model,
                "messages": [{"role": "user", "content": args.prompt}],
                "temperature": 0,
                "max_tokens": args.max_tokens,
            },
        )
        print(
            f"request {index}/{args.repeat} assistant: "
            f"{response['choices'][0]['message']['content']}"
        )
        print(f"usage: {json.dumps(response['usage'], ensure_ascii=False)}")


if __name__ == "__main__":
    main()
