#!/usr/bin/env python3
"""Send a short experiment update to Telegram using environment credentials.

Required environment variables are intentionally not stored in this repository:
``TELEGRAM_CHAT_ID`` and ``TELEGRAM_BOT_TOKEN``.
"""

from __future__ import annotations

import argparse
import json
import os
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def send_message(message: str, timeout: float = 15.0) -> bool:
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not chat_id or not token:
        return False

    payload = urlencode({"chat_id": chat_id, "text": message}).encode("utf-8")
    request = Request(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data=payload,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
        return bool(result.get("ok"))
    except Exception:
        # Notification failure must never stop a scientific experiment.
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("message")
    args = parser.parse_args()
    return 0 if send_message(args.message) else 1


if __name__ == "__main__":
    raise SystemExit(main())
