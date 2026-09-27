from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from app.config import get_settings
from app.inference.runner import run_forecast


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def next_run_time(now: datetime | None = None) -> datetime:
    settings = get_settings()
    zone = ZoneInfo(settings.APP_TIMEZONE)
    local_now = (now or datetime.now(timezone.utc)).astimezone(zone)
    candidate = local_now.replace(minute=settings.FORECAST_RUN_MINUTE, second=0, microsecond=0)
    if candidate <= local_now:
        candidate += timedelta(hours=1)
    return candidate.astimezone(timezone.utc)


def run_forever() -> None:
    while True:
        run_at = next_run_time()
        delay = max((run_at - datetime.now(timezone.utc)).total_seconds(), 0.0)
        logging.info("Next forecast run at %s", run_at.isoformat())
        time.sleep(delay)
        result = run_forecast()
        logging.info("Forecast run result: %s", json.dumps(result, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Hourly gas forecast worker.")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--cutoff", help="Optional ISO-8601 data cutoff for a one-shot run.")
    args = parser.parse_args()
    if args.once:
        cutoff = datetime.fromisoformat(args.cutoff) if args.cutoff else None
        if cutoff is not None and cutoff.tzinfo is None:
            cutoff = cutoff.replace(tzinfo=timezone.utc)
        print(json.dumps(run_forecast(cutoff), ensure_ascii=False, indent=2))
    else:
        run_forever()


if __name__ == "__main__":
    main()
