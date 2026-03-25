"""
Shared API utilities: rate limiting, retries, and client creation.
"""

import time
import os
from functools import wraps
from dotenv import load_dotenv
from anthropic import Anthropic, RateLimitError, APIStatusError

load_dotenv()

# Default delay between API calls (seconds).
# Adjust based on your tier: Free=12s (5 RPM), Build=1.2s (50 RPM), Scale=0s
DEFAULT_CALL_DELAY = 1.5


def get_client() -> Anthropic:
    """Create an Anthropic client from env var."""
    return Anthropic()


def api_call_with_retry(fn, *args, max_retries=5, base_delay=DEFAULT_CALL_DELAY, **kwargs):
    """
    Call an API function with rate-limit handling.

    - Waits `base_delay` seconds before each call
    - On 429 (rate limit): exponential backoff with jitter
    - On other API errors: retry up to max_retries
    - On parse errors: retry once (model might give different output)
    """
    time.sleep(base_delay)

    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except RateLimitError as e:
            wait = (2 ** attempt) * 5  # 5s, 10s, 20s, 40s, 80s
            print(f"  Rate limited (attempt {attempt + 1}/{max_retries}). "
                  f"Waiting {wait}s...")
            time.sleep(wait)
        except APIStatusError as e:
            if e.status_code >= 500:
                wait = (2 ** attempt) * 3
                print(f"  Server error {e.status_code} (attempt {attempt + 1}/{max_retries}). "
                      f"Waiting {wait}s...")
                time.sleep(wait)
            else:
                raise
        except (ValueError, KeyError) as e:
            # Parse error — retry once in case model gives different output
            if attempt == 0:
                print(f"  Parse error: {e}. Retrying once...")
                time.sleep(base_delay)
            else:
                raise

    raise RuntimeError(f"Failed after {max_retries} retries")
