import time
import random
import logging
import requests
from functools import wraps

# config.config 모듈이 프로젝트 루트에 있다고 가정합니다.
# 실제 환경에 따라 경로 조정이 필요할 수 있습니다.
try:
    from config.config import RETRY_CONFIG
except ImportError:
    # Fallback if running in a context where config.config is not directly importable
    # This might happen during isolated testing of the decorator or if PYTHONPATH is not set up.
    # For production, ensure config.config is accessible.
    print("Warning: Could not import RETRY_CONFIG from config.config. Using default retry settings.")
    RETRY_CONFIG = {
        "max_retry_attempts": 3,
        "initial_backoff_seconds": 1,
        "max_backoff_seconds": 10,
        "jitter": True,
        "retry_http_status_codes": [500, 502, 503, 504],
    }

logger = logging.getLogger(__name__)

def retry_on_request_exception(func):
    """
    A decorator to retry a function call upon encountering requests.exceptions.RequestException
    (including HTTPError for specific status codes) or its subclasses.

    Uses settings from RETRY_CONFIG for max attempts, backoff strategy, and jitter.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        attempts = 0
        current_backoff = RETRY_CONFIG.get("initial_backoff_seconds", 1)
        max_attempts = RETRY_CONFIG.get("max_retry_attempts", 3)
        max_backoff = RETRY_CONFIG.get("max_backoff_seconds", 60)
        use_jitter = RETRY_CONFIG.get("jitter", True)
        retry_status_codes = RETRY_CONFIG.get("retry_http_status_codes", [500, 502, 503, 504])
        
        last_exception = None

        while attempts < max_attempts:
            attempts += 1
            try:
                logger.debug(f"Attempt {attempts}/{max_attempts} for {func.__name__} with args: {args}, kwargs: {kwargs}")
                return func(*args, **kwargs)
            except requests.exceptions.HTTPError as e:
                last_exception = e
                status_code = e.response.status_code if e.response is not None else None
                if status_code in retry_status_codes:
                    logger.warning(
                        f"HTTPError with status {status_code} on attempt {attempts}/{max_attempts} for {func.__name__}. Retrying..."
                    )
                    # Proceed to backoff and retry
                else:
                    logger.error(
                        f"HTTPError with status {status_code} on attempt {attempts}/{max_attempts} for {func.__name__}. Not a retriable HTTP status. Raising."
                    )
                    raise
            except requests.exceptions.RequestException as e:  # Catches ConnectionError, Timeout, and custom subclasses
                last_exception = e
                logger.warning(
                    f"RequestException ({type(e).__name__}) on attempt {attempts}/{max_attempts} for {func.__name__}: {str(e)}. Retrying..."
                )
                # Proceed to backoff and retry
            except Exception as e:
                last_exception = e
                logger.error(
                    f"Unexpected error ({type(e).__name__}) on attempt {attempts}/{max_attempts} for {func.__name__}: {str(e)}. Not retrying."
                )
                raise # Re-raise immediately for non-RequestExceptions or non-HTTP retriable errors

            if attempts < max_attempts:
                wait_time = current_backoff
                if use_jitter:
                    # Add jitter: random value between 0 and 10% of current_backoff
                    jitter_amount = random.uniform(0, current_backoff * 0.1)
                    wait_time += jitter_amount
                
                wait_time = min(wait_time, max_backoff) # Cap wait time at max_backoff

                logger.info(f"Waiting {wait_time:.2f} seconds before next retry for {func.__name__} (attempt {attempts+1}/{max_attempts}).")
                time.sleep(wait_time)

                # Exponential backoff, capped at max_backoff
                current_backoff = min(current_backoff * 2, max_backoff)
            else:
                logger.error(
                    f"All {max_attempts} retry attempts failed for {func.__name__}. Last error: {type(last_exception).__name__}: {str(last_exception)}"
                )
                if last_exception:
                    raise last_exception
                # This part should ideally not be reached if an exception was always caught
                raise RuntimeError(f"All {max_attempts} retry attempts failed for {func.__name__} without a specific captured exception.")
        
        # This line should be unreachable if logic is correct and max_attempts > 0
        return None 
    return wrapper
