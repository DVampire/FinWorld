import os
import httpx
import openai
import contextlib
from dotenv import load_dotenv

load_dotenv(verbose=True)

PROXY_URL = os.getenv('LOCAL_PROXY_BASE', None)

if PROXY_URL:
    HTTP_CLIENT = httpx.Client(proxy=PROXY_URL, timeout=httpx.Timeout(600.0, connect=60.0))
    ASYNC_HTTP_CLIENT = httpx.AsyncClient(proxy=PROXY_URL, timeout=httpx.Timeout(600.0, connect=60.0))
else:
    HTTP_CLIENT = httpx.Client()
    ASYNC_HTTP_CLIENT = httpx.AsyncClient()

@contextlib.contextmanager
def proxy_env(proxy_url: str = PROXY_URL):
    os.environ["HTTP_PROXY"] = proxy_url
    os.environ["HTTPS_PROXY"] = proxy_url
    try:
        yield
    finally:
        del os.environ["HTTP_PROXY"]
        del os.environ["HTTPS_PROXY"]

__all__ = [
    "PROXY_URL",
    "HTTP_CLIENT",
    "ASYNC_HTTP_CLIENT",
]