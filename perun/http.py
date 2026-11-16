import asyncio
import json
from pathlib import Path

import aiohttp

from .config import USER_AGENT

RETRY_STATUSES = {429, 500, 502, 503, 504}


async def _get(session: aiohttp.ClientSession, url: str, sem: asyncio.Semaphore,
               retries: int = 5) -> bytes | None:
    delay = 2.0
    for attempt in range(retries):
        async with sem:
            try:
                async with session.get(url) as r:
                    if r.status == 404:
                        return None
                    if r.status in RETRY_STATUSES:
                        raise aiohttp.ClientResponseError(r.request_info, (), status=r.status)
                    r.raise_for_status()
                    return await r.read()
            except (aiohttp.ClientError, asyncio.TimeoutError):
                if attempt == retries - 1:
                    raise
        await asyncio.sleep(delay)
        delay *= 2
    return None


async def fetch_many(urls: list[str], concurrency: int = 4,
                     timeout: float = 60) -> dict[str, bytes | None]:
    sem = asyncio.Semaphore(concurrency)
    headers = {"User-Agent": USER_AGENT}
    tmo = aiohttp.ClientTimeout(total=timeout)
    async with aiohttp.ClientSession(headers=headers, timeout=tmo) as s:
        bodies = await asyncio.gather(*(_get(s, u, sem) for u in urls))
    return dict(zip(urls, bodies))


def fetch_to(url: str, dest: Path, force: bool = False) -> Path:
    if dest.exists() and not force:
        return dest
    body = asyncio.run(fetch_many([url], concurrency=1, timeout=300))[url]
    if body is None:
        raise FileNotFoundError(url)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(body)
    return dest


def load_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)
