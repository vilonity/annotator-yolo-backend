from urllib.parse import urlparse

from fastapi import Request


def _get_origin(request: Request) -> str | None:
    origin = request.headers.get("origin") or request.headers.get("referer", "")
    if not origin:
        return None
    parsed = urlparse(origin)
    return f"{parsed.scheme}://{parsed.netloc}"


def resolve_relative_url(url: str, request: Request) -> str:
    if url.startswith("/"):
        base = _get_origin(request)
        if base:
            return f"{base}{url}"
    return url


def resolve_relative_urls(urls: list[str], request: Request) -> list[str]:
    base = _get_origin(request)
    if not base:
        return urls
    return [f"{base}{url}" if url.startswith("/") else url for url in urls]
