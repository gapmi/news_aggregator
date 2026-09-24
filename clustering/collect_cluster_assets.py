from __future__ import annotations

import asyncio
import csv
import io
import re
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
from PIL import Image, UnidentifiedImageError
from playwright.async_api import Browser, Page, async_playwright


INPUT_CSV = Path("cluster_centroids.csv")
OUTPUT_DIR = Path("test_assets")
LOG_PATH = OUTPUT_DIR / "log.csv"

HTTP_TIMEOUT_SECONDS = 30.0
PAGE_TIMEOUT_MS = 30_000
PAGE_RENDER_WAIT_MS = 2_000

MIN_IMAGE_BYTES = 15_000
MIN_IMAGE_WIDTH = 640
MIN_IMAGE_HEIGHT = 360
MIN_ASPECT_RATIO = 0.80
MAX_ASPECT_RATIO = 2.50

SCREENSHOT_BLOCKLIST = {
    "bloomberg.com",
}


def clean_url(value: str | None, base_url: str | None = None) -> str | None:
    if not value:
        return None

    value = value.strip()

    if not value or value.startswith(("data:", "javascript:")):
        return None

    if base_url:
        value = urljoin(base_url, value)

    if value.startswith(("http://", "https://")):
        return value

    return None


def hostname(url: str) -> str:
    return urlparse(url).hostname.lower() if urlparse(url).hostname else ""


def is_blocked_for_screenshot(article_url: str) -> bool:
    host = hostname(article_url)

    return any(
        host == blocked_host or host.endswith(f".{blocked_host}")
        for blocked_host in SCREENSHOT_BLOCKLIST
    )


def image_extension(content_type: str | None, source_url: str) -> str:
    if content_type:
        mime_type = content_type.lower().split(";", 1)[0].strip()
        mime_extensions = {
            "image/jpeg": ".jpg",
            "image/jpg": ".jpg",
            "image/png": ".png",
            "image/webp": ".webp",
            "image/gif": ".gif",
            "image/avif": ".avif",
        }
        if mime_type in mime_extensions:
            return mime_extensions[mime_type]

    url_suffix = Path(urlparse(source_url).path).suffix.lower()
    if url_suffix in {".jpg", ".jpeg", ".png", ".webp", ".gif", ".avif"}:
        return url_suffix

    return ".jpg"


def dedupe_candidates(
    candidates: Iterable[tuple[str, str]],
) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    seen: set[str] = set()

    for source_type, image_url in candidates:
        normalized = image_url.strip()

        if normalized not in seen:
            seen.add(normalized)
            result.append((source_type, normalized))

    return result


def extract_rss_images(rss_html: str | None) -> list[str]:
    if not rss_html:
        return []

    soup = BeautifulSoup(rss_html, "html.parser")
    urls: list[str] = []

    for tag in soup.select("img[src], source[src]"):
        image_url = clean_url(tag.get("src"))
        if image_url:
            urls.append(image_url)

    return urls


def extract_page_image_candidates(
    html: str,
    page_url: str,
) -> list[tuple[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    candidates: list[tuple[str, str]] = []

    meta_selectors = [
        ("og_image", 'meta[property="og:image"]'),
        ("og_image", 'meta[property="og:image:url"]'),
        ("twitter_image", 'meta[name="twitter:image"]'),
        ("twitter_image", 'meta[name="twitter:image:src"]'),
    ]

    for source_type, selector in meta_selectors:
        tag = soup.select_one(selector)
        content = tag.get("content") if tag else None
        image_url = clean_url(content, page_url)

        if image_url:
            candidates.append((source_type, image_url))

    content_root = soup.select_one("article") or soup.select_one("main") or soup

    for tag in content_root.select("img[src], source[src]"):
        image_url = clean_url(tag.get("src"), page_url)

        if not image_url:
            continue

        width = tag.get("width")
        height = tag.get("height")

        try:
            if width and int(width) < 300:
                continue
            if height and int(height) < 180:
                continue
        except ValueError:
            pass

        lowered_url = image_url.lower()
        forbidden_markers = (
            "logo",
            "icon",
            "avatar",
            "sprite",
            "banner",
            "advert",
            "adserver",
        )

        if any(marker in lowered_url for marker in forbidden_markers):
            continue

        candidates.append(("article_image", image_url))

    return dedupe_candidates(candidates)


async def download_and_validate_image(
    client: httpx.AsyncClient,
    image_url: str,
    source_type: str,
    output_stem: Path,
) -> dict[str, Any]:
    try:
        response = await client.get(image_url)
        response.raise_for_status()

        content_type = response.headers.get("content-type", "")
        if not content_type.lower().startswith("image/"):
            return {
                "status": "rejected",
                "error": f"Unexpected content type: {content_type or 'missing'}",
                "asset_source": source_type,
                "asset_url": image_url,
                "width": None,
                "height": None,
                "size_bytes": len(response.content),
                "asset_path": "",
            }

        content = response.content

        if len(content) < MIN_IMAGE_BYTES:
            return {
                "status": "rejected",
                "error": f"Image is too small: {len(content)} bytes",
                "asset_source": source_type,
                "asset_url": image_url,
                "width": None,
                "height": None,
                "size_bytes": len(content),
                "asset_path": "",
            }

        try:
            with Image.open(io.BytesIO(content)) as image:
                width, height = image.size
                image.verify()
        except (UnidentifiedImageError, OSError) as exc:
            return {
                "status": "rejected",
                "error": f"Invalid image: {exc}",
                "asset_source": source_type,
                "asset_url": image_url,
                "width": None,
                "height": None,
                "size_bytes": len(content),
                "asset_path": "",
            }

        if width < MIN_IMAGE_WIDTH or height < MIN_IMAGE_HEIGHT:
            return {
                "status": "rejected",
                "error": f"Image dimensions are too small: {width}x{height}",
                "asset_source": source_type,
                "asset_url": image_url,
                "width": width,
                "height": height,
                "size_bytes": len(content),
                "asset_path": "",
            }

        aspect_ratio = width / height
        if not MIN_ASPECT_RATIO <= aspect_ratio <= MAX_ASPECT_RATIO:
            return {
                "status": "rejected",
                "error": (
                    f"Unsupported aspect ratio: {aspect_ratio:.3f} "
                    f"for {width}x{height}"
                ),
                "asset_source": source_type,
                "asset_url": image_url,
                "width": width,
                "height": height,
                "size_bytes": len(content),
                "asset_path": "",
            }

        suffix = image_extension(content_type, image_url)
        output_path = output_stem.with_suffix(suffix)
        output_path.write_bytes(content)

        return {
            "status": "ok",
            "error": "",
            "asset_source": source_type,
            "asset_url": image_url,
            "width": width,
            "height": height,
            "size_bytes": len(content),
            "asset_path": str(output_path),
        }

    except httpx.HTTPError as exc:
        return {
            "status": "error",
            "error": f"Image download failed: {exc}",
            "asset_source": source_type,
            "asset_url": image_url,
            "width": None,
            "height": None,
            "size_bytes": None,
            "asset_path": "",
        }


async def load_article_html(
    client: httpx.AsyncClient,
    article_url: str,
) -> tuple[str | None, str | None, str | None]:
    try:
        response = await client.get(
            article_url,
            headers={"Accept": "text/html,application/xhtml+xml"},
        )
        response.raise_for_status()

        content_type = response.headers.get("content-type", "")
        if "html" not in content_type.lower():
            return None, str(response.url), f"Unexpected page content type: {content_type}"

        return response.text, str(response.url), None

    except httpx.HTTPError as exc:
        return None, None, f"Article HTML download failed: {exc}"


async def screenshot_viewport(
    browser: Browser,
    article_url: str,
    output_path: Path,
) -> dict[str, Any]:
    page: Page | None = None

    try:
        page = await browser.new_page(
            viewport={"width": 1440, "height": 1080},
            device_scale_factor=1,
        )

        await page.goto(
            article_url,
            wait_until="domcontentloaded",
            timeout=PAGE_TIMEOUT_MS,
        )
        await page.wait_for_timeout(PAGE_RENDER_WAIT_MS)

        await page.screenshot(
            path=str(output_path),
            full_page=False,
        )

        with Image.open(output_path) as image:
            width, height = image.size

        return {
            "status": "ok",
            "error": "",
            "asset_source": "viewport_screenshot",
            "asset_url": article_url,
            "width": width,
            "height": height,
            "size_bytes": output_path.stat().st_size,
            "asset_path": str(output_path),
        }

    except Exception as exc:
        return {
            "status": "error",
            "error": f"Screenshot failed: {exc}",
            "asset_source": "viewport_screenshot",
            "asset_url": article_url,
            "width": None,
            "height": None,
            "size_bytes": None,
            "asset_path": "",
        }

    finally:
        if page:
            await page.close()


def make_output_stem(cluster_id: str, article_id: str) -> Path:
    return OUTPUT_DIR / f"cluster_{cluster_id}_article_{article_id}"


async def collect_asset(
    browser: Browser,
    client: httpx.AsyncClient,
    row: dict[str, str],
) -> dict[str, Any]:
    cluster_id = row["cluster_id"]
    article_id = row["article_id"]
    article_url = row["article_url"]
    output_stem = make_output_stem(cluster_id, article_id)

    candidates: list[tuple[str, str]] = [
        ("rss_image", image_url)
        for image_url in extract_rss_images(row.get("rss_description"))
    ]

    html, final_url, page_error = await load_article_html(client, article_url)

    if html and final_url:
        candidates.extend(extract_page_image_candidates(html, final_url))

    candidates = dedupe_candidates(candidates)
    rejected_errors: list[str] = []

    for source_type, image_url in candidates:
        image_result = await download_and_validate_image(
            client=client,
            image_url=image_url,
            source_type=source_type,
            output_stem=output_stem,
        )

        if image_result["status"] == "ok":
            return {
                **row,
                "asset_type": "image",
                **image_result,
            }

        rejected_errors.append(
            f"{source_type}: {image_result.get('error', 'rejected')}"
        )

    if is_blocked_for_screenshot(article_url):
        details = "; ".join(rejected_errors)

        return {
            **row,
            "asset_type": "none",
            "status": "blocked",
            "asset_source": "screenshot_blocklist",
            "asset_url": article_url,
            "asset_path": "",
            "width": None,
            "height": None,
            "size_bytes": None,
            "error": (
                "Screenshot skipped for blocked domain"
                + (f"; {details}" if details else "")
            ),
        }

    screenshot_path = output_stem.with_name(
        f"{output_stem.name}_viewport"
    ).with_suffix(".png")

    screenshot_result = await screenshot_viewport(
        browser=browser,
        article_url=article_url,
        output_path=screenshot_path,
    )

    errors = [error for error in [page_error, *rejected_errors] if error]
    if screenshot_result["error"]:
        errors.append(screenshot_result["error"])

    return {
        **row,
        "asset_type": (
            "screenshot"
            if screenshot_result["status"] == "ok"
            else "none"
        ),
        **screenshot_result,
        "error": "; ".join(errors),
    }


def read_rows() -> list[dict[str, str]]:
    if not INPUT_CSV.exists():
        print(
            f"Файл {INPUT_CSV} не найден.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    with INPUT_CSV.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        print(
            f"Файл {INPUT_CSV} не содержит записей.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    return rows


async def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = read_rows()

    timeout = httpx.Timeout(HTTP_TIMEOUT_SECONDS)
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/128.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9,ru;q=0.8",
    }

    results: list[dict[str, Any]] = []

    async with httpx.AsyncClient(
        timeout=timeout,
        follow_redirects=True,
        headers=headers,
    ) as client:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(headless=True)

            try:
                for index, row in enumerate(rows, start=1):
                    print(
                        f"[{index}/{len(rows)}] "
                        f"cluster={row['cluster_id']} "
                        f"article={row['article_id']}"
                    )
                    result = await collect_asset(browser, client, row)
                    results.append(result)

            finally:
                await browser.close()

    fieldnames = [
        "cluster_id",
        "run_id",
        "cluster_size",
        "representative_article_id",
        "representative_title",
        "article_id",
        "article_title",
        "article_url",
        "source",
        "published",
        "asset_type",
        "status",
        "asset_source",
        "asset_url",
        "asset_path",
        "width",
        "height",
        "size_bytes",
        "error",
    ]

    with LOG_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(results)

    summary: dict[str, int] = {}
    for result in results:
        key = f"{result['asset_type']}:{result['status']}"
        summary[key] = summary.get(key, 0) + 1

    print()
    print(f"Готово. Лог: {LOG_PATH}")
    print("Сводка:", summary)


if __name__ == "__main__":
    asyncio.run(main())