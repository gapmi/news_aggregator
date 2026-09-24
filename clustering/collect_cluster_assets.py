import asyncio
import csv
import os
import re
import sys
from pathlib import Path

import httpx
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

# === Конфиг ===
DB_URL = os.getenv("DATABASE_URL", "postgresql://user:pass@localhost:5432/news_db")
OUTPUT_DIR = Path("test_assets")
OUTPUT_DIR.mkdir(exist_ok=True)

QUERY = """
SELECT c.id AS cluster_id,
       c.representative_article_id,
       a.id AS article_id,
       a.title AS article_title,
       a.url AS article_url,
       a.rss_description
FROM clusters c
JOIN articles a ON a.id = c.representative_article_id
WHERE c.run_id = (
    SELECT cr.id
    FROM clustering_runs cr
    ORDER BY cr.started_at DESC
    LIMIT 1
)
LIMIT 10;
"""

# === Парсинг картинки из rss_description ===
def extract_image_from_rss(rss_html: str | None) -> str | None:
    if not rss_html:
        return None
    soup = BeautifulSoup(rss_html, "html.parser")
    img = soup.find("img", src=True)
    if img and img.get("src"):
        return img["src"]
    return None

# === Скачивание картинки ===
async def download_image(url: str, out_path: Path) -> dict:
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        try:
            resp = await client.get(url)
            resp.raise_for_status()
            out_path.write_bytes(resp.content)
            # Можно добавить проверку размера, но для теста хватит факта скачивания
            return {"status": "ok", "width": None, "height": None}
        except Exception as e:
            return {"status": "error", "error": str(e), "width": None, "height": None}

# === Скриншот страницы ===
async def screenshot_page(url: str, out_path: Path) -> dict:
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        try:
            await page.goto(url, wait_until="networkidle", timeout=30000)
            await page.screenshot(path=str(out_path), full_page=True)
            return {"status": "ok"}
        except Exception as e:
            return {"status": "error", "error": str(e)}
        finally:
            await browser.close()

# === Основной пайплайн ===
async def main():
    # Для простоты: данные передадим через stdin или захардкодим после запроса в БД
    # Здесь я ожидаю, что ты сначала выгрузишь данные в CSV, а скрипт прочитает его.
    # Альтернатива — подключить psycopg и делать запрос прямо отсюда.
    
    # Вариант 1: читаем из stdin (ты сделаешь: psql ... -c "COPY (...)" TO stdout | python collect_cluster_assets.py)
    # Вариант 2 (проще для старта): я дам отдельный скрипт для выгрузки, а этот будет читать CSV.
    
    # Для примера — заглушка: ожидаем файл cluster_centroids.csv
    input_csv = Path("cluster_centroids.csv")
    if not input_csv.exists():
        print(f"Файл {input_csv} не найден. Сначала выгрузи данные из БД.", file=sys.stderr)
        sys.exit(1)

    log_rows = []

    with input_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cluster_id = row["cluster_id"]
            article_url = row["article_url"]
            rss_description = row["rss_description"]

            img_url = extract_image_from_rss(rss_description)
            asset_type = None
            asset_path = None
            status = None
            width = height = None

            if img_url:
                # Скачиваем картинку
                ext = Path(img_url.split("?")[0]).suffix or ".jpg"
                out_file = OUTPUT_DIR / f"cluster_{cluster_id}_image{ext}"
                res = await download_image(img_url, out_file)
                asset_type = "image"
                asset_path = str(out_file)
                status = res["status"]
                if status == "ok":
                    # Можно добавить Pillow для получения размеров
                    pass
            else:
                # Делаем скриншот
                out_file = OUTPUT_DIR / f"cluster_{cluster_id}_screenshot.png"
                res = await screenshot_page(article_url, out_file)
                asset_type = "screenshot"
                asset_path = str(out_file)
                status = res["status"]

            log_rows.append({
                "cluster_id": cluster_id,
                "article_url": article_url,
                "asset_type": asset_type,
                "asset_path": asset_path,
                "status": status,
            })

    # Лог
    log_file = OUTPUT_DIR / "log.csv"
    with log_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["cluster_id", "article_url", "asset_type", "asset_path", "status"])
        writer.writeheader()
        writer.writerows(log_rows)

    print(f"Готово. Лог в {log_file}")

if __name__ == "__main__":
    asyncio.run(main())