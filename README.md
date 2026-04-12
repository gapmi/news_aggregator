# News Aggregator

Динамический сборщик новостей из RSS-лент и HTML-страниц.

## Быстрый старт

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

Результат — файл `news_output.json`.

## Как добавить новый источник

### RSS
В `config.py` добавь в `rss_sources`:
```python
RSSSource(name="My Feed", url="https://example.com/rss.xml"),
```

### HTML
В `config.py` добавь в `html_sources`:
```python
HTMLSource(
    name="My Site",
    url="https://example.com/news",
    article_selector=".news-item",
    title_selector="h2 a",
    link_selector="h2 a",
    description_selector="p.summary",
),
```

## Архитектура

- `scrapers/` — сборщики (RSS, HTML)
- `processors/` — обработка (дедупликация)
- `storage/` — хранение (JSON, расширяемо до PostgreSQL/MongoDB)
- `config.py` — конфигурация источников
- `main.py` — точка входа
