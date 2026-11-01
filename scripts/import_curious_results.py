"""Import saved curious-story assessments. Dry-run by default; no HTTP or LLM calls."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import psycopg2
from psycopg2.extras import Json, RealDictCursor


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(body):
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def prepare(root, assessment_run):
    summary = load(assessment_run / "summary.json")
    if summary.get("mode") != "execute":
        raise ValueError("Assessment run must have mode=execute")
    version = summary.get("assessment_prompt_version")
    if not isinstance(version, str) or not version:
        raise ValueError("Missing prompt version")
    stories = load(assessment_run / "stories.json")
    candidates = {r["id"]: r for r in load(root / "candidates.json")}
    prepared, seen = [], set()
    skipped = []
    for record in stories:
        row = record["article"]
        aid = row["id"]
        if record.get("assessment_status") != "assessed":
            skipped.append(aid)
            continue
        if type(aid) is not int or aid in seen:
            raise ValueError(f"Invalid or duplicate article ID: {aid}")
        seen.add(aid)
        if aid not in candidates or candidates[aid]["url"] != row["url"]:
            raise ValueError(f"Candidate mismatch: {aid}")
        if record.get("assessment_prompt_version") != version:
            raise ValueError(f"Prompt version mismatch: {aid}")
        assessment = record["assessment"]
        if type(assessment.get("suitable")) is not bool or type(assessment.get("text_sufficient")) is not bool:
            raise ValueError(f"Invalid assessment flags: {aid}")
        if not isinstance(assessment.get("story_title"), str) or not assessment["story_title"].strip():
            raise ValueError(f"Missing story title: {aid}")
        matches = []
        for request_path in sorted((root / "cache_v2").glob("assess_*_request.json")):
            request = load(request_path)
            payload = json.loads(request["messages"][-1]["content"])
            if payload.get("article_id") != aid:
                continue
            response_path = request_path.with_name(request_path.name.replace("_request.json", "_response.json"))
            if not response_path.exists():
                continue
            raw = load(response_path)
            choice = raw["choices"][0]
            if choice.get("finish_reason") != "stop":
                continue
            if json.loads(choice["message"]["content"]) != assessment:
                continue
            fingerprint = digest(request)
            if request_path.name != f"assess_{fingerprint}_request.json":
                raise ValueError(f"Request fingerprint mismatch: {aid}")
            if request.get("model") != summary.get("model") or payload.get("url") != row["url"]:
                raise ValueError(f"Model or URL mismatch: {aid}")
            text = payload.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Empty request text: {aid}")
            matches.append((request, raw, text, fingerprint))
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one matching cached assessment for {aid}, found {len(matches)}")
        request, raw, text, fingerprint = matches[0]
        extraction = None
        for path in (root / "extraction_test" / f"article_{aid}.json", root / f"article_{aid}.json", root / "cache_v2" / f"article_{aid}.json"):
            if path.exists():
                candidate = load(path)
                e = candidate.get("extraction", {})
                if candidate.get("article", {}).get("url") == row["url"] and e.get("text") == text and e.get("status") == "extracted_unverified":
                    extraction = e
                    break
        if extraction is None:
            raise ValueError(f"No extraction matching assessed text: {aid}")
        prepared.append({"article": row, "text": text, "text_sha256": hashlib.sha256(text.encode()).hexdigest(), "extractor_version": "legacy_trafilatura_version_unrecorded", "extraction": extraction, "assessment": assessment, "raw_response": raw, "requested_model": request["model"], "prompt_version": version, "request_sha256": fingerprint})
    if not prepared:
        raise ValueError("No assessed articles to import")
    return prepared, skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--assessment-run", required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    root, result = Path(args.run_dir), Path(args.assessment_run)
    if not result.resolve().is_relative_to(root.resolve()):
        parser.error("assessment-run must be inside run-dir")
    prepared, skipped = prepare(root, result)
    password = os.getenv("DB_PASSWORD")
    if not password:
        raise RuntimeError("DB_PASSWORD must be set")
    stats = {"mode": "execute" if args.execute else "dry_run", "prepared": len(prepared), "skipped_unassessed": skipped, "new_texts": 0, "new_assessments": 0, "new_seed_stories": 0, "new_links": 0, "api_calls": 0, "downloads": 0, "items": []}
    with psycopg2.connect(host=os.getenv("DB_HOST", "db"), port=os.getenv("DB_PORT", "5432"), dbname=os.getenv("DB_NAME", "news_db"), user=os.getenv("DB_USER", "postgres"), password=password, connect_timeout=15) as conn:
        if not args.execute:
            conn.set_session(readonly=True)
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SET LOCAL lock_timeout = '5s'")
            cur.execute("SET LOCAL statement_timeout = '30s'")
            if args.execute:
                cur.execute("SELECT pg_advisory_xact_lock(74125, 1)")
            for item in prepared:
                row, a = item["article"], item["assessment"]
                aid = row["id"]
                cur.execute("SELECT id,url FROM public.articles WHERE id=%s", (aid,))
                db_article = cur.fetchone()
                if not db_article or db_article["url"] != row["url"]:
                    raise ValueError(f"Database article URL mismatch or missing article: {aid}")
                cur.execute("SELECT id,full_text FROM public.curious_article_texts WHERE article_id=%s AND content_sha256=%s AND extractor_version=%s", (aid,item["text_sha256"],item["extractor_version"]))
                existing_text = cur.fetchone()
                if existing_text and existing_text["full_text"] != item["text"]:
                    raise ValueError(f"Text hash/content mismatch: {aid}")
                cur.execute("SELECT id,article_text_id,assessment FROM public.curious_article_assessments WHERE article_id=%s AND request_sha256=%s", (aid,item["request_sha256"]))
                existing_assessment = cur.fetchone()
                if existing_assessment and (not existing_text or existing_assessment["article_text_id"] != existing_text["id"] or existing_assessment["assessment"] != a):
                    raise ValueError(f"Existing assessment mismatch: {aid}")
                cur.execute("SELECT id,status FROM public.curious_stories WHERE seed_article_id=%s", (aid,))
                existing_story = cur.fetchone()
                make_seed = a["suitable"] and a["text_sufficient"]
                link_exists = False
                if existing_story and make_seed:
                    cur.execute("SELECT relation_type FROM public.curious_story_articles WHERE story_id=%s AND article_id=%s", (existing_story["id"],aid))
                    link = cur.fetchone()
                    if link and link["relation_type"] != "seed":
                        raise ValueError(f"Existing seed link mismatch: {aid}")
                    link_exists = link is not None
                stats["new_texts"] += int(existing_text is None)
                stats["new_assessments"] += int(existing_assessment is None)
                stats["new_seed_stories"] += int(make_seed and existing_story is None)
                stats["new_links"] += int(make_seed and not link_exists)
                text_id = existing_text["id"] if existing_text else None
                assessment_id = existing_assessment["id"] if existing_assessment else None
                story_id = existing_story["id"] if existing_story else None
                if args.execute:
                    if text_id is None:
                        e = item["extraction"]
                        metadata = {"legacy_import": True,"original_fetched_at_unknown": True,"extractor_version_unrecorded": True,"extraction_status": e["status"],"import_run": str(root)}
                        cur.execute("""INSERT INTO public.curious_article_texts
                            (article_id,content_sha256,extractor_version,full_text,original_url,final_url,extracted_title,extracted_date_raw,metadata)
                            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id""",
                            (aid,item["text_sha256"],item["extractor_version"],item["text"],row["url"],e.get("final_url"),e.get("extracted_title"),e.get("extracted_date"),Json(metadata)))
                        text_id = cur.fetchone()["id"]
                    if assessment_id is None:
                        raw = item["raw_response"]
                        cur.execute("""INSERT INTO public.curious_article_assessments
                            (article_id,article_text_id,request_sha256,prompt_version,requested_model,returned_model,assessment,raw_response,usage,source_snapshot)
                            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id""",
                            (aid,text_id,item["request_sha256"],item["prompt_version"],item["requested_model"],raw.get("model"),Json(a),Json(raw),Json(raw.get("usage") or {}),Json(row)))
                        assessment_id = cur.fetchone()["id"]
                    if make_seed and story_id is None:
                        details = {"matching_status": "not_matched", "provisional_seed": True,"assessment_id": assessment_id,"story_type": a["story_type"],"recommended_tone": a["recommended_tone"],"import_assessment_run": str(result)}
                        cur.execute("""INSERT INTO public.curious_stories
                            (seed_article_id,title,summary,status,event_details)
                            VALUES (%s,%s,%s,'candidate',%s) RETURNING id""",
                            (aid,a["story_title"],a.get("summary", ""),Json(details)))
                        story_id = cur.fetchone()["id"]
                    if make_seed and not link_exists:
                        cur.execute("""INSERT INTO public.curious_story_articles
                            (story_id,article_id,relation_type,match_method,match_reason)
                            VALUES (%s,%s,'seed','seed_import','Provisional seed; cross-source matching has not run')""", (story_id,aid))
                stats["items"].append({"article_id": aid,"text_id": text_id,"assessment_id": assessment_id,"story_id": story_id,"seed_candidate": make_seed})
    stats["committed"] = bool(args.execute)
    stats["note"] = "Dry-run counts are planned inserts, not writes" if not args.execute else "Committed atomically; seed stories remain candidates, not approved events"
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()