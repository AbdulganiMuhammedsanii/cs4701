"""Download FEVER wiki-pages and resolve (page_title, sentence_id) -> text."""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import zipfile
from pathlib import Path
from typing import Iterator

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

WIKI_ZIP_URL = "https://fever.ai/download/fever/wiki-pages.zip"


def find_wiki_jsonl_root(search_root: Path) -> Path | None:
    """Return directory containing wiki-*.jsonl shards (any depth under search_root)."""
    for p in sorted(search_root.rglob("wiki-*.jsonl")):
        return p.parent
    return None


def download_wiki_pages(raw_dir: Path, force: bool = False) -> Path:
    """Download and extract official FEVER wiki-pages; return folder with wiki-*.jsonl."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "wiki-pages.zip"
    found = find_wiki_jsonl_root(raw_dir)
    if found is not None and not force:
        return found

    if not zip_path.is_file() or force:
        logger.info("Downloading wiki-pages.zip (~1.7 GB, one-time)…")
        with requests.get(WIKI_ZIP_URL, stream=True, timeout=300) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(zip_path, "wb") as f, tqdm(
                total=total, unit="B", unit_scale=True, desc="wiki-pages.zip"
            ) as pbar:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

    logger.info("Extracting wiki-pages.zip…")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(raw_dir)

    found = find_wiki_jsonl_root(raw_dir)
    if found is None:
        raise FileNotFoundError(
            f"Could not find wiki-*.jsonl under {raw_dir} after extract"
        )
    return found


def _iter_wiki_records(wiki_dir: Path) -> Iterator[tuple[str, dict[int, str]]]:
    """Yield (normalized_title, {sent_id: sentence_text}) from all wiki-*.jsonl files."""
    files = sorted(wiki_dir.glob("wiki-*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No wiki-*.jsonl under {wiki_dir}")

    for path in files:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                title = obj.get("id") or obj.get("title") or ""
                if not title:
                    continue
                norm = _normalize_title(title)
                sentences: dict[int, str] = {}
                raw_lines = obj.get("lines") or obj.get("text") or ""
                if isinstance(raw_lines, list):
                    for i, s in enumerate(raw_lines):
                        if isinstance(s, str) and s.strip():
                            sentences[i] = s.strip()
                elif isinstance(raw_lines, str):
                    for part in raw_lines.split("\n"):
                        part = part.strip()
                        if not part:
                            continue
                        m = re.match(r"^(\d+)\t(.*)$", part, re.DOTALL)
                        if m:
                            sentences[int(m.group(1))] = m.group(2).strip()
                        else:
                            sentences[len(sentences)] = part
                if sentences:
                    yield norm, sentences


def _normalize_title(title: str) -> str:
    t = title.replace(" ", "_").strip()
    return t


def normalize_evidence_title(wiki_url_field: str) -> str:
    """Normalize FEVER `evidence_wiki_url` to wiki page key used in the dump."""
    return _normalize_title(wiki_url_field.replace("/wiki/", "").split("/")[-1])


class WikiSentenceIndex:
    """SQLite-backed index: (page_title, sentence_id) -> sentence text."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)

    def build_from_wiki_dir(
        self,
        wiki_dir: Path,
        rebuild: bool = False,
        only_titles: set[str] | None = None,
    ) -> None:
        """
        Build SQLite index from extracted wiki-*.jsonl.

        If `only_titles` is set, only pages whose normalized title is in the set
        are indexed (fast for small FEVER subsets).
        """
        wiki_dir = Path(wiki_dir)
        if self.db_path.is_file() and not rebuild:
            logger.info("Using existing wiki index %s", self.db_path)
            return

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        if self.db_path.is_file():
            self.db_path.unlink()
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "CREATE TABLE sentences (title TEXT NOT NULL, sent_id INTEGER NOT NULL, "
            "text TEXT NOT NULL, PRIMARY KEY (title, sent_id))"
        )
        conn.execute("CREATE INDEX idx_title ON sentences(title)")
        conn.commit()
        batch: list[tuple[str, int, str]] = []
        batch_size = 5000
        desc = "Indexing wiki (subset)" if only_titles else "Indexing wiki (full)"

        for title, sents in tqdm(_iter_wiki_records(wiki_dir), desc=desc):
            if only_titles is not None and title not in only_titles:
                continue
            for sid, text in sents.items():
                batch.append((title, int(sid), text))
                if len(batch) >= batch_size:
                    conn.executemany(
                        "INSERT OR REPLACE INTO sentences VALUES (?,?,?)", batch
                    )
                    conn.commit()
                    batch.clear()
        if batch:
            conn.executemany("INSERT OR REPLACE INTO sentences VALUES (?,?,?)", batch)
            conn.commit()
        conn.close()
        logger.info("Wiki index built at %s", self.db_path)

    def lookup(self, wiki_url_field: str, sentence_id: int) -> str | None:
        title = _normalize_title(wiki_url_field.replace("/wiki/", "").split("/")[-1])
        conn = sqlite3.connect(self.db_path)
        cur = conn.execute(
            "SELECT text FROM sentences WHERE title = ? AND sent_id = ?",
            (title, int(sentence_id)),
        )
        row = cur.fetchone()
        conn.close()
        if row:
            return row[0]
        return None
