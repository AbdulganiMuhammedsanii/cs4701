"""Join FEVER rows with wiki sentence text and write processed JSONL."""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

from tqdm import tqdm

from halludet.data.fever import fever_row_to_internal_label, load_fever_split
from halludet.data.wiki_index import (
    WikiSentenceIndex,
    download_wiki_pages,
    find_wiki_jsonl_root,
    normalize_evidence_title,
)

logger = logging.getLogger(__name__)


def _titles_for_subset(split: str, max_samples: int | None) -> set[str]:
    ds = load_fever_split(split)
    n = len(ds) if max_samples is None else min(max_samples, len(ds))
    titles: set[str] = set()
    for i in range(n):
        w = ds[i]["evidence_wiki_url"]
        titles.add(normalize_evidence_title(str(w)))
    return titles


def preprocess_split(
    split: str,
    raw_dir: Path,
    processed_dir: Path,
    wiki_index: WikiSentenceIndex,
    max_samples: int | None = None,
    skip_missing_evidence: bool = True,
    nei_seed: int = 0,
) -> Path:
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / f"fever_{split}.jsonl"
    ds = load_fever_split(split)
    n = len(ds) if max_samples is None else min(max_samples, len(ds))

    rng = random.Random(nei_seed)
    kept = 0
    skipped = 0
    nei_kept = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for i in tqdm(range(n), desc=f"preprocess {split}"):
            row = ds[i]
            label_raw = row["label"]
            if label_raw not in ("SUPPORTS", "REFUTES", "NOT ENOUGH INFO"):
                skipped += 1
                continue
            claim = row["claim"]
            if label_raw == "NOT ENOUGH INFO":
                text = wiki_index.random_sentence(rng)
                if text is None:
                    skipped += 1
                    continue
                nei_kept += 1
            else:
                wiki = row["evidence_wiki_url"]
                sid = row["evidence_sentence_id"]
                text = wiki_index.lookup(wiki, sid)
                if text is None:
                    skipped += 1
                    if skip_missing_evidence:
                        continue
                    text = ""
            rec = {
                "id": row["id"],
                "claim": claim,
                "evidence": text,
                "fever_label": label_raw,
                "label": fever_row_to_internal_label(label_raw),
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept += 1
    logger.info("NEI rows kept with distractor evidence: %d", nei_kept)

    logger.info(
        "Wrote %s (%d rows kept, %d skipped)", out_path, kept, skipped
    )
    return out_path


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(description="Preprocess FEVER with wiki evidence text")
    p.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory for wiki_pages.zip and extracted wiki",
    )
    p.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed"),
        help="Output directory for JSONL",
    )
    p.add_argument(
        "--splits",
        nargs="+",
        default=["paper_dev", "labelled_dev"],
        help="FEVER splits to export (train is large)",
    )
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument(
        "--rebuild-wiki-index",
        action="store_true",
        help="Rebuild SQLite wiki index",
    )
    p.add_argument(
        "--full-wiki-index",
        action="store_true",
        help="Index all wiki pages (slow); default is subset pages for --max-samples or full split",
    )
    p.add_argument(
        "--skip-wiki-download",
        action="store_true",
        help="Assume wiki_pages already extracted under raw-dir",
    )
    args = p.parse_args()

    if not args.skip_wiki_download:
        wiki_dir = download_wiki_pages(args.raw_dir)
    else:
        wiki_dir = find_wiki_jsonl_root(args.raw_dir)
        if wiki_dir is None:
            wiki_dir = args.raw_dir

    db_path = args.processed_dir / "wiki_sentences.sqlite"
    index = WikiSentenceIndex(db_path)

    if args.full_wiki_index:
        index.build_from_wiki_dir(wiki_dir, rebuild=args.rebuild_wiki_index)
    else:
        subset_titles: set[str] = set()
        for sp in args.splits:
            subset_titles |= _titles_for_subset(sp, args.max_samples)
        if args.rebuild_wiki_index and db_path.is_file():
            db_path.unlink()
        index.build_from_wiki_dir(
            wiki_dir,
            rebuild=args.rebuild_wiki_index or not db_path.is_file(),
            only_titles=subset_titles,
        )

    for sp in args.splits:
        preprocess_split(
            sp,
            args.raw_dir,
            args.processed_dir,
            index,
            max_samples=args.max_samples,
        )


if __name__ == "__main__":
    main()
