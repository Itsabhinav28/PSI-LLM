#!/usr/bin/env python3
import os
import sys
import json
import tarfile
import gzip
import shutil
import zipfile
from pathlib import Path
from typing import List, Dict, Any
import logging

# Ensure project imports
sys.path.append(str(Path(__file__).parent))

from competition_rag_system import CompetitionRAGRetriever
from src.vector_store.chroma_store import ChromaStore

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MOCK_DIR = Path("mock data")
OUTPUT_DIR = Path("submission_output")
STARTUP_NAME = "GreedyGeeks"


def _rebuild_tar_gz_from_parts(parts: List[Path], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Rebuilding archive from parts: {[p.name for p in parts]} -> {output_path}")
    with open(output_path, "wb") as w:
        for p in parts:
            with open(p, "rb") as r:
                shutil.copyfileobj(r, w)
    return output_path


def _extract_any_archive(archive_path: Path, extract_to: Path) -> Path:
    extract_to.mkdir(parents=True, exist_ok=True)
    logger.info(f"Extracting {archive_path} -> {extract_to}")
    # Try gzip tar
    try:
        with tarfile.open(archive_path, mode="r:gz") as tar:
            tar.extractall(path=extract_to)
            return extract_to
    except Exception as e_gz:
        logger.warning(f"r:gz failed: {e_gz}")
    # Try plain tar
    try:
        with tarfile.open(archive_path, mode="r:") as tar:
            tar.extractall(path=extract_to)
            return extract_to
    except Exception as e_tar:
        logger.warning(f"r: failed: {e_tar}")
    # Try zip
    try:
        with zipfile.ZipFile(archive_path, 'r') as z:
            z.extractall(extract_to)
            return extract_to
    except Exception as e_zip:
        logger.warning(f"zip extract failed: {e_zip}")
    raise RuntimeError(f"Unsupported or corrupted archive: {archive_path}")


def _collect_corpus_and_queries(root: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Heuristically find queries and corpus files inside extracted dataset.
    Expecting folders like queries/ and corpus/ if present; otherwise try to infer.
    """
    queries: List[Dict[str, Any]] = []
    corpus_files: List[Path] = []

    # Heuristic: look for common names
    for sub in root.rglob("*"):
        if sub.is_file():
            lname = sub.name.lower()
            if lname.endswith((".txt", ".md", ".html", ".pdf", ".json")):
                if "query" in lname or "queries" in lname:
                    # Try load as JSON list of objects or line-delimited queries
                    try:
                        with open(sub, "r", encoding="utf-8", errors="ignore") as f:
                            content = f.read().strip()
                        if content.startswith("["):
                            data = json.loads(content)
                            # Normalize: each item may be str or object
                            for i, item in enumerate(data, 1):
                                if isinstance(item, str):
                                    queries.append({"id": i, "query": item})
                                elif isinstance(item, dict) and "query" in item:
                                    qid = item.get("id") or item.get("qid") or i
                                    queries.append({"id": int(qid), "query": item["query"]})
                        else:
                            # One query per line
                            for i, line in enumerate(content.splitlines(), 1):
                                line = line.strip()
                                if line:
                                    queries.append({"id": i, "query": line})
                        logger.info(f"Loaded queries from {sub}")
                    except Exception:
                        pass
                else:
                    corpus_files.append(sub)

    # If queries empty, fallback to a default single query
    if not queries:
        queries = [{"id": 1, "query": "What are the key topics in the dataset?"}]
        logger.warning("No queries found in dataset; using a default placeholder query.")

    return {"queries": queries, "corpus_files": corpus_files}


def _index_corpus_into_competition_retriever(corpus_files: List[Path]) -> CompetitionRAGRetriever:
    retriever = CompetitionRAGRetriever()
    documents: List[Dict[str, Any]] = []

    for fp in corpus_files:
        try:
            if fp.suffix.lower() in {".txt", ".md"}:
                text = fp.read_text(encoding="utf-8", errors="ignore")
            else:
                # Fallback: store filename as minimal content; system still returns file identifiers
                text = f"Document placeholder for {fp.name}"
            documents.append({
                "id": fp.stem,
                "text": text,
                "source_file": fp.name,
                "path": str(fp)
            })
        except Exception as e:
            logger.warning(f"Skipping {fp}: {e}")

    if documents:
        retriever.index_documents(documents)
    else:
        logger.warning("No corpus texts were indexed; retrieval results may be empty.")
    return retriever


def _write_per_query_json(output_dir: Path, query_id: int, query_text: str, top_files: List[str]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "query": query_text,
        "response": top_files
    }
    out_path = output_dir / f"{query_id}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    return out_path


def _zip_submission(output_dir: Path, startup_name: str) -> Path:
    zip_path = output_dir.parent / f"{startup_name}_PS4.zip"
    logger.info(f"Creating submission zip: {zip_path}")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for fp in sorted(output_dir.glob("*.json")):
            z.write(fp, arcname=fp.name)
    return zip_path


def main():
    # 1) Rebuild archive from parts
    parts = sorted(list(MOCK_DIR.glob("train.tar.gz.part-*")))
    if not parts:
        logger.error(f"No part files found in {MOCK_DIR}; please place mock dataset parts there.")
        sys.exit(1)

    rebuilt = Path("build/mock_train.tar.gz")
    rebuilt.parent.mkdir(parents=True, exist_ok=True)
    rebuilt = _rebuild_tar_gz_from_parts(parts, rebuilt)

    # 2) Extract (robust), or fall back to scanning mock folder directly
    try:
        extracted_root = _extract_any_archive(rebuilt, Path("build/extracted"))
        scan_root = extracted_root
    except Exception as e:
        logger.error(f"Archive extraction failed, falling back to scanning mock folder directly: {e}")
        scan_root = MOCK_DIR

    # 3) Collect queries and corpus
    discovered = _collect_corpus_and_queries(scan_root)
    queries = discovered["queries"]
    corpus_files = discovered["corpus_files"]
    logger.info(f"Discovered {len(queries)} queries and {len(corpus_files)} corpus files")

    # 4) Index corpus in competition retriever; if none found, fallback to ChromaStore
    retriever = None
    if corpus_files:
        retriever = _index_corpus_into_competition_retriever(corpus_files)
    else:
        logger.info("No corpus files discovered in mock data; falling back to ChromaStore documents.")
        try:
            store = ChromaStore()
            chroma_docs = store.get_all_documents()
            docs: List[Dict[str, Any]] = []
            for i, d in enumerate(chroma_docs):
                # Support both dict-like and object-like docs
                text = getattr(d, 'page_content', None)
                meta = getattr(d, 'metadata', None)
                if text is None and isinstance(d, dict):
                    text = d.get('text') or d.get('content')
                    meta = d.get('metadata')
                if not text:
                    continue
                source_file = None
                path = None
                if isinstance(meta, dict):
                    source_file = meta.get('source') or meta.get('source_file')
                    path = meta.get('path')
                docs.append({
                    "id": f"chroma_{i}",
                    "text": text,
                    "source_file": source_file or f"chroma_{i}.txt",
                    "path": path or f"chroma:{i}"
                })
            retriever = CompetitionRAGRetriever()
            if docs:
                retriever.index_documents(docs)
            else:
                logger.warning("ChromaStore returned no documents; retrieval may be empty.")
        except Exception as e:
            logger.error(f"Failed to load from ChromaStore: {e}")
            retriever = CompetitionRAGRetriever()

    # 5) Run retrieval per query and write numbered JSON
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    generated: List[Path] = []
    no_index = False
    try:
        no_index = not getattr(retriever, 'documents', [])
    except Exception:
        no_index = False
    for q in queries:
        qid = int(q["id"])
        qtext = str(q["query"]).strip()
        top_files = []
        if not no_index:
            try:
                results = retriever.retrieve(qtext, k=100, final_k=10)
                for item in results:
                    doc = item.get("document") or {}
                    fname = doc.get("source_file") or doc.get("path") or doc.get("id") or "unknown.txt"
                    top_files.append(fname)
            except Exception as e:
                logger.warning(f"Retrieval failed for qid={qid}: {e}; writing empty response.")
        # de-duplicate while preserving order
        seen = set()
        top_files = [f for f in top_files if not (f in seen or seen.add(f))][:10]
        generated.append(_write_per_query_json(OUTPUT_DIR, qid, qtext, top_files))

    # 6) Zip them
    zip_path = _zip_submission(OUTPUT_DIR, STARTUP_NAME)
    logger.info(f"Submission ready: {zip_path}")


if __name__ == "__main__":
    main()
