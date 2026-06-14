import gc
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Optional

import numpy as np
from botocore.exceptions import ClientError
from langchain_core.documents import Document
from tqdm import tqdm

from llm_factory import get_embeddings
from pipeline.constants import (
    ANOMALY_HIGH_THRESHOLD,
    ANOMALY_K_NEIGHBOURS,
    ANOMALY_REF_MAX,
    ANOMALY_REF_SAMPLE_RATIO,
    EMBEDDING_BACKOFF_BASE_SECONDS,
    EMBEDDING_CONCURRENCY,
    EMBEDDING_MAX_CHARS,
    EMBEDDING_MAX_RETRIES,
    ERROR_SCORE_BOOST,
    IAM_CRITICAL_SCORE_BOOST,
)
from pipeline.query import load_retrieval_signals
from pipeline.text_utils import _is_iam_critical_text

embeddings = get_embeddings()


def _embed_batch_with_retry(
    batch_index: int,
    batch_texts: list[str],
) -> tuple[int, list[list[float]]]:
    """
    Embed one batch with retry/backoff for transient Bedrock failures.

    Args:
        batch_index: Sequential batch index for deterministic re-ordering
        batch_texts: Text payload for one embedding API call

    Returns:
        Tuple of (batch index, embedding vectors)
    """
    last_error: Optional[Exception] = None

    for attempt in range(1, EMBEDDING_MAX_RETRIES + 1):
        try:
            vectors = embeddings.embed_documents(batch_texts)
            return batch_index, vectors
        except ClientError as exc:
            last_error = exc
            err_code = exc.response.get("Error", {}).get("Code", "")
            retryable = err_code in {
                "ThrottlingException",
                "TooManyRequestsException",
                "ServiceUnavailableException",
                "InternalServerException",
            }
            if not retryable or attempt == EMBEDDING_MAX_RETRIES:
                raise

            sleep_seconds = EMBEDDING_BACKOFF_BASE_SECONDS ** attempt
            print(
                f"    [Embed Retry] batch={batch_index} attempt={attempt} "
                f"code={err_code} sleep={sleep_seconds:.1f}s"
            )
            time.sleep(sleep_seconds)
        except Exception as exc:
            last_error = exc
            if attempt == EMBEDDING_MAX_RETRIES:
                raise

            sleep_seconds = EMBEDDING_BACKOFF_BASE_SECONDS ** attempt
            print(
                f"    [Embed Retry] batch={batch_index} attempt={attempt} "
                f"sleep={sleep_seconds:.1f}s"
            )
            time.sleep(sleep_seconds)

    if last_error is not None:
        raise last_error
    raise RuntimeError("Embedding failed with unknown error")


def _embed_documents_batched(
    docs: list[Document],
    batch_size: int = 50,
    label: str = "Embedding",
) -> np.ndarray:
    """
    Embed a list of Documents in batches using Titan embeddings.
    Returns a 2-D numpy array of shape (n_docs, dim).

    Args:
        docs:       List of Document objects
        batch_size: Number of texts per API call
        label:      Label for the progress bar

    Returns:
        np.ndarray of shape (n_docs, embedding_dim)
    """
    # Truncate texts to a token-safe cap (Bedrock validates input by tokens).
    texts = [d.page_content[:EMBEDDING_MAX_CHARS] for d in docs]
    truncated_count = sum(1 for d in docs if len(d.page_content) > EMBEDDING_MAX_CHARS)
    if truncated_count:
        print(
            f"  [Embed] Token-safe truncation applied to {truncated_count:,} chunk(s) "
            f"at {EMBEDDING_MAX_CHARS:,} chars"
        )
    batches: list[list[str]] = [
        texts[i:i + batch_size]
        for i in range(0, len(texts), batch_size)
    ]
    ordered_batch_vectors: list[Optional[list[list[float]]]] = [None] * len(batches)

    with tqdm(total=len(texts), desc=f"    {label}", unit=" docs",
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
        with ThreadPoolExecutor(max_workers=EMBEDDING_CONCURRENCY) as executor:
            future_to_batch_info = {
                executor.submit(_embed_batch_with_retry, idx, batch): (idx, len(batch))
                for idx, batch in enumerate(batches)
            }

            for future in as_completed(future_to_batch_info):
                _, batch_len = future_to_batch_info[future]
                batch_index, batch_vectors = future.result()
                ordered_batch_vectors[batch_index] = batch_vectors
                pbar.update(batch_len)

    all_vecs: list[list[float]] = []
    for batch_vectors in ordered_batch_vectors:
        if batch_vectors is None:
            raise RuntimeError("Missing embedding batch result")
        all_vecs.extend(batch_vectors)

    return np.array(all_vecs, dtype=np.float32)


def score_anomalies(
    docs: list[Document],
    precomputed_embeddings: Optional[np.ndarray] = None,
    index_save_dir: Optional[str] = None,
) -> list[Document]:
    """
    Zero-shot runtime anomaly scoring inspired by the Ladle paper.

    Steps:
        1. Embed all chunks.
        2. Sample 20-30% of chunks (max 600) as "normal" reference set.
        3. Build an in-memory FAISS index on the reference embeddings.
        4. For every chunk, query k=6 nearest neighbours -> mean Euclidean distance.
        5. Compute trimmed mean & std on reference distances (remove top/bottom 10%).
        6. z_score = (distance - trimmed_mean) / std_dev for each chunk.
        7. Annotate metadata with anomaly_score and raw_distance.
        8. Optionally save the FAISS index, embeddings, and chunk metadata to disk.

    Args:
        docs:                  List of Document chunks (from hybrid_chunk_log)
        precomputed_embeddings: Optional precomputed embedding matrix of shape
                                (n_docs, embedding_dim) to avoid re-embedding
        index_save_dir:        Optional directory path to persist the FAISS reference
                               index, full embedding matrix, and chunk metadata for
                               post-analysis. Created if it does not exist.

    Returns:
        Same list of Documents with added anomaly metadata, sorted by score desc
    """
    import faiss  # Local import — only needed during scoring

    n = len(docs)
    if n == 0:
        return docs

    print(f"\n  [Anomaly] Scoring {n:,} chunks...")

    owns_embeddings = precomputed_embeddings is None

    # 1. Embed all chunks (or reuse precomputed embeddings)
    if precomputed_embeddings is None:
        all_embeddings = _embed_documents_batched(docs, batch_size=50, label="Embedding chunks")
    else:
        all_embeddings = precomputed_embeddings
        if all_embeddings.shape[0] != n:
            raise ValueError(
                f"Precomputed embeddings row count mismatch: {all_embeddings.shape[0]} != {n}"
            )
    dim = all_embeddings.shape[1]

    # 2. Sample reference set (20-30%, capped)
    ref_size = min(max(int(n * ANOMALY_REF_SAMPLE_RATIO), 1), ANOMALY_REF_MAX)
    if ref_size >= n:
        ref_indices = list(range(n))
    else:
        ref_indices = sorted(random.sample(range(n), ref_size))

    ref_embeddings = all_embeddings[ref_indices]
    print(f"    Reference set: {len(ref_indices):,} chunks")

    # 3. Build FAISS index on reference embeddings (L2 / Euclidean)
    index = faiss.IndexFlatL2(dim)
    index.add(ref_embeddings)

    # 4. Query every chunk against the reference index
    k = min(ANOMALY_K_NEIGHBOURS, len(ref_indices))
    distances, _ = index.search(all_embeddings, k)  # shape (n, k)
    mean_distances = distances.mean(axis=1)          # shape (n,)

    # 5. Compute baseline from reference set only
    ref_distances = mean_distances[ref_indices]
    sorted_ref = np.sort(ref_distances)
    trim = max(1, int(len(sorted_ref) * 0.10))
    trimmed = sorted_ref[trim:-trim] if trim < len(sorted_ref) // 2 else sorted_ref
    trimmed_mean_val = float(trimmed.mean())
    std_val = float(trimmed.std()) if len(trimmed) > 1 else 1.0

    # Guard against near-zero std
    if std_val < 1e-6:
        std_val = 1.0

    # 6. Compute z-scores
    print(f"    Baseline: trimmed_mean={trimmed_mean_val:.4f}, std={std_val:.4f}")

    for i, doc in enumerate(docs):
        raw_dist = float(mean_distances[i])
        z = (raw_dist - trimmed_mean_val) / std_val
        doc.metadata['anomaly_score'] = round(z, 4)
        doc.metadata['raw_distance'] = round(raw_dist, 4)

    retrieval_signals = load_retrieval_signals()
    iam_critical_keywords: list[str] = retrieval_signals['iam_critical_keywords']
    iam_critical_boosted = 0

    for doc in docs:
        content = doc.page_content
        score = doc.metadata['anomaly_score']
        has_iam_critical = doc.metadata.get('iam_critical')
        if not isinstance(has_iam_critical, bool):
            has_iam_critical = _is_iam_critical_text(content, iam_critical_keywords)
            doc.metadata['iam_critical'] = has_iam_critical

        if has_iam_critical:
            doc.metadata['anomaly_score'] = round(
                score + ERROR_SCORE_BOOST + IAM_CRITICAL_SCORE_BOOST, 4
            )
            iam_critical_boosted += 1

    print(f"    Score adjustments: {iam_critical_boosted} IAM-critical boosts")

    # Sort descending by anomaly score
    docs.sort(key=lambda d: d.metadata['anomaly_score'], reverse=True)

    high_count = sum(1 for d in docs if d.metadata['anomaly_score'] > ANOMALY_HIGH_THRESHOLD)
    print(f"    {high_count:,} chunks flagged as high-anomaly (z > {ANOMALY_HIGH_THRESHOLD})")

    # Save FAISS index and embeddings for post-analysis (if requested)
    if index_save_dir:
        try:
            os.makedirs(index_save_dir, exist_ok=True)
            faiss.write_index(index, os.path.join(index_save_dir, "index.faiss"))
            np.save(os.path.join(index_save_dir, "embeddings.npy"), all_embeddings)
            def _to_json_safe(value: Any) -> Any:
                if isinstance(value, datetime):
                    return value.isoformat()
                if isinstance(value, (str, int, float, bool)) or value is None:
                    return value
                return str(value)

            chunk_meta = [
                {
                    "content": d.page_content,
                    **{k: _to_json_safe(v) for k, v in d.metadata.items()},
                }
                for d in docs
            ]
            meta_payload = {"ref_indices": ref_indices, "chunks": chunk_meta}
            with open(os.path.join(index_save_dir, "metadata.json"), "w", encoding="utf-8") as fh:
                json.dump(meta_payload, fh, indent=2)
            print(f"    [FAISS] Saved index + embeddings to: {index_save_dir}/")
        except Exception as e:
            print(f"    [FAISS] Warning: Could not save index: {e}")

    # Clean up FAISS index from memory
    if owns_embeddings:
        del index, all_embeddings, ref_embeddings, distances
    else:
        del index, ref_embeddings, distances
    gc.collect()

    return docs
