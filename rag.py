"""
rag.py - Retrieval-Augmented Generation (RAG) Core Pipeline
InnovateX 2026 - College Hackathon Assistant

Features:
- Markdown document loading & semantic chunking preserving section hierarchy.
- In-memory vector indexing with sentence-transformers ('all-MiniLM-L6-v2').
- Cosine similarity retrieval via normalized vector dot products with NumPy.
- Configurable relevance thresholding to guard against out-of-domain / irrelevant queries.
"""

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class RulebookChunk:
    """Represents a single semantically meaningful chunk of the event rulebook."""
    chunk_id: int
    section_name: str
    content: str
    token_estimate: int = 0


def load_rulebook(file_path: str | Path) -> str:
    """Load the Markdown rulebook from the given path."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Rulebook file not found at: {file_path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def chunk_markdown(markdown_text: str) -> List[RulebookChunk]:
    """
    Split a Markdown document into meaningful chunks while preserving section titles.
    
    Splits by major headings (##) and further chunks subheadings (###) or individual
    FAQs (####) to keep chunks compact, focused, and semantically coherent.
    """
    chunks: List[RulebookChunk] = []
    chunk_id = 0

    # Split document by top-level or secondary Markdown headings: '## '
    section_pattern = re.compile(r"^##\s+(.+)$", re.MULTILINE)
    matches = list(section_pattern.finditer(markdown_text))

    if not matches:
        # Fallback if no '## ' found
        clean_text = markdown_text.strip()
        if clean_text:
            chunks.append(
                RulebookChunk(
                    chunk_id=0,
                    section_name="General Document",
                    content=clean_text,
                    token_estimate=len(clean_text.split()),
                )
            )
        return chunks

    for i, match in enumerate(matches):
        section_title = match.group(1).strip()
        start_idx = match.start()
        end_idx = matches[i + 1].start() if i + 1 < len(matches) else len(markdown_text)
        raw_section = markdown_text[start_idx:end_idx].strip()

        # If section is FAQ or Schedule, sub-chunk to enhance granular retrieval
        if "FAQ" in section_title or "Frequently Asked Questions" in section_title:
            faq_pattern = re.compile(r"^(####\s+Q\d+:?.+)$", re.MULTILINE)
            faq_matches = list(faq_pattern.finditer(raw_section))
            if faq_matches:
                for j, f_match in enumerate(faq_matches):
                    faq_header = f_match.group(1).replace("#", "").strip()
                    f_start = f_match.start()
                    f_end = faq_matches[j + 1].start() if j + 1 < len(faq_matches) else len(raw_section)
                    faq_block = raw_section[f_start:f_end].strip()
                    chunks.append(
                        RulebookChunk(
                            chunk_id=chunk_id,
                            section_name=f"{section_title} > {faq_header}",
                            content=f"[{section_title} - {faq_header}]\n{faq_block}",
                            token_estimate=len(faq_block.split()),
                        )
                    )
                    chunk_id += 1
                continue

        # If section contains subheadings '### ', chunk by '### '
        sub_pattern = re.compile(r"^###\s+(.+)$", re.MULTILINE)
        sub_matches = list(sub_pattern.finditer(raw_section))
        if len(sub_matches) > 1:
            # First preamble before first '### '
            preamble = raw_section[: sub_matches[0].start()].strip()
            if len(preamble.split()) > 20:
                chunks.append(
                    RulebookChunk(
                        chunk_id=chunk_id,
                        section_name=section_title,
                        content=preamble,
                        token_estimate=len(preamble.split()),
                    )
                )
                chunk_id += 1

            for k, s_match in enumerate(sub_matches):
                s_title = s_match.group(1).strip()
                s_start = s_match.start()
                s_end = sub_matches[k + 1].start() if k + 1 < len(sub_matches) else len(raw_section)
                sub_block = raw_section[s_start:s_end].strip()
                chunks.append(
                    RulebookChunk(
                        chunk_id=chunk_id,
                        section_name=f"{section_title} > {s_title}",
                        content=f"[{section_title} - {s_title}]\n{sub_block}",
                        token_estimate=len(sub_block.split()),
                    )
                )
                chunk_id += 1
            continue

        # Otherwise treat the entire section as a chunk
        chunks.append(
            RulebookChunk(
                chunk_id=chunk_id,
                section_name=section_title,
                content=raw_section,
                token_estimate=len(raw_section.split()),
            )
        )
        chunk_id += 1

    return chunks


class RAGPipeline:
    """
    In-memory RAG pipeline implementing vector indexing, normalized cosine similarity,
    and threshold-based relevance checks.
    """

    def __init__(
        self,
        rulebook_path: str | Path = "event_rulebook.md",
        model_name: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.35,
    ) -> None:
        self.rulebook_path = Path(rulebook_path)
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.model: Optional[SentenceTransformer] = None
        self.chunks: List[RulebookChunk] = []
        self.chunk_embeddings: Optional[np.ndarray] = None
        self.is_indexed: bool = False

    def load_model(self) -> SentenceTransformer:
        """Loads and caches the SentenceTransformer embedding model."""
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)
        return self.model

    def initialize_index(self) -> None:
        """Loads the rulebook, splits it into chunks, and computes normalized embeddings."""
        if self.is_indexed and self.chunk_embeddings is not None:
            return

        rulebook_text = load_rulebook(self.rulebook_path)
        self.chunks = chunk_markdown(rulebook_text)

        if not self.chunks:
            raise ValueError(f"No chunks extracted from rulebook: {self.rulebook_path}")

        model = self.load_model()
        texts_to_embed = [chunk.content for chunk in self.chunks]

        # Generate embeddings
        raw_embeddings = model.encode(
            texts_to_embed,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,  # Automatically normalizes to unit length (L2 norm)
        )

        self.chunk_embeddings = np.array(raw_embeddings, dtype=np.float32)
        self.is_indexed = True

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Retrieve the top-k most relevant rulebook chunks for a given user query.

        Returns a dictionary containing:
        - 'is_relevant': bool, True if the top score meets or exceeds the similarity threshold.
        - 'max_score': float, the highest cosine similarity score.
        - 'threshold': float, the threshold applied.
        - 'chunks': list of dicts with chunk details and similarity scores.
        """
        clean_query = query.strip()
        if not clean_query:
            return {
                "is_relevant": False,
                "max_score": 0.0,
                "threshold": threshold or self.similarity_threshold,
                "chunks": [],
                "reason": "Empty query provided.",
            }

        if not self.is_indexed or self.chunk_embeddings is None:
            self.initialize_index()

        applied_threshold = threshold if threshold is not None else self.similarity_threshold
        model = self.load_model()

        # Compute normalized query embedding
        query_embedding = model.encode(
            clean_query,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        # Calculate cosine similarity via dot product of unit-normalized vectors:
        # cos_sim(u, v) = u . v (since ||u|| = 1 and ||v|| = 1)
        scores = np.dot(self.chunk_embeddings, query_embedding)

        # Get indices of top_k matches sorted in descending order
        num_chunks = len(self.chunks)
        effective_k = min(top_k, num_chunks)
        top_indices = np.argsort(scores)[::-1][:effective_k]

        retrieved_items = []
        for idx in top_indices:
            score_val = float(scores[idx])
            chunk = self.chunks[idx]
            retrieved_items.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "section_name": chunk.section_name,
                    "content": chunk.content,
                    "score": round(score_val, 4),
                }
            )

        max_score = retrieved_items[0]["score"] if retrieved_items else 0.0
        is_relevant = max_score >= applied_threshold

        return {
            "is_relevant": is_relevant,
            "max_score": max_score,
            "threshold": applied_threshold,
            "chunks": retrieved_items,
            "reason": None if is_relevant else "Top similarity score is below the relevance threshold.",
        }
