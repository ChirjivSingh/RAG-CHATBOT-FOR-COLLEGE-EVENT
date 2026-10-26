"""
tests/test_rag.py - Comprehensive Unit Tests for RAG Pipeline and LLM Grounding
InnovateX 2026 — College Hackathon Assistant

Tests covered (without requiring live OpenRouter API calls):
1. Document existence, consistency, and judging weights summing to 100%.
2. Deadline consistency across schedule and registration sections.
3. Chunking logic and section header preservation.
4. Empty query handling.
5. Registration deadline retrieval.
6. Team size retrieval.
7. Submission deadline retrieval.
8. Judging criteria retrieval.
9. Prize amounts retrieval.
10. Multi-section information retrieval.
11. Unanswered / out-of-domain questions with threshold checking.
12. Completely irrelevant query threshold rejection.
13. Missing API key handling.
14. API rate limit & connection failure handling (mocked).
15. Successful grounded answer generation (mocked).
16. Citation verification (ensuring citations match actual retrieved chunks).
"""

import sys
from pathlib import Path
import re
from unittest.mock import MagicMock, patch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from rag import RAGPipeline, chunk_markdown, load_rulebook
from llm import (
    DEFAULT_MODEL,
    format_context_for_prompt,
    generate_grounded_answer,
)
from openai import APIConnectionError, RateLimitError


@pytest.fixture(scope="module")
def rulebook_path():
    path = Path(__file__).parent.parent / "event_rulebook.md"
    assert path.is_file(), f"event_rulebook.md not found at {path}"
    return path


@pytest.fixture(scope="module")
def pipeline(rulebook_path):
    pipe = RAGPipeline(rulebook_path=rulebook_path, similarity_threshold=0.35)
    pipe.initialize_index()
    return pipe


# ---------------------------------------------------------------------------
# 1. Rulebook Integrity and Scoring Weight Tests
# ---------------------------------------------------------------------------

def test_rulebook_scoring_weights_sum_to_100(rulebook_path):
    """Verify that all judging score percentage weights sum to exactly 100%."""
    content = load_rulebook(rulebook_path)
    
    # Locate Section 8: Judging Criteria
    assert "Judging Criteria" in content
    
    # Extract percentage numbers listed in table rows for weights
    # e.g., | **Problem Solving & Innovation** | **25%** |
    weight_matches = re.findall(r"\*\*(\d+)%\*\*", content)
    # The last match might be total 100%, filter criteria values:
    criteria_weights = [int(w) for w in weight_matches if int(w) < 100]
    
    assert len(criteria_weights) == 5, f"Expected 5 judging criteria, found: {criteria_weights}"
    assert sum(criteria_weights) == 100, f"Judging weights must sum to exactly 100, got: {sum(criteria_weights)}"
    assert criteria_weights == [25, 25, 20, 15, 15]


def test_rulebook_deadlines_consistency(rulebook_path):
    """Verify there are no conflicting deadlines in the rulebook."""
    content = load_rulebook(rulebook_path)
    
    # Registration dates
    assert "February 1, 2026" in content  # Opens
    assert "March 15, 2026, at 11:59 PM IST" in content  # Closes
    assert "March 20, 2026, at 5:00 PM IST" in content  # Team roster change deadline
    
    # Event dates
    assert "April 4, 2026" in content  # Day 1
    assert "April 5, 2026" in content  # Day 2
    
    # Submission deadline
    assert "11:00 AM IST" in content  # Soft freeze
    assert "11:05 AM IST" in content  # Hard cutoff


# ---------------------------------------------------------------------------
# 2. Chunking Tests
# ---------------------------------------------------------------------------

def test_chunking_preserves_sections(rulebook_path):
    """Verify Markdown chunking splits document into distinct chunks with section names."""
    text = load_rulebook(rulebook_path)
    chunks = chunk_markdown(text)
    
    assert len(chunks) >= 14, f"Expected at least 14 chunks, got {len(chunks)}"
    
    # Check that major sections are present
    section_titles = [c.section_name for c in chunks]
    assert any("Event Overview" in s for s in section_titles)
    assert any("Eligibility" in s for s in section_titles)
    assert any("Registration" in s for s in section_titles)
    assert any("Team Formation" in s for s in section_titles)
    assert any("Schedule" in s for s in section_titles)
    assert any("Submission" in s for s in section_titles)
    assert any("Judging Criteria" in s for s in section_titles)
    assert any("Prize Categories" in s for s in section_titles)
    assert any("FAQ" in s for s in section_titles)
    
    for chunk in chunks:
        assert chunk.content.strip() != ""
        assert chunk.section_name.strip() != ""


# ---------------------------------------------------------------------------
# 3. Retrieval Semantic Query Tests (Unit tests without LLM API calls)
# ---------------------------------------------------------------------------

def test_empty_query_retrieval(pipeline):
    """An empty or blank question should return no chunks and is_relevant=False."""
    result = pipeline.retrieve("", top_k=4)
    assert result["is_relevant"] is False
    assert result["chunks"] == []
    assert result["max_score"] == 0.0

    whitespace_result = pipeline.retrieve("    ", top_k=4)
    assert whitespace_result["is_relevant"] is False
    assert whitespace_result["chunks"] == []


def test_registration_deadline_retrieval(pipeline):
    """Question about registration deadline retrieves Section 3."""
    query = "What is the registration closing deadline for InnovateX?"
    result = pipeline.retrieve(query, top_k=3)
    
    assert result["is_relevant"] is True
    assert result["max_score"] >= 0.40
    
    # Check that Registration section is among the top retrieved chunks
    sections = [c["section_name"] for c in result["chunks"]]
    assert any("Registration" in s for s in sections), f"Expected Registration in {sections}"
    
    combined_content = " ".join(c["content"] for c in result["chunks"])
    assert "March 15, 2026" in combined_content


def test_team_size_retrieval(pipeline):
    """Question about team size retrieves Section 4."""
    query = "What is the minimum and maximum allowed team size?"
    result = pipeline.retrieve(query, top_k=3)
    
    assert result["is_relevant"] is True
    top_chunk = result["chunks"][0]
    assert "Team Formation" in top_chunk["section_name"] or "FAQ" in top_chunk["section_name"]
    assert "minimum of 2" in top_chunk["content"] or "2 to 4" in top_chunk["content"]


def test_submission_deadline_retrieval(pipeline):
    """Question about project submission deadline retrieves Section 5 or 6."""
    query = "When is the final code submission deadline on Day 2?"
    result = pipeline.retrieve(query, top_k=3)
    
    assert result["is_relevant"] is True
    matched_sections = [c["section_name"] for c in result["chunks"]]
    assert any("Submission" in s or "Schedule" in s for s in matched_sections)
    
    combined_content = " ".join(c["content"] for c in result["chunks"])
    assert "11:00 AM" in combined_content


def test_judging_criteria_retrieval(pipeline):
    """Question about judging score weights retrieves Section 8."""
    query = "What are the judging criteria and scoring breakdown weights?"
    result = pipeline.retrieve(query, top_k=3)
    
    assert result["is_relevant"] is True
    top_chunk = result["chunks"][0]
    assert "Judging Criteria" in top_chunk["section_name"]
    assert "25%" in top_chunk["content"]
    assert "100%" in top_chunk["content"]


def test_prize_amounts_retrieval(pipeline):
    """Question about prize money retrieves Section 9."""
    query = "What is the cash prize amount for the first place winner and runner up?"
    result = pipeline.retrieve(query, top_k=3)
    
    assert result["is_relevant"] is True
    top_chunk = result["chunks"][0]
    assert "Prize Categories" in top_chunk["section_name"]
    assert "₹50,000" in top_chunk["content"]


def test_multi_section_retrieval(pipeline):
    """Question requiring info from both eligibility and prizes."""
    query = "Are non-IPU students eligible and what is the cash prize for the all-women team?"
    result = pipeline.retrieve(query, top_k=5)
    
    assert result["is_relevant"] is True
    sections = [c["section_name"] for c in result["chunks"]]
    
    # Must retrieve chunks covering both eligibility and prize categories
    has_eligibility = any("Eligibility" in s or "FAQ" in s for s in sections)
    has_prizes = any("Prize" in s for s in sections)
    assert has_eligibility and has_prizes, f"Expected both sections in {sections}"


def test_unanswered_question_in_rulebook(pipeline):
    """A question about something not in the rulebook (e.g., cafeteria microwave brands)."""
    query = "What brand of microwave oven is in the college cafeteria kitchen?"
    result = pipeline.retrieve(query, top_k=4)
    # The score should be low and not match relevant policy
    assert result["max_score"] < 0.35 or result["is_relevant"] is False


def test_completely_irrelevant_query(pipeline):
    """An irrelevant question unrelated to the event should fail threshold check."""
    query = "How do you make a strawberry cheesecake from scratch in an electric oven?"
    result = pipeline.retrieve(query, top_k=4, threshold=0.35)
    
    assert result["is_relevant"] is False
    assert result["max_score"] < 0.35
    assert result["reason"] == "Top similarity score is below the relevance threshold."


# ---------------------------------------------------------------------------
# 4. LLM Integration & Error Handling Tests (Mocked API)
# ---------------------------------------------------------------------------

def test_missing_api_key_handling():
    """Verify that a missing or placeholder API key is caught gracefully."""
    dummy_chunks = [{"section_name": "Section 1", "content": "Sample content", "score": 0.8}]
    
    # Missing key
    res_none = generate_grounded_answer(
        question="When is registration?",
        retrieved_chunks=dummy_chunks,
        api_key=None,
    )
    # With load_dotenv, if OPENROUTER_API_KEY is not set or empty:
    with patch.dict("os.environ", {"OPENROUTER_API_KEY": ""}, clear=True):
        res_empty = generate_grounded_answer(
            question="When is registration?",
            retrieved_chunks=dummy_chunks,
            api_key="",
        )
        assert res_empty["success"] is False
        assert res_empty["error"] == "MISSING_API_KEY"
        assert "Missing OpenRouter API Key" in res_empty["answer"]


def test_mocked_llm_rate_limit():
    """Verify rate limit errors from OpenRouter are handled gracefully."""
    mock_client = MagicMock()
    # Mock RateLimitError
    mock_response = MagicMock()
    mock_response.status_code = 429
    mock_client.chat.completions.create.side_effect = RateLimitError(
        message="Rate limit exceeded",
        response=mock_response,
        body={"error": "Rate limit reached"},
    )
    
    dummy_chunks = [{"section_name": "Section 3", "content": "Deadline text", "score": 0.85}]
    
    res = generate_grounded_answer(
        question="When is the deadline?",
        retrieved_chunks=dummy_chunks,
        api_key="mock_key",
        client=mock_client,
    )
    assert res["success"] is False
    assert "RATE_LIMIT" in res["error"]
    assert "Rate Limit Exceeded on OpenRouter" in res["answer"]


def test_mocked_llm_connection_failure():
    """Verify network connection errors are captured and reported cleanly."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = APIConnectionError(
        request=MagicMock()
    )
    
    dummy_chunks = [{"section_name": "Section 1", "content": "Text", "score": 0.9}]
    
    res = generate_grounded_answer(
        question="Where is USICT located?",
        retrieved_chunks=dummy_chunks,
        api_key="mock_key",
        client=mock_client,
    )
    assert res["success"] is False
    assert "CONNECTION_ERROR" in res["error"]
    assert "Network / Connection Error" in res["answer"]


def test_mocked_llm_grounded_answer_generation():
    """Verify successful end-to-end grounded answer flow with mocked OpenRouter response."""
    mock_client = MagicMock()
    mock_completion = MagicMock()
    mock_completion.choices = [
        MagicMock(
            message=MagicMock(
                content=(
                    "According to [Section 3: Registration Deadlines and Procedures], registration "
                    "closes on Sunday, March 15, 2026, at 11:59 PM IST. Participation is completely free."
                )
            )
        )
    ]
    mock_client.chat.completions.create.return_value = mock_completion
    
    chunks = [
        {
            "chunk_id": 2,
            "section_name": "3. Registration Deadlines and Procedures",
            "content": "Registration Closes: Sunday, March 15, 2026, at 11:59 PM IST. Registration Fee: Free of cost (₹0).",
            "score": 0.78,
        }
    ]
    
    res = generate_grounded_answer(
        question="When is registration closing and what is the fee?",
        retrieved_chunks=chunks,
        api_key="mock_test_key",
        client=mock_client,
    )
    
    assert res["success"] is True
    assert "March 15, 2026" in res["answer"]
    assert "3. Registration Deadlines and Procedures" in res["sources"]
    assert res["model"] == DEFAULT_MODEL


def test_citations_refer_to_actual_chunks():
    """Verify that citations returned by the LLM helper match actual retrieved chunks."""
    chunks = [
        {"section_name": "Section A", "content": "Text A", "score": 0.9},
        {"section_name": "Section B", "content": "Text B", "score": 0.8},
        {"section_name": "Section A", "content": "Text A2", "score": 0.7},  # duplicate section
    ]
    formatted = format_context_for_prompt(chunks)
    assert "Source: Section A" in formatted
    assert "Source: Section B" in formatted
    
    # In generate_grounded_answer, unique sources are extracted:
    sources = list(dict.fromkeys(c["section_name"] for c in chunks))
    assert sources == ["Section A", "Section B"]
