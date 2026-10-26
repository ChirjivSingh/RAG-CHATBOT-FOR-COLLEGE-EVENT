"""
app.py - Streamlit Web Application
InnovateX 2026 — College Hackathon Assistant (RAG Demonstration)

Clean, user-facing conversational interface built for TechSpace (USICT, GGSIPU).
All backend parameters (API keys, similarity thresholds, and models) are managed
behind the scenes without exposing technical controls to students.
"""

import os
from dotenv import load_dotenv
import streamlit as st

from rag import RAGPipeline
from llm import DEFAULT_MODEL, generate_grounded_answer, get_api_key

# Page Configuration - Clean, focused layout without sidebar
st.set_page_config(
    page_title="InnovateX 2026 — Hackathon Assistant",
    page_icon="⚡",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Custom CSS to completely remove and hide the sidebar & technical elements
st.markdown(
    """
    <style>
    /* Hide Streamlit sidebar and its toggle button */
    [data-testid="stSidebar"] {
        display: none !important;
    }
    [data-testid="stSidebarCollapsedControl"] {
        display: none !important;
    }
    /* Polish main container padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 5rem;
        max-width: 800px;
    }
    /* Subtle subtitle and badge styling */
    .chat-subtext {
        color: #6b7280;
        font-size: 0.95rem;
        margin-bottom: 1.2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Load local environment variables (.env)
load_dotenv()

# Fixed Backend Configuration (Hidden from frontend users)
SIMILARITY_THRESHOLD = 0.35
TOP_K_CHUNKS = 4
ACTIVE_API_KEY = get_api_key()
SELECTED_MODEL = os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL)


@st.cache_resource(show_spinner=False)
def get_rag_pipeline() -> RAGPipeline:
    """Initialize and cache the in-memory RAG pipeline and SentenceTransformer model."""
    rulebook_file = os.path.join(os.path.dirname(__file__), "event_rulebook.md")
    pipeline = RAGPipeline(
        rulebook_path=rulebook_file,
        model_name="all-MiniLM-L6-v2",
        similarity_threshold=SIMILARITY_THRESHOLD,
    )
    pipeline.initialize_index()
    return pipeline


# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Quick query placeholder for suggested question clicks
if "preset_query" not in st.session_state:
    st.session_state.preset_query = None

# Header Section
col_title, col_action = st.columns([5, 1])
with col_title:
    st.title("⚡ InnovateX 2026")
    st.markdown(
        "**College Hackathon Assistant** &nbsp;•&nbsp; *TechSpace — USICT, GGSIPU*",
    )
    st.markdown(
        "<p class='chat-subtext'>Ask any question about registration, eligibility, schedule, "
        "team formation, judging criteria, prizes, or event policies.</p>",
        unsafe_allow_html=True,
    )

with col_action:
    st.write("")
    if st.button("🗑️ Clear", help="Clear conversation history", use_container_width=True):
        st.session_state.messages = []
        st.session_state.preset_query = None
        st.rerun()

# Load RAG Pipeline silently
try:
    pipeline = get_rag_pipeline()
except Exception as e:
    st.error(f"Unable to load event knowledge base: {e}")
    st.stop()

# Suggested Starter Questions (displayed only when chat is fresh)
if len(st.session_state.messages) == 0:
    st.markdown("##### 💡 Suggested Questions")
    sample_questions = [
        "When is the registration deadline and is there any fee?",
        "What are the minimum and maximum team sizes allowed?",
        "What are the exact judging criteria and scoring weights?",
        "What prizes are awarded for the winner and special categories?",
        "Can students from different IPU colleges form a team together?",
        "What is the policy on using AI coding tools like GitHub Copilot?",
    ]

    sample_cols = st.columns(2)
    for idx, q_text in enumerate(sample_questions):
        col_target = sample_cols[idx % 2]
        if col_target.button(f"📌 {q_text}", key=f"sample_{idx}", use_container_width=True):
            st.session_state.preset_query = q_text

st.markdown("---")

# Render Existing Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            sources_clean = [s.replace("## ", "").strip() for s in message["sources"]]
            st.caption("📌 **Source:** " + " • ".join(f"`{s}`" for s in sources_clean))

# Chat Input & User Interaction
user_query = st.chat_input("Ask a question about InnovateX 2026...")

if st.session_state.preset_query:
    user_query = st.session_state.preset_query
    st.session_state.preset_query = None

if user_query:
    clean_query = user_query.strip()
    if not clean_query:
        st.stop()

    # 1. Record and display user message
    st.session_state.messages.append({"role": "user", "content": clean_query})
    with st.chat_message("user"):
        st.markdown(clean_query)

    # 2. Semantic Retrieval from Rulebook
    with st.spinner("Checking event rulebook..."):
        retrieval_result = pipeline.retrieve(
            clean_query,
            top_k=TOP_K_CHUNKS,
            threshold=SIMILARITY_THRESHOLD,
        )

    is_relevant = retrieval_result["is_relevant"]
    retrieved_chunks = retrieval_result["chunks"]

    # 3. Response Generation
    with st.chat_message("assistant"):
        if not is_relevant:
            # Query is out of domain or unsupported by the rulebook
            fallback_answer = (
                "The InnovateX 2026 rulebook does not contain information regarding this topic.\n\n"
                "For queries not covered in the official documentation, please reach out directly to the "
                "**TechSpace Organizing Committee** at `support@innovatex.techspace.org` or visit "
                "the Central Help Desk at USICT E-Block (Room 101)."
            )
            st.markdown(fallback_answer)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": fallback_answer,
                    "sources": [],
                }
            )
        else:
            # Query is relevant -> invoke grounded answer generation
            with st.spinner("Generating answer..."):
                llm_response = generate_grounded_answer(
                    question=clean_query,
                    retrieved_chunks=retrieved_chunks,
                    api_key=ACTIVE_API_KEY,
                    model_name=SELECTED_MODEL,
                )

            answer_content = llm_response["answer"]
            sources = llm_response.get("sources", [])

            st.markdown(answer_content)
            if sources:
                sources_clean = [s.replace("## ", "").strip() for s in sources]
                st.caption("📌 **Source:** " + " • ".join(f"`{s}`" for s in sources_clean))

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer_content,
                    "sources": sources,
                }
            )
