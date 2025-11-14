# agent_fully_aligned_with_pinecone.py
# 100% ALIGNED with your Pinecone upload script schema
# Handles all field name variations and flattened metadata structure

from pinecone import Pinecone
from openai import OpenAI
import os
from typing import List, Dict, Optional
import json
import mlflow
from mlflow.entities import SpanType
import time
from datetime import datetime
import sqlite3
import hashlib
from pathlib import Path
from llama_index.embeddings.vertex import VertexTextEmbedding
from vertexai.language_models import TextEmbeddingModel
import vertexai
# For groundedness evaluation
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from datasets import Dataset
from google.oauth2 import service_account
from VectorDatabase.safe_pinecone_uploader_main import SafeVertexAIEmbedder
from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper

BASE_DATA_PATH = 'C:/Users/db234/PycharmProjects/EdgarFilingsBod/Python/'
INPUT_JSON_PATH = r'C:/Users/db234/PycharmProjects/EdgarFilingsBod/Python/output/processed_10k_chunks.json'
# CHECKPOINT_PATH = r'C:/Users/db234/PycharmProjects/EdgarFilingsBod/Python/VectorDatabase/checkpoint'
# Pinecone configuration
PINECONE_API_KEY = ""
OPENAI_KEY = ''
PINECONE_ENVIRONMENT = "us-central1"  # or your preferred region
INDEX_NAME = "financial-10k-filings-total-final-768-3"

GCP_PROJECT_ID = "pineconeproject"
GCP_LOCATION = "us-central1"  # or your preferred region
VERTEX_MODEL = "text-embedding-004"  # Latest Vertex AI embedding model
MAX_BUDGET_USD = 250.0


class MetadataRouter:
    """
    Route queries using metadata intelligence
    100% ALIGNED WITH FLATTENED PINECONE SCHEMA from upload script
    """

    def __init__(self, client: OpenAI):
        self.client = client

    def extract_metadata_requirements(self, question: str) -> Dict:
        """Use LLM to extract metadata filters from question"""

        prompt = f"""Analyze this financial question and extract metadata requirements for searching 10-K filings.

Question: {question}

Extract and return JSON with:
{{
    "tickers": ["list of company TICKERS (AMD, AAPL, MSFT) or null for all"],
    "years": ["list of filing years (2021, 2022) or null for all"],
    "statement_types": null,  # ← DISABLE THIS FOR NOW
    "data_types": null,  # ← DISABLE THIS TOO
    "requires_numeric_data": false,  # ← AND THIS
    "requires_xbrl": false,
    "temporal_scope": "single_year" or "multi_year" or "trend"
}}


Field descriptions:
- tickers: Stock ticker symbols (AMD, AAPL, MSFT, etc)
- years: Filing years (e.g., 2021 10-K filing)
- statement_types: BS (Balance Sheet), IS (Income Statement), CF (Cash Flow)
- data_types: "table_content" for financial tables/numbers, "text_chunk" for narrative text
- requires_numeric_data: true if query needs actual numbers
- requires_xbrl: true if query needs structured XBRL financial statements
- temporal_scope: whether comparing single year, multiple years, or trends

Examples:
"What was Apple's revenue in 2021?" → {{"tickers": ["AAPL"], "years": ["2021"], "statement_types": ["IS"], "data_types": ["table_content"], "requires_numeric_data": true}}
"AMD's inventory breakdown in 2021?" → {{"tickers": ["AMD"], "years": ["2021"], "statement_types": ["BS"], "data_types": ["table_content"], "requires_numeric_data": true}}
"Describe Tesla's business strategy" → {{"tickers": ["TSLA"], "data_types": ["text_chunk"], "requires_numeric_data": false}}
"Compare R&D spending across tech companies" → {{"statement_types": ["IS"], "data_types": ["table_content"], "requires_numeric_data": true, "temporal_scope": "trend"}}

Return ONLY valid JSON, no other text.
"""
        prompt_artifact = f"""Analyze this financial question and extract metadata requirements for searching 10-K filings.

        Question: 'question'

        Extract and return JSON with:
        {{
            "tickers": ["list of company TICKERS (AMD, AAPL, MSFT) or null for all"],
            "years": ["list of filing years (2021, 2022) or null for all"],
            "statement_types": null,  # ← DISABLE THIS FOR NOW
            "data_types": null,  # ← DISABLE THIS TOO
            "requires_numeric_data": false,  # ← AND THIS
            "requires_xbrl": false,
            "temporal_scope": "single_year" or "multi_year" or "trend"
        }}


        Field descriptions:
        - tickers: Stock ticker symbols (AMD, AAPL, MSFT, etc)
        - years: Filing years (e.g., 2021 10-K filing)
        - statement_types: BS (Balance Sheet), IS (Income Statement), CF (Cash Flow)
        - data_types: "table_content" for financial tables/numbers, "text_chunk" for narrative text
        - requires_numeric_data: true if query needs actual numbers
        - requires_xbrl: true if query needs structured XBRL financial statements
        - temporal_scope: whether comparing single year, multiple years, or trends

        Examples:
        "What was Apple's revenue in 2021?" → {{"tickers": ["AAPL"], "years": ["2021"], "statement_types": ["IS"], "data_types": ["table_content"], "requires_numeric_data": true}}
        "AMD's inventory breakdown in 2021?" → {{"tickers": ["AMD"], "years": ["2021"], "statement_types": ["BS"], "data_types": ["table_content"], "requires_numeric_data": true}}
        "Describe Tesla's business strategy" → {{"tickers": ["TSLA"], "data_types": ["text_chunk"], "requires_numeric_data": false}}
        "Compare R&D spending across tech companies" → {{"statement_types": ["IS"], "data_types": ["table_content"], "requires_numeric_data": true, "temporal_scope": "trend"}}

        Return ONLY valid JSON, no other text.
        """
        mlflow.genai.register_prompt(
            name="System-Instruction-Template-MetaData",
            template=prompt_artifact,
            # description="The base system instruction for the LLM.",
            commit_message="Initial commit of base template"
        )
        with mlflow.start_span(name='MetaDataRouterModel', span_type=SpanType.LLM) as span:
            span.set_attribute("llm.model_name", "gpt-4o-mini")
            span.set_attribute("llm.temperature", 0.3)
            span.set_inputs({"prompt": prompt, "user_query": question})

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )

        try:
            content = response.choices[0].message.content.strip()
            # Remove markdown code blocks if present
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            metadata_reqs = json.loads(content)
            print(f"🔍 DEBUG - Extracted metadata: {metadata_reqs}")
            return metadata_reqs
        except Exception as e:
            print(f"⚠️  Failed to parse metadata requirements: {e}")
            print(f"   Raw response: {response.choices[0].message.content[:200]}")
            return {}

    def build_pinecone_filters(self, metadata_reqs: Dict) -> List[Dict]:
        """
        Convert metadata requirements to Pinecone filter combinations
        HANDLES DUAL STATEMENT TYPE FIELDS: 'stmt_type' AND 'statement_type'
        """

        filters = []

        if metadata_reqs.get("tickers"):
            # For multiple tickers, create separate filters for better coverage
            for ticker in metadata_reqs["tickers"]:
                ticker_filter = {"ticker": {"$eq": ticker}}

                # Add year filter if specified
                if metadata_reqs.get("years"):
                    ticker_filter["year"] = {"$in": metadata_reqs["years"]}

                # Add statement type filter - HANDLES BOTH FIELD NAMES
                # Your upload script creates BOTH 'stmt_type' (from statement_info)
                # AND 'statement_type' (from xbrl_context)
                if metadata_reqs.get("statement_types"):
                    # Use $or to check both possible field names
                    ticker_filter["$or"] = [
                        {"stmt_type": {"$in": metadata_reqs["statement_types"]}},
                        {"statement_type": {"$in": metadata_reqs["statement_types"]}}
                    ]

                # Add data type filter (table_content vs text_chunk)
                if metadata_reqs.get("data_types"):
                    ticker_filter["data_type"] = {"$in": metadata_reqs["data_types"]}

                # Add numeric data requirement
                if metadata_reqs.get("requires_numeric_data"):
                    ticker_filter["has_numeric_data"] = {"$eq": True}

                # Add XBRL requirement
                if metadata_reqs.get("requires_xbrl"):
                    ticker_filter["is_xbrl"] = {"$eq": True}

                filters.append(ticker_filter)
        else:
            # Single filter for all companies
            base_filter = {}

            if metadata_reqs.get("years"):
                base_filter["year"] = {"$in": metadata_reqs["years"]}

            if metadata_reqs.get("statement_types"):
                # Handle both field names
                base_filter["$or"] = [
                    {"stmt_type": {"$in": metadata_reqs["statement_types"]}},
                    {"statement_type": {"$in": metadata_reqs["statement_types"]}}
                ]

            if metadata_reqs.get("data_types"):
                base_filter["data_type"] = {"$in": metadata_reqs["data_types"]}

            if metadata_reqs.get("requires_numeric_data"):
                base_filter["has_numeric_data"] = {"$eq": True}

            if metadata_reqs.get("requires_xbrl"):
                base_filter["is_xbrl"] = {"$eq": True}

            filters.append(base_filter if base_filter else None)

        return filters

    def adjust_top_k(self, metadata_reqs: Dict, base_top_k: int = 5) -> int:
        """
        Dynamically adjust top_k based on query complexity
        Ensures sufficient results when using metadata filters
        """

        # If filtering by specific tickers, need more results per ticker
        if metadata_reqs.get("tickers"):
            num_tickers = len(metadata_reqs["tickers"])
            return base_top_k * max(num_tickers, 3)

        # For trend analysis, need more historical data points
        if metadata_reqs.get("temporal_scope") == "trend":
            return base_top_k * 5

        # For multi-year comparisons, need results from each year
        if metadata_reqs.get("years") and len(metadata_reqs["years"]) > 1:
            return base_top_k * len(metadata_reqs["years"])

        # For multi-year scope without specific years
        if metadata_reqs.get("temporal_scope") == "multi_year":
            return base_top_k * 3

        return base_top_k


class CitationManager:
    """
    Manage citations with hyperlinks to source files
    100% ALIGNED WITH FLATTENED PINECONE SCHEMA
    """

    def __init__(self, base_path: str = BASE_DATA_PATH):
        self.base_path = Path(base_path)

    def create_file_url(self, folder_path: str) -> str:
        """
        Convert folder path to file:// URL for local system
        Cross-platform compatible
        """
        # Clean the path
        if folder_path.startswith("Python/"):
            folder_path = folder_path.replace("Python/", "")

        # Construct full path
        full_path = self.base_path / folder_path

        # Convert to file URL (works on Windows, Mac, Linux)
        if os.name == 'nt':  # Windows
            file_url = full_path.as_uri()
        else:  # Unix/Linux/Mac
            file_url = full_path.as_uri()

        return file_url

    def create_citation_reference(self, source_metadata: Dict, source_number: int) -> Dict:
        """
        Create a rich citation reference with hyperlink
        ALL FIELDS ARE TOP-LEVEL (flattened from upload script)
        HANDLES BOTH 'stmt_type' AND 'statement_type'
        """

        # Extract metadata fields (all at top level, no nesting)
        ticker = source_metadata.get("ticker", "Unknown")
        year = source_metadata.get("year", "")
        cik = source_metadata.get("cik", "")
        exhibit_type = source_metadata.get("exhibit_type", "")
        filename = source_metadata.get("filename", "")
        folder = source_metadata.get("folder", "")

        # Statement/XBRL info (flattened at top level)
        # HANDLES BOTH POSSIBLE FIELD NAMES
        stmt_type = source_metadata.get("stmt_type") or source_metadata.get("statement_type", "")
        statement_name = source_metadata.get("statement_name", "")
        xbrl_title = source_metadata.get("xbrl_title", "")

        # Additional context
        data_type = source_metadata.get("data_type", "")
        r_type = source_metadata.get("r_type", "")

        # Create hyperlink to file
        file_url = self.create_file_url(folder) if folder else None

        # Create human-readable citation text
        citation_parts = []

        # Company identifier
        if ticker and ticker != "UNKNOWN":
            citation_parts.append(ticker)
        elif cik:
            citation_parts.append(f"CIK {cik}")

        # Year
        if year and year != "UNKNOWN":
            citation_parts.append(f"{year} 10-K")

        # Document title (prioritize XBRL title, then statement name, then statement type)
        if xbrl_title:
            # Truncate long XBRL titles
            citation_parts.append(xbrl_title[:60] + "..." if len(xbrl_title) > 60 else xbrl_title)
        elif statement_name:
            citation_parts.append(statement_name)
        elif stmt_type:
            # Map statement codes to readable names
            stmt_map = {
                "BS": "Balance Sheet",
                "IS": "Income Statement",
                "CF": "Cash Flow Statement"
            }
            citation_parts.append(stmt_map.get(stmt_type, stmt_type))

        # Exhibit type (if not R-FILE or MAIN-10K)
        if exhibit_type and exhibit_type not in ["R-FILE", "MAIN-10K", "UNKNOWN"]:
            citation_parts.append(f"Exhibit {exhibit_type}")

        # R-file type if available
        if r_type:
            citation_parts.append(f"({r_type})")

        citation_text = " - ".join(citation_parts) if citation_parts else "Unknown Source"

        return {
            "source_number": source_number,
            "citation_text": citation_text,
            "file_url": file_url,
            "filename": filename,
            "folder": folder,
            "ticker": ticker,
            "cik": cik,
            "year": year,
            "data_type": data_type,
            "statement_type": stmt_type,
            "metadata": source_metadata  # Store full metadata for reference
        }

    def format_citation_for_output(self, citation: Dict, format_type: str = "markdown") -> str:
        """
        Format citation with hyperlink in various output formats
        Supports: markdown, html, text
        """

        source_num = citation["source_number"]
        text = citation["citation_text"]
        url = citation["file_url"]

        if format_type == "markdown":
            if url:
                return f"[Source {source_num}: {text}]({url})"
            else:
                return f"Source {source_num}: {text}"

        elif format_type == "html":
            if url:
                return f'<a href="{url}" target="_blank" class="citation-link">Source {source_num}: {text}</a>'
            else:
                return f"<span class='citation'>Source {source_num}: {text}</span>"

        elif format_type == "text":
            if url:
                return f"Source {source_num}: {text}\n    File: {url}"
            else:
                return f"Source {source_num}: {text}"

        elif format_type == "json":
            return json.dumps({
                "source_number": source_num,
                "citation_text": text,
                "file_url": url
            }, indent=2)

        else:
            return f"Source {source_num}: {text}"

    def create_bibliography(self, citations: List[Dict], format_type: str = "markdown") -> str:
        """
        Create a formatted bibliography section from all citations
        """

        if not citations:
            return ""

        if format_type == "markdown":
            bib = "\n\n---\n\n### 📚 Sources\n\n"
            for citation in sorted(citations, key=lambda x: x["source_number"]):
                formatted = self.format_citation_for_output(citation, "markdown")
                bib += f"{citation['source_number']}. {formatted}\n"
            return bib

        elif format_type == "html":
            bib = '\n\n<hr>\n\n<h3 class="sources-header">📚 Sources</h3>\n<ol class="sources-list">\n'
            for citation in sorted(citations, key=lambda x: x["source_number"]):
                formatted = self.format_citation_for_output(citation, "html")
                bib += f"<li>{formatted}</li>\n"
            bib += "</ol>\n"
            return bib

        elif format_type == "text":
            bib = "\n\n" + "=" * 60 + "\nSOURCES\n" + "=" * 60 + "\n\n"
            for citation in sorted(citations, key=lambda x: x["source_number"]):
                formatted = self.format_citation_for_output(citation, "text")
                bib += f"{formatted}\n\n"
            return bib

        return ""

    def get_citation_stats(self, citations: List[Dict]) -> Dict:
        """
        Get statistics about citations for quality metrics
        """

        if not citations:
            return {
                "total_citations": 0,
                "unique_tickers": 0,
                "unique_years": 0,
                "data_types": {},
                "has_file_urls": 0
            }

        tickers = set()
        years = set()
        data_types = {}
        has_urls = 0

        for citation in citations:
            if citation.get("ticker"):
                tickers.add(citation["ticker"])
            if citation.get("year"):
                years.add(citation["year"])

            dtype = citation.get("data_type", "unknown")
            data_types[dtype] = data_types.get(dtype, 0) + 1

            if citation.get("file_url"):
                has_urls += 1

        return {
            "total_citations": len(citations),
            "unique_tickers": len(tickers),
            "unique_years": len(years),
            "tickers": sorted(tickers),
            "years": sorted(years),
            "data_types": data_types,
            "has_file_urls": has_urls,
            "url_coverage_pct": (has_urls / len(citations) * 100) if citations else 0
        }


class ConversationManager:
    """Manage conversation history with context window limits"""

    def __init__(self, db_path: str = "./conversation_history.db", max_context_tokens: int = 100000):
        self.db_path = db_path
        self.max_context_tokens = max_context_tokens
        self._init_db()

    def _init_db(self):
        """Initialize SQLite database for query logging"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS queries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                user_id TEXT,
                session_id TEXT,
                question TEXT,
                answer TEXT,
                run_id TEXT,
                cost REAL,
                latency REAL,
                num_sources INTEGER,
                model_used TEXT,
                context_used TEXT,
                metadata JSON
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversation_context (
                session_id TEXT,
                timestamp TEXT,
                turn_number INTEGER,
                question TEXT,
                answer_summary TEXT,
                key_entities JSON,
                PRIMARY KEY (session_id, turn_number)
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_user_session 
            ON queries(user_id, session_id, timestamp)
        """)

        conn.commit()
        conn.close()

    def log_query(self, user_id: str, session_id: str, question: str,
                  answer: str, run_id: str, cost: float, latency: float,
                  num_sources: int, model_used: str, metadata: Dict = None):
        """Log query to persistent storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO queries 
            (timestamp, user_id, session_id, question, answer, run_id, 
             cost, latency, num_sources, model_used, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            user_id,
            session_id,
            question,
            answer,
            run_id,
            cost,
            latency,
            num_sources,
            model_used,
            json.dumps(metadata or {})
        ))

        conn.commit()
        conn.close()

    def get_conversation_history(self, session_id: str, max_turns: int = 5) -> List[Dict]:
        """Retrieve recent conversation history for context"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT question, answer, timestamp, run_id
            FROM queries
            WHERE session_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (session_id, max_turns))

        rows = cursor.fetchall()
        conn.close()

        return [
            {"question": row[0], "answer": row[1], "timestamp": row[2], "run_id": row[3]}
            for row in reversed(rows)
        ]

    def summarize_conversation_turn(self, question: str, answer: str,
                                    session_id: str, turn_number: int,
                                    client: OpenAI):
        """Create a compressed summary of a conversation turn"""

        prompt = f"""Summarize this Q&A exchange into a concise context snippet (max 100 words).
Extract key entities (companies, metrics, topics) as a JSON list.

Question: {question}
Answer: {answer[:1000]}...

Provide:
1. Summary: [concise summary]
2. Key entities: ["entity1", "entity2", ...]
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )

        content = response.choices[0].message.content
        lines = content.split('\n')
        summary = ""
        entities = []

        for line in lines:
            if line.startswith("1. Summary:"):
                summary = line.replace("1. Summary:", "").strip()
            elif line.startswith("2. Key entities:"):
                try:
                    entities_str = line.replace("2. Key entities:", "").strip()
                    entities = json.loads(entities_str)
                except:
                    entities = []

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO conversation_context
            (session_id, timestamp, turn_number, question, answer_summary, key_entities)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            session_id,
            datetime.now().isoformat(),
            turn_number,
            question,
            summary,
            json.dumps(entities)
        ))

        conn.commit()
        conn.close()

        return {"summary": summary, "entities": entities}

    def get_contextual_prompt_prefix(self, session_id: str, max_turns: int = 3) -> str:
        """Get compressed conversation context for prompt"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT turn_number, question, answer_summary, key_entities
            FROM conversation_context
            WHERE session_id = ?
            ORDER BY turn_number DESC
            LIMIT ?
        """, (session_id, max_turns))

        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return ""

        context_parts = ["Previous conversation context:"]
        for row in reversed(rows):
            turn_num, question, summary, entities_json = row
            entities = json.loads(entities_json)
            context_parts.append(
                f"Turn {turn_num}: User asked about {', '.join(entities[:3])}. {summary}"
            )

        return "\n".join(context_parts) + "\n\n"


class GroundednessEvaluator:
    """Evaluate answer quality using RAGAS and custom metrics"""

    def __init__(self, client: OpenAI):
        self.client = client

    def evaluate_with_ragas(self, question: str, answer: str,
                            contexts: List[str], run_id: str) -> Dict:
        """Use RAGAS for groundedness evaluation"""
        os.environ['OPENAI_API_KEY'] = OPENAI_KEY
        evaluator_llm = ChatOpenAI(model="gpt-4o-mini",
                                   openai_api_key=OPENAI_KEY)
        data = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
        }

        dataset = Dataset.from_dict(data)

        result = evaluate(
            dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                # context_precision,
            ], llm=evaluator_llm
        )

        scores = {
            "faithfulness": result["faithfulness"],
            "answer_relevancy": result["answer_relevancy"],
            # "context_precision": result["context_precision"],
        }

        return scores

    def evaluate_citation_quality(self, answer: str, num_sources: int) -> Dict:
        """Evaluate citation coverage and quality"""
        import re

        citations = re.findall(r'\[Source (\d+)\]', answer)
        unique_citations = set(citations)
        citation_count = len(citations)
        unique_citation_count = len(unique_citations)

        word_count = len(answer.split())
        citation_density = (citation_count / word_count * 100) if word_count > 0 else 0
        source_coverage = unique_citation_count / num_sources if num_sources > 0 else 0

        return {
            "total_citations": citation_count,
            "unique_citations": unique_citation_count,
            "citation_density_per_100_words": citation_density,
            "source_coverage_pct": source_coverage * 100,
            "citation_diversity": unique_citation_count / citation_count if citation_count > 0 else 0
        }

    def evaluate_answer_completeness(self, question: str, answer: str) -> Dict:
        """Use LLM to judge answer completeness"""

        prompt = f"""Evaluate this answer on a scale of 1-10 for each criterion:

Question: {question}
Answer: {answer}

Rate the answer on:
1. Completeness (1-10): Does it fully address all aspects of the question?
2. Specificity (1-10): Does it provide concrete numbers, dates, and details?
3. Clarity (1-10): Is it well-structured and easy to understand?
4. Confidence (1-10): Does it appropriately note limitations and uncertainties?

Return ONLY a JSON object:
{{"completeness": X, "specificity": Y, "clarity": Z, "confidence": W, "reasoning": "brief explanation"}}
"""

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )

        try:
            scores = json.loads(response.choices[0].message.content)
            return scores
        except:
            return {
                "completeness": 5,
                "specificity": 5,
                "clarity": 5,
                "confidence": 5,
                "reasoning": "Failed to parse evaluation"
            }

    def evaluate_all(self, question: str, answer: str, contexts: List[str],
                     num_sources: int, run_id: str) -> Dict:
        """Run all evaluation metrics"""

        try:
            ragas_scores = self.evaluate_with_ragas(question, answer, contexts, run_id)
        except Exception as e:
            print(f"⚠️  RAGAS evaluation failed: {e}")
            ragas_scores = {"faithfulness": 0, "answer_relevancy": 0}

        citation_scores = self.evaluate_citation_quality(answer, num_sources)
        completeness_scores = self.evaluate_answer_completeness(question, answer)

        return {
            **ragas_scores,
            **citation_scores,
            **completeness_scores
        }


class TenKAgent:
    """
    Production 10-K RAG Agent
    100% ALIGNED with Pinecone upload script schema
    """

    def __init__(self, experiment_name: str = "10k-rag-agent",
                 pinecone_index_name: str = "your-index-name",
                 gcp_project_id: str = None,
                 gcp_location: str = "us-central1",
                 vertex_model: str = "text-embedding-004",
                 max_embedding_budget: int = MAX_BUDGET_USD):

        info = {

        }
        creds = service_account.Credentials.from_service_account_info(info)

        # Initialize safe embedder
        self.embedder = SafeVertexAIEmbedder(
            project_id=gcp_project_id,
            location=gcp_location,
            model_name=vertex_model,
            credentials=creds,  # Supports service account
            max_cost=MAX_BUDGET_USD  # Budget tracking
        )

        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.index = self.pc.Index(INDEX_NAME)
        self.client = OpenAI(api_key=OPENAI_KEY)

        mlflow.set_tracking_uri("file:///C:/Users/db234/PycharmProjects/EdgarFilingsBod/Python/AgenticSystem/mlruns")
        mlflow.set_experiment(experiment_name)

        self.citation_manager = CitationManager()
        self.metadata_router = MetadataRouter(self.client)
        self.conversation_manager = ConversationManager()
        self.evaluator = GroundednessEvaluator(self.client)

        self.config = {
            "decomposition_model": "gpt-4o-mini",
            "embedding_model": vertex_model,
            "reasoning_model": "o1",
            "top_k": 5,
            "max_sources_for_reasoning": 15,
            "pinecone_index": pinecone_index_name,
            "enable_groundedness_eval": True,
            "enable_conversation_context": True,
            "enable_metadata_routing": True,
        }

    def decompose_query(self, user_question: str, run_id: str,
                        conversation_context: str = "") -> List[Dict]:
        """Break down complex question into sub-queries"""

        with mlflow.start_run(run_id=run_id, nested=True):
            mlflow.set_tag("step", "decomposition")

            # mlflow.langchain.log_model(
            #     lc_model=self.config['decomposition_model'],
            #     artifact_path="llm_model",
            #     registered_model_name="Financial-Chatbot-Decomposition-Model"  # 🎯 This makes it appear in the Models tab
            # )

            # mlflow.langchain.autolog()
            #
            # mlflow.llama_index.autolog()

            start_time = time.time()

            context_prefix = f"{conversation_context}\nCurrent question: " if conversation_context else ""

            prompt = f"""{context_prefix}You are a financial analyst assistant. Break down this complex question into 2-4 specific sub-queries that can be answered by searching SEC 10-K filings.

User Question: {user_question}

Return a JSON array of sub-queries with this structure:
[
  {{"query": "specific search query", "ticker": "stock ticker (AMD, AAPL, MSFT) or 'multiple'", "reasoning": "why this sub-query is needed"}},
  ...
]

Use stock TICKERS not company names.
"""

            prompt_artifact = f"""'context_prefix' You are a financial analyst assistant. Break down this complex question into 2-4 specific sub-queries that can be answered by searching SEC 10-K filings.

            User Question: 'user_question'

            Return a JSON array of sub-queries with this structure:
            [
              {{"query": "specific search query", "ticker": "stock ticker (AMD, AAPL, MSFT) or 'multiple'", "reasoning": "why this sub-query is needed"}},
              ...
            ]

            Use stock TICKERS not company names.
            """

            mlflow.genai.register_prompt(
                name="System-Instruction-Template-Decomposition",
                template=prompt_artifact,
                # description="The base system instruction for the LLM.",
                commit_message="Initial commit of base template"
            )

            with mlflow.start_span(name='DecompositionModel', span_type=SpanType.LLM) as span:
                span.set_attribute("llm.model_name", self.config['decomposition_model'])
                span.set_attribute("llm.temperature", 0.3)
                span.set_inputs({"prompt": prompt, "user_query": user_question})

            response = self.client.chat.completions.create(
                model=self.config["decomposition_model"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )

            latency = time.time() - start_time

            try:
                sub_queries = json.loads(response.choices[0].message.content)
                mlflow.log_metric("decomposition_latency_sec", latency)
                mlflow.log_metric("num_subqueries", len(sub_queries))

                print(f"\n🧠 Decomposed into {len(sub_queries)} sub-queries:")
                for sq in sub_queries:
                    print(f"  - [{sq.get('ticker', 'multiple')}] {sq['query']}")

                return sub_queries
            except json.JSONDecodeError as e:
                mlflow.log_param("decomposition_error", str(e))
                return [{"query": user_question, "ticker": "multiple", "reasoning": "fallback"}]

    def search_with_metadata_routing(self, query_text: str, user_question: str,
                                     run_id: str) -> List[Dict]:
        """Search with intelligent metadata routing - 100% ALIGNED WITH PINECONE SCHEMA"""

        with mlflow.start_run(run_id=run_id, nested=True):
            mlflow.set_tag("step", "metadata_aware_retrieval")

            # mlflow.langchain.autolog()
            #
            # mlflow.llama_index.autolog()

            metadata_reqs = self.metadata_router.extract_metadata_requirements(user_question)
            mlflow.log_dict(metadata_reqs, "metadata_requirements.json")

            print(f"\n🔍 Metadata requirements: {json.dumps(metadata_reqs, indent=2)}")

            filters = self.metadata_router.build_pinecone_filters(metadata_reqs)

            adjusted_top_k = self.metadata_router.adjust_top_k(
                metadata_reqs,
                base_top_k=self.config["top_k"]
            )
            mlflow.log_metric("adjusted_top_k", adjusted_top_k)
            with mlflow.start_span(name="MetaDataFilters", span_type=SpanType.RETRIEVER) as span:
                span.set_inputs({
                    # Sending the filters object directly to the span's inputs
                    "filter": filters
                })

            print(f"📊 Using top_k={adjusted_top_k} (base: {self.config['top_k']})")

            # embedding_response = self.client.embeddings.create(
            #     model=self.config["embedding_model"],
            #     input=query_text
            # )
            # query_embedding = embedding_response.data[0].embedding
            with mlflow.start_span(name="Embedding Generation", span_type="EMBEDDING") as span:

                # Log the model *parameter* as an attribute/tag
                span.set_attribute("embedding_model_name", "vertexai")

                # Execute the embedding call, which is the actual work you want to measure
                # embeddings = embedder.get_embeddings(data)

                with mlflow.start_span(name="Embedding Generation2", span_type="EMBEDDING") as span:
                    # Log the model *parameter* as an attribute/tag
                    span.set_attribute("embedding_model_name", "vertexai")

                    # Execute the embedding call, which is the actual work you want to measure
                    # embeddings = embedder.get_embeddings(data)
                    query_embedding = self.embedder.get_text_embedding(query_text)
                    # Log metrics related to the *call*
                    mlflow.log_metric("embedding_latency_s", 0.5)
                    span.set_attribute("input_text_length", len(query_text[0]))

                    print("Logged embedding span.")

                # Log metrics related to the *call*
                mlflow.log_metric("embedding_latency_s", 0.5)
                span.set_attribute("input_text_length", len(query_text[0]))

                print("Logged embedding span.")

            all_results = []

            for i, filter_dict in enumerate(filters):
                print(f"\n   Executing search {i + 1}/{len(filters)}")
                if filter_dict:
                    print(f"   Filter: {json.dumps(filter_dict, indent=6)}")

                results = self.index.query(
                    vector=query_embedding,
                    top_k=adjusted_top_k,
                    include_metadata=True,
                    filter=filter_dict if filter_dict else None
                )

                for match in results["matches"]:
                    # ALIGNED WITH PINECONE SCHEMA - checks all possible field variations
                    all_results.append({
                        "text": match["metadata"].get("text", ""),
                        "ticker": match["metadata"].get("ticker", "Unknown"),
                        # Check multiple possible section field names
                        "section": (match["metadata"].get("xbrl_title") or
                                    match["metadata"].get("statement_name") or
                                    match["metadata"].get("stmt_type") or
                                    match["metadata"].get("statement_type") or
                                    "Unknown"),
                        "year": match["metadata"].get("year", "Unknown"),
                        "score": match["score"],
                        "full_metadata": match["metadata"]
                    })

                mlflow.log_metric(f"filter_{i}_results", len(results["matches"]))

            all_results.sort(key=lambda x: x["score"], reverse=True)

            seen = set()
            unique_results = []
            for r in all_results:
                key = r["text"][:200]
                if key not in seen:
                    seen.add(key)
                    unique_results.append(r)

            mlflow.log_metric("total_unique_results", len(unique_results))

            return unique_results[:adjusted_top_k]

    def search_pinecone_simple(self, query_text: str, ticker_filter: str = None,
                               top_k: int = 5, run_id: str = None) -> List[Dict]:
        """Simple search without metadata routing - 100% ALIGNED WITH PINECONE SCHEMA"""

        with mlflow.start_run(run_id=run_id, nested=True):
            mlflow.set_tag("step", "simple_retrieval")
            # mlflow.langchain.autolog()
            #
            # mlflow.llama_index.autolog()

            # embedding_response = self.client.embeddings.create(
            #     model=self.config["embedding_model"],
            #     input=query_text
            # )
            # query_embedding = embedding_response.data[0].embedding
            query_embedding = self.embedder.get_text_embedding(query_text)
            filter_dict = {}
            if ticker_filter and ticker_filter.lower() != "multiple":
                filter_dict = {"ticker": {"$eq": ticker_filter}}

            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict if filter_dict else None
            )

            mlflow.log_metric("num_results", len(results['matches']))

            return [
                {
                    "text": match["metadata"].get("text", ""),
                    "ticker": match["metadata"].get("ticker", "Unknown"),
                    # Check multiple possible section field names
                    "section": (match["metadata"].get("xbrl_title") or
                                match["metadata"].get("statement_name") or
                                match["metadata"].get("stmt_type") or
                                match["metadata"].get("statement_type") or
                                "Unknown"),
                    "year": match["metadata"].get("year", "Unknown"),
                    "score": match["score"],
                    "full_metadata": match["metadata"]
                }
                for match in results["matches"]
            ]

    def synthesize_answer_with_reasoning(self, user_question: str,
                                         all_results: List[Dict],
                                         run_id: str) -> Dict:
        """Use O1 for reasoning with enhanced source tracking"""

        with mlflow.start_run(run_id=run_id, nested=True):
            mlflow.set_tag("step", "synthesis")
            # mlflow.langchain.autolog()
            #
            # mlflow.llama_index.autolog()
            start_time = time.time()

            citations = []
            context_parts = []

            for i, result in enumerate(all_results[:self.config["max_sources_for_reasoning"]], 1):
                citation = self.citation_manager.create_citation_reference(
                    result["full_metadata"],
                    i
                )
                citations.append(citation)

                context_parts.append(
                    f"[Source {i}] {result['ticker']} - {result['section']} ({result['year']}) "
                    f"[Relevance: {result['score']:.3f}]\n"
                    f"{result['text']}\n"
                )

            context = "\n".join(context_parts)

            prompt = f"""You are an expert financial analyst conducting competitive intelligence research using SEC 10-K filings.

User Question: {user_question}

Retrieved 10-K Excerpts:
{context}

Your task:
1. Analyze the retrieved excerpts carefully for relevant financial data, strategic insights, and competitive positioning
2. Compare and contrast information across different companies when relevant
3. Identify patterns, trends, and strategic differences
4. Note any data quality issues, missing information, or contradictions
5. Provide a comprehensive answer with specific citations

Requirements:
- ALWAYS cite sources using [Source N] format for factual claims
- Include specific numbers, percentages, and years when available
- Provide strategic insights beyond just reporting numbers
- Note any limitations in the data or analysis
- If comparing companies, structure comparison clearly
- Be precise and avoid speculation beyond what the data supports

Provide your analysis and answer:"""
            prompt_artifact = f"""You are an expert financial analyst conducting competitive intelligence research using SEC 10-K filings.

User Question: 'user_question'

Retrieved 10-K Excerpts:
'context'

Your task:
1. Analyze the retrieved excerpts carefully for relevant financial data, strategic insights, and competitive positioning
2. Compare and contrast information across different companies when relevant
3. Identify patterns, trends, and strategic differences
4. Note any data quality issues, missing information, or contradictions
5. Provide a comprehensive answer with specific citations

Requirements:
- ALWAYS cite sources using [Source N] format for factual claims
- Include specific numbers, percentages, and years when available
- Provide strategic insights beyond just reporting numbers
- Note any limitations in the data or analysis
- If comparing companies, structure comparison clearly
- Be precise and avoid speculation beyond what the data supports

Provide your analysis and answer:"""

            # mlflow.langchain.log_model(
            #     lc_model=self.config['reasoning_model'],
            #     artifact_path="llm_model",
            #     registered_model_name="Financial-Chatbot-Reasoning-Model"  # 🎯 This makes it appear in the Models tab
            # )

            mlflow.genai.register_prompt(
                name="System-Instruction-Template-ReasoningModel",
                template=prompt_artifact,
                # description="The base system instruction for the LLM.",
                commit_message="Initial commit of base template"
            )

            with mlflow.start_span(name="ReasoningModel", span_type="LLM") as span:
                span.set_attribute("llm.model_name", self.config['reasoning_model'])
                span.set_attribute("llm.temperature", 0.3)
                span.set_inputs({"prompt": prompt, "user_query": user_question})

            print(f"\n🤔 Using O1 for deep reasoning...")

            response = self.client.chat.completions.create(
                model=self.config["reasoning_model"],
                messages=[{"role": "user", "content": prompt}]
            )

            answer = response.choices[0].message.content
            reasoning_latency = time.time() - start_time

            mlflow.log_metric("reasoning_latency_sec", reasoning_latency)
            mlflow.log_metric("reasoning_tokens", response.usage.completion_tokens)

            reasoning_cost = (response.usage.prompt_tokens * 0.000015 +
                              response.usage.completion_tokens * 0.00006)
            mlflow.log_metric("reasoning_cost", reasoning_cost)

            mlflow.log_dict([{k: v for k, v in c.items() if k != 'metadata'} for c in citations], "citations.json")

            return {
                "answer": answer,
                "reasoning_tokens": response.usage.completion_tokens,
                "reasoning_cost": reasoning_cost,
                "contexts": [r["text"] for r in all_results[:self.config["max_sources_for_reasoning"]]],
                "citations": citations
            }

    def format_answer_with_hyperlinks(self, answer: str, citations: List[Dict],
                                      format_type: str = "markdown") -> str:
        """Replace [Source N] references with hyperlinked citations"""
        import re

        formatted_answer = answer

        source_refs = re.findall(r'\[Source (\d+)\]', answer)
        unique_sources = sorted(set(int(s) for s in source_refs))

        if format_type == "markdown":
            reference_section = "\n\n---\n\n### 📚 Sources\n\n"

            for source_num in unique_sources:
                citation = next((c for c in citations if c["source_number"] == source_num), None)
                if citation:
                    formatted_citation = self.citation_manager.format_citation_for_output(
                        citation, "markdown"
                    )
                    reference_section += f"{source_num}. {formatted_citation}\n"

            formatted_answer = answer + reference_section

        elif format_type == "html":
            for source_num in unique_sources:
                citation = next((c for c in citations if c["source_number"] == source_num), None)
                if citation and citation.get("file_url"):
                    pattern = f"\\[Source {source_num}\\]"
                    replacement = f'<a href="{citation["file_url"]}" target="_blank" title="{citation["citation_text"]}">[Source {source_num}]</a>'
                    formatted_answer = re.sub(pattern, replacement, formatted_answer)

            formatted_answer += "\n\n<hr>\n\n<h3>📚 Sources</h3>\n<ul>\n"
            for source_num in unique_sources:
                citation = next((c for c in citations if c["source_number"] == source_num), None)
                if citation:
                    formatted_citation = self.citation_manager.format_citation_for_output(
                        citation, "html"
                    )
                    formatted_answer += f"<li>{formatted_citation}</li>\n"
            formatted_answer += "</ul>\n"

        return formatted_answer

    def clean_context_for_ragas(self, contexts: List[str]) -> List[str]:
        """Convert raw CSV/table contexts into readable text for RAGAS evaluation"""
        cleaned = []

        for context in contexts:
            # Remove Table Context prefix if present
            text = context
            if text.startswith("Table Context:"):
                text = text.split("\n", 1)[1] if "\n" in text else text

            # Replace CSV formatting with readable text
            text = text.replace("\r\n", " ")
            text = text.replace(",,,", " ")
            text = text.replace("  ", " ")

            # For very structured data, try to extract key facts
            # Keep only first 1000 chars to avoid token limits
            text = text[:1000]

            cleaned.append(text)

        return cleaned

    def answer(self, user_question: str, user_id: str = "anonymous",
               session_id: Optional[str] = None,
               output_format: str = "markdown") -> Dict:
        """
        Main agent loop with metadata routing and hyperlinked citations
        100% ALIGNED WITH PINECONE UPLOAD SCHEMA
        """

        if not session_id:
            session_id = hashlib.md5(f"{user_id}_{datetime.now().date()}".encode()).hexdigest()[:16]

        with mlflow.start_run(run_name=f"query_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
            run_id = run.info.run_id
            start_time = time.time()

            mlflow.log_param("user_question", user_question)
            mlflow.log_param("user_id", user_id)
            mlflow.log_param("session_id", session_id)
            mlflow.log_param("output_format", output_format)

            print(f"\n{'=' * 60}")
            print(f"❓ Question: {user_question}")
            print(f"👤 User: {user_id} | Session: {session_id}")
            print(f"🔖 Run ID: {run_id}")
            print(f"{'=' * 60}")

            try:
                conversation_context = ""
                if self.config["enable_conversation_context"]:
                    conversation_context = self.conversation_manager.get_contextual_prompt_prefix(
                        session_id, max_turns=3
                    )
                    if conversation_context:
                        print(f"📚 Using conversation context from previous turns")

                sub_queries = self.decompose_query(user_question, run_id, conversation_context)

                all_results = []

                if self.config["enable_metadata_routing"]:
                    if sub_queries:
                        main_results = self.search_with_metadata_routing(
                            query_text=sub_queries[0]["query"],
                            user_question=user_question,
                            run_id=run_id
                        )
                        all_results.extend(main_results)

                        for sq in sub_queries[1:]:
                            results = self.search_pinecone_simple(
                                query_text=sq["query"],
                                ticker_filter=sq.get("ticker"),
                                top_k=self.config["top_k"],
                                run_id=run_id
                            )
                            all_results.extend(results)
                else:
                    for sq in sub_queries:
                        results = self.search_pinecone_simple(
                            query_text=sq["query"],
                            ticker_filter=sq.get("ticker"),
                            top_k=self.config["top_k"],
                            run_id=run_id
                        )
                        all_results.extend(results)

                seen_texts = set()
                unique_results = []
                for r in all_results:
                    text_snippet = r["text"][:200]
                    if text_snippet not in seen_texts:
                        seen_texts.add(text_snippet)
                        unique_results.append(r)

                mlflow.log_metric("total_results_retrieved", len(all_results))
                mlflow.log_metric("unique_results_after_dedup", len(unique_results))

                synthesis = self.synthesize_answer_with_reasoning(
                    user_question, unique_results, run_id
                )

                answer = synthesis["answer"]
                citations = synthesis["citations"]

                formatted_answer = self.format_answer_with_hyperlinks(
                    answer, citations, output_format
                )
                clean_context = self.clean_context_for_ragas(synthesis["contexts"])
                # print(f"DEBUG synthesis contexts are {synthesis["contexts"]}")
                eval_scores = {}
                if self.config["enable_groundedness_eval"]:
                    print(f"\n📊 Evaluating answer quality...")

                    eval_scores = self.evaluator.evaluate_all(
                        question=user_question,
                        answer=answer,
                        contexts=clean_context,
                        num_sources=len(unique_results),
                        run_id=run_id
                    )

                    for metric_name, score in eval_scores.items():
                        if isinstance(score, (int, float)):
                            mlflow.log_metric(f"eval_{metric_name}", score)

                    print(f"   Faithfulness: {eval_scores.get('faithfulness', 'N/A')}")
                    print(f"   Citation coverage: {eval_scores.get('source_coverage_pct', 'N/A'):.1f}%")

                total_latency = time.time() - start_time
                total_cost = synthesis["reasoning_cost"]

                mlflow.log_metric("total_pipeline_latency_sec", total_latency)
                mlflow.log_metric("total_cost_usd", total_cost)
                mlflow.set_tag("status", "success")

                self.conversation_manager.log_query(
                    user_id=user_id,
                    session_id=session_id,
                    question=user_question,
                    answer=formatted_answer,
                    run_id=run_id,
                    cost=total_cost,
                    latency=total_latency,
                    num_sources=len(unique_results),
                    model_used=self.config["reasoning_model"],
                    metadata={
                        "eval_scores": eval_scores,
                        "sub_queries": sub_queries,
                        "citations": [{k: v for k, v in c.items() if k != 'metadata'} for c in citations]
                    }
                )

                history = self.conversation_manager.get_conversation_history(session_id)
                turn_number = len(history) + 1

                self.conversation_manager.summarize_conversation_turn(
                    question=user_question,
                    answer=answer,
                    session_id=session_id,
                    turn_number=turn_number,
                    client=self.client
                )

                result = {
                    "question": user_question,
                    "answer": formatted_answer,
                    "raw_answer": answer,
                    "run_id": run_id,
                    "session_id": session_id,
                    "evaluation_scores": eval_scores,
                    "num_sources": len(unique_results),
                    "total_cost": total_cost,
                    "total_latency": total_latency,
                    "citations": citations,
                    "top_sources": unique_results[:5]
                }

                return result

            except Exception as e:
                mlflow.set_tag("status", "failed")
                mlflow.log_param("error", str(e))
                import traceback
                traceback.print_exc()
                raise


# ============================================================================
# FASTAPI ENDPOINT
# ============================================================================

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="10-K RAG Agent API - Fully Aligned")


class QueryRequest(BaseModel):
    question: str
    user_id: str = "anonymous"
    session_id: Optional[str] = None
    output_format: str = "markdown"
    enable_metadata_routing: bool = True


class QueryResponse(BaseModel):
    question: str
    answer: str
    run_id: str
    citations: List[Dict]
    evaluation_scores: Dict
    total_cost: float
    total_latency: float
    num_sources: int


@app.post("/query", response_model=QueryResponse)
async def query_agent(request: QueryRequest):
    """Query the RAG agent with metadata routing and hyperlinked citations"""
    try:
        agent = TenKAgent(
            experiment_name="10k-rag-production",
            pinecone_index_name=INDEX_NAME,
            gcp_project_id=GCP_PROJECT_ID,
            gcp_location=GCP_LOCATION,
            vertex_model=VERTEX_MODEL,  # Optional
            max_embedding_budget=50
        )

        agent.config["enable_metadata_routing"] = request.enable_metadata_routing

        result = agent.answer(
            user_question=request.question,
            user_id=request.user_id,
            session_id=request.session_id,
            output_format=request.output_format
        )

        return QueryResponse(
            question=result["question"],
            answer=result["answer"],
            run_id=result["run_id"],
            citations=result["citations"],
            evaluation_scores=result["evaluation_scores"],
            total_cost=result["total_cost"],
            total_latency=result["total_latency"],
            num_sources=result["num_sources"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.get("/")
async def root():
    return {
        "message": "10-K RAG Agent API - 100% Aligned with Pinecone Upload Schema",
        "features": [
            "Metadata-aware retrieval routing (handles both stmt_type and statement_type)",
            "Hyperlinked citations to source files",
            "Conversation context management",
            "RAGAS groundedness evaluation",
            "MLflow experiment tracking",
            "Multi-format output (markdown/html/text)"
        ],
        "schema_alignment": {
            "ticker": "✅ Top-level field",
            "year": "✅ Top-level field (not 'yr')",
            "stmt_type": "✅ From statement_info",
            "statement_type": "✅ From xbrl_context",
            "xbrl_title": "✅ Top-level field",
            "statement_name": "✅ Top-level field",
            "has_numeric_data": "✅ Boolean flag",
            "is_xbrl": "✅ Boolean flag",
            "text": "✅ Content field (1500 char preview)",
            "folder": "✅ For hyperlinks"
        }
    }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("10-K RAG AGENT - 100% ALIGNED WITH PINECONE UPLOAD SCHEMA")
    print("=" * 80)
    print("\nSchema Alignment:")
    print("✅ ticker (top-level)")
    print("✅ year (top-level, not 'yr')")
    print("✅ stmt_type AND statement_type (handles both)")
    print("✅ xbrl_title, statement_name (for section display)")
    print("✅ has_numeric_data, is_xbrl (boolean filters)")
    print("✅ text (content field)")
    print("✅ folder (for file:// hyperlinks)")
    print("=" * 80)

    # Initialize agent
    agent = TenKAgent(
        experiment_name="10k-rag-production",
        pinecone_index_name=INDEX_NAME,
        gcp_project_id=GCP_PROJECT_ID,
        gcp_location=GCP_LOCATION,
        vertex_model=VERTEX_MODEL,  # Optional
        max_embedding_budget=50
    )

    # Test query
    test_query = "What are Microsoft's risk factors?"

    print(f"\n\nRunning test query: {test_query}")
    print("=" * 80)

    result = agent.answer(
        user_question=test_query,
        user_id="test_user",
        output_format="markdown"
    )

    print(f"\n{'=' * 80}")
    print(f"ANSWER:")
    print(f"{'=' * 80}")
    print(result['answer'])
    print(f"\n💰 Cost: ${result['total_cost']:.4f} | ⏱️  Latency: {result['total_latency']:.2f}s")
    print(f"📚 Sources: {result['num_sources']}")

    if result['evaluation_scores']:
        print(f"\n📊 Quality Scores:")
        for metric, score in result['evaluation_scores'].items():
            if isinstance(score, (int, float)):
                print(f"   {metric}: {score:.3f}")

    print(f"\n🔖 MLflow Run ID: {result['run_id']}")

    print(f"\n{'=' * 80}")
    print("✅ AGENT IS 100% ALIGNED WITH YOUR PINECONE UPLOAD SCHEMA!")
    print(f"{'=' * 80}\n")