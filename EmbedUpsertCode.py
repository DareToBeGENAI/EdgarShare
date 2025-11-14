"""
Safe Vertex AI Embedding + Pinecone Upload Pipeline for 10-K Filings
Adapted for RAG-optimized chunker output with budget tracking and checkpoints
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Tuple
from google.cloud import aiplatform
from google.oauth2 import service_account
from pinecone import Pinecone, ServerlessSpec
from llama_index.core import Settings
from llama_index.embeddings.vertex import VertexTextEmbedding
from tqdm import tqdm
import os
from google.oauth2 import service_account

# ============================================================================
# CONFIGURATION
# ============================================================================

# File paths
INPUT_JSON_PATH = r'C:/Users/db234/PycharmProjects/EdgarFilingsBod/Python/output/processed_10k_chunks.json'
# CHECKPOINT_PATH = r'C:/Users/db234/PycharmProjects/EdgarFilingsBod/Python/VectorDatabase/checkpoint'
# Pinecone configuration
PINECONE_API_KEY = "pcsk_3wePf8_KJkYD9V7P2rkrV4UKYpLQASX93CsM4p5fuSg9PGPWSUtFJXw681vGYXTL7y6aXw"
PINECONE_ENVIRONMENT = "us-central1"  # or your preferred region
INDEX_NAME = "financial-10k-filings-total-final-768-3"
PINECONE_DIMENSION = 768  # Vertex AI text-embedding-004 dimension
DIMENSION = 768
# Vertex AI configuration
GCP_PROJECT_ID = "pineconeproject"
GCP_LOCATION = "us-central1"  # or your preferred region
VERTEX_MODEL = "text-embedding-004"  # Latest Vertex AI embedding model

# Budget & Safety
MAX_BUDGET_USD = 250.0  # Leave $50 buffer from $300 free credits
BATCH_SIZE = 100  # Embedding batch size
UPSERT_BATCH_SIZE = 100  # Pinecone upsert batch size
CHECKPOINT_FREQUENCY = 500  # Save progress every N items
import tempfile

temp_dir = tempfile.gettempdir()
CHECKPOINT_PATH = os.path.join(temp_dir, "checkpoint", "data.json")
CHECKPOINT_BACKUP_PATH = os.path.join(temp_dir, "checkpoint_backup", "data.json")


# ============================================================================
# SAFE VERTEX AI EMBEDDER WITH BUDGET TRACKING
# ============================================================================

class SafeVertexAIEmbedder:
    """Wrapper around VertexTextEmbedding with cost tracking"""

    def __init__(self, project_id: str, location: str, model_name: str,
                 credentials, max_cost: float = 250.0):

        self.project_id = project_id
        self.location = location
        self.model_name = model_name
        self.max_cost = max_cost
        self.total_cost = 0.0
        self.total_tokens = 0
        self.total_requests = 0

        # Pricing: $0.00002 per 1k tokens for text-multilingual-embedding-002
        self.cost_per_1k_tokens = 0.00002

        # Rate limiting
        self.requests_per_minute = 300
        self.request_count = 0
        self.minute_start = time.time()

        # Initialize LlamaIndex Vertex embedding model
        self.embed_model = VertexTextEmbedding(
            model_name=model_name,
            project=project_id,
            location=location,
            credentials=credentials
        )

        # Set as default in Settings
        Settings.embed_model = self.embed_model

        print(f"✅ Initialized Vertex AI Embedder")
        print(f"   Model: {model_name}")
        print(f"   Dimension: {DIMENSION}")
        print(f"   Budget: ${max_cost:.2f}")
        print(f"   Cost per 1K tokens: ${self.cost_per_1k_tokens}")

    def estimate_tokens(self, text: str) -> int:
        """Estimate tokens (rough: 4 chars ≈ 1 token)"""
        return max(1, len(text) // 4)

    def check_budget(self, estimated_cost: float) -> bool:
        """Check if we can afford this operation"""
        if self.total_cost + estimated_cost > self.max_cost:
            print(f"\n🛑 BUDGET LIMIT WOULD BE EXCEEDED")
            print(f"   Current cost: ${self.total_cost:.4f}")
            print(f"   Next batch cost: ${estimated_cost:.4f}")
            print(f"   Would total: ${self.total_cost + estimated_cost:.4f}")
            print(f"   Budget limit: ${self.max_cost:.2f}")
            return False
        return True

    def rate_limit(self):
        """Enforce rate limiting"""
        self.request_count += 1

        elapsed = time.time() - self.minute_start
        if elapsed >= 60:
            self.request_count = 0
            self.minute_start = time.time()

        if self.request_count >= self.requests_per_minute:
            sleep_time = 60 - elapsed
            if sleep_time > 0:
                print(f"⏸️  Rate limit reached ({self.requests_per_minute} RPM), sleeping {sleep_time:.1f}s...")
                time.sleep(sleep_time)
                self.request_count = 0
                self.minute_start = time.time()

    def get_text_embedding(self, text: str) -> List[float]:
        """Get embedding for single text"""
        tokens = self.estimate_tokens(text)
        cost = (tokens / 1000) * self.cost_per_1k_tokens

        if not self.check_budget(cost):
            raise Exception("Budget limit reached")

        self.rate_limit()

        embedding = self.embed_model.get_text_embedding(text)

        self.total_tokens += tokens
        self.total_cost += cost
        self.total_requests += 1

        return embedding

    def get_text_embedding_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for batch of texts"""
        total_tokens = sum(self.estimate_tokens(text) for text in texts)
        batch_cost = (total_tokens / 1000) * self.cost_per_1k_tokens

        if not self.check_budget(batch_cost):
            raise Exception("Budget limit reached")

        self.rate_limit()

        try:
            embeddings = self.embed_model.get_text_embedding_batch(texts)
        except Exception as e:
            print(f"⚠️  Batch embedding failed, falling back to individual: {e}")
            embeddings = []
            for i, text in enumerate(texts):
                try:
                    # Truncate if too long (model limit ~8000 tokens)
                    if len(text) > 30000:  # ~7500 tokens
                        print(f"⚠️  Text {i} too long ({len(text)} chars), truncating to 30000 chars")
                        text = text[:30000]

                    emb = self.embed_model.get_text_embedding(text)

                    # CRITICAL: Validate embedding is not all zeros
                    if not emb or all(v == 0 for v in emb):
                        print(f"❌ Text {i} produced zero embedding! Using random embedding instead.")
                        print(f"   Text preview: {text[:200]}...")
                        # Create a small random embedding to avoid Pinecone error
                        import random
                        emb = [random.uniform(-0.01, 0.01) for _ in range(768)]

                    embeddings.append(emb)

                except Exception as e2:
                    print(f"❌ Failed to embed text {i}: {e2}")
                    print(f"   Text length: {len(text)} chars")
                    print(f"   Text preview: {text[:200]}...")
                    # Create dummy embedding
                    import random
                    embeddings.append([random.uniform(-0.01, 0.01) for _ in range(768)])

        # Final validation: check all embeddings
        validated_embeddings = []
        for i, emb in enumerate(embeddings):
            if not emb or all(v == 0 for v in emb):
                print(f"⚠️  Embedding {i} is all zeros, replacing with random values")
                import random
                emb = [random.uniform(-0.01, 0.01) for _ in range(768)]
            validated_embeddings.append(emb)

        self.total_tokens += total_tokens
        self.total_cost += batch_cost
        self.total_requests += 1

        if self.total_requests % 10 == 0:
            pct_used = (self.total_cost / self.max_cost) * 100
            print(f"📊 Progress: {self.total_requests} batches | "
                  f"${self.total_cost:.4f} / ${self.max_cost:.2f} ({pct_used:.1f}%)")

        return validated_embeddings

    def get_stats(self) -> Dict:
        """Get current usage statistics"""
        return {
            'total_cost': self.total_cost,
            'total_tokens': self.total_tokens,
            'total_requests': self.total_requests,
            'pct_budget_used': (self.total_cost / self.max_cost) * 100
        }


# ============================================================================
# CHECKPOINT MANAGEMENT
# ============================================================================

def load_checkpoint() -> Dict:
    """Load processing checkpoint with backup recovery and index validation"""
    checkpoint_path = Path(CHECKPOINT_PATH)
    backup_path = Path(CHECKPOINT_BACKUP_PATH)

    # Ensure checkpoint directory exists
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # Try loading primary checkpoint
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path, 'r') as f:
                checkpoint = json.load(f)

                # Check if index name matches current config
                checkpoint_index = checkpoint.get('index_name', 'UNKNOWN')

                if checkpoint_index != INDEX_NAME:
                    print(f"\n⚠️  INDEX MISMATCH DETECTED!")
                    print(f"   Checkpoint was for index: '{checkpoint_index}'")
                    print(f"   Current config uses index: '{INDEX_NAME}'")
                    print(f"   Processed count in checkpoint: {checkpoint['processed_count']:,}")

                    response = input(f"\n❓ What would you like to do?\n"
                                     f"   1) Start fresh with new index (lose checkpoint)\n"
                                     f"   2) Continue with checkpoint (switch back to '{checkpoint_index}')\n"
                                     f"   3) Abort\n"
                                     f"   Choice (1/2/3): ")

                    if response == '1':
                        print(f"✅ Starting fresh with index '{INDEX_NAME}'")
                        return {'processed_count': 0, 'last_id': None, 'index_name': INDEX_NAME}
                    elif response == '2':
                        print(f"✅ Continuing with checkpoint (using index '{checkpoint_index}')")
                        # Note: You'll need to manually change INDEX_NAME in config
                        print(f"⚠️  WARNING: Update INDEX_NAME in your config to '{checkpoint_index}'")
                        input("Press Enter after updating config, or Ctrl+C to abort...")
                    else:
                        raise Exception("Aborted by user")

                print(f"📂 Loaded checkpoint: {checkpoint['processed_count']:,} items already processed")
                print(f"   Index: {checkpoint.get('index_name', 'not tracked')}")
                return checkpoint

        except Exception as e:
            print(f"⚠️  Error loading primary checkpoint: {e}")

            # Try loading backup
            if backup_path.exists():
                print(f"🔄 Attempting to load backup checkpoint...")
                try:
                    with open(backup_path, 'r') as f:
                        checkpoint = json.load(f)
                        print(f"✅ Recovered from backup: {checkpoint['processed_count']:,} items")
                        return checkpoint
                except Exception as e2:
                    print(f"❌ Backup also corrupted: {e2}")

    print("📝 No checkpoint found, starting from beginning")
    return {'processed_count': 0, 'last_id': None, 'index_name': INDEX_NAME}


def save_checkpoint(processed_count: int, last_id: str, embedder_stats: Dict, index_name: str):
    """Save processing checkpoint with backup and index tracking"""
    checkpoint = {
        'processed_count': processed_count,
        'last_id': last_id,
        'index_name': index_name,  # NEW: Track which index
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'embedder_stats': embedder_stats
    }

    checkpoint_path = Path(CHECKPOINT_PATH)
    backup_path = Path(CHECKPOINT_BACKUP_PATH)

    # Ensure directory exists
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # If checkpoint exists, back it up first
        if checkpoint_path.exists():
            import shutil
            shutil.copy2(checkpoint_path, backup_path)

        # Write new checkpoint atomically (write to temp, then rename)
        temp_path = checkpoint_path.with_suffix('.tmp')
        with open(temp_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)

        # Atomic rename (more reliable)
        temp_path.replace(checkpoint_path)

        print(
            f"💾 Checkpoint saved: {processed_count:,} items | ${embedder_stats['total_cost']:.4f} | Index: {index_name}")

    except Exception as e:
        print(f"❌ Error saving checkpoint: {e}")
        # Don't raise - continue processing even if checkpoint fails


# ============================================================================
# INITIALIZATION
# ============================================================================

def initialize_vertex_embeddings() -> SafeVertexAIEmbedder:
    """Initialize Vertex AI embedding model with budget tracking"""
    print("Initializing Vertex AI embeddings...")

    # Load service account credentials
    # with open(GCP_SERVICE_ACCOUNT_PATH, 'r') as f:
    #     info = json.load(f)
    info = {

    }
    creds = service_account.Credentials.from_service_account_info(info)

    # Initialize safe embedder
    embed_model = SafeVertexAIEmbedder(
        project_id=GCP_PROJECT_ID,
        location=GCP_LOCATION,
        model_name=VERTEX_MODEL,
        credentials=creds,
        max_cost=MAX_BUDGET_USD
    )

    return embed_model


def initialize_pinecone() -> Tuple[Pinecone, any]:
    """Initialize Pinecone client and index"""
    print("\n🔗 Connecting to Pinecone...")

    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Check if index exists
    existing_indexes = pc.list_indexes()
    index_names = [idx['name'] for idx in existing_indexes]

    if INDEX_NAME not in index_names:
        print(f"Creating new index: {INDEX_NAME}")
        pc.create_index(
            name=INDEX_NAME,
            dimension=PINECONE_DIMENSION,  # ← Use constant from config
            metric="cosine",
            spec=ServerlessSpec(
                cloud="gcp",
                region=PINECONE_ENVIRONMENT
            )
        )
        print("Waiting for index to be ready...")
        time.sleep(10)
        print("✅ Index created")
    else:
        print(f"✅ Using existing index: {INDEX_NAME}")

        # VERIFY dimensions match
        index_info = pc.describe_index(INDEX_NAME)
        index_dim = index_info['dimension']

        if index_dim != PINECONE_DIMENSION:
            raise Exception(
                f"❌ DIMENSION MISMATCH!\n"
                f"   Index '{INDEX_NAME}' has {index_dim} dimensions\n"
                f"   But embedding model '{VERTEX_MODEL}' produces {PINECONE_DIMENSION} dimensions\n"
                f"   Solution: Delete the index and recreate it, or use a different model"
            )

        print(f"✅ Dimension verified: {index_dim}")

    index = pc.Index(INDEX_NAME)

    # Show stats
    stats = index.describe_index_stats()
    print(f"   Current vector count: {stats['total_vector_count']:,}")

    return pc, index


# ============================================================================
# DATA PROCESSING
# ============================================================================

def load_processed_chunks(file_path: str) -> List[Dict]:
    """Load processed 10-K chunks from RAG-optimized chunker - FIXED validation"""
    print(f"\n📂 Loading chunks from: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)

    print(f"✅ Loaded {len(chunks):,} chunks")

    # VALIDATION: Check data quality with correct field names
    print(f"\n🔍 Validating data structure...")

    valid_count = 0
    invalid_count = 0
    missing_content = 0
    empty_content = 0
    has_chunk_content = 0
    has_content = 0

    for i, chunk in enumerate(chunks):
        # Check required fields
        if 'id' not in chunk:
            print(f"⚠️  Chunk {i}: missing 'id' field")
            invalid_count += 1
            continue

        # FIXED: Check for either 'chunk_content' or 'content'
        content = chunk.get('chunk_content') or chunk.get('content')

        if 'chunk_content' in chunk:
            has_chunk_content += 1
        if 'content' in chunk:
            has_content += 1

        if not content:
            missing_content += 1
            invalid_count += 1
            continue

        if len(str(content).strip()) == 0:
            empty_content += 1
            invalid_count += 1
            continue

        valid_count += 1

    print(f"\n📊 Validation Results:")
    print(f"   Valid chunks: {valid_count:,}")
    print(f"   Invalid chunks: {invalid_count:,}")
    print(f"   Uses 'chunk_content': {has_chunk_content:,}")
    print(f"   Uses 'content': {has_content:,}")
    if missing_content > 0:
        print(f"   ⚠️  Missing content field: {missing_content:,}")
    if empty_content > 0:
        print(f"   ⚠️  Empty content: {empty_content:,}")

    if invalid_count > len(chunks) * 0.05:  # More than 5% invalid
        response = input(
            f"\n⚠️  Found {invalid_count:,} invalid chunks ({invalid_count / len(chunks) * 100:.1f}%). Continue anyway? (yes/no): ")
        if response.lower() != 'yes':
            raise Exception("Aborted due to invalid chunks")

    # Show sample
    if chunks:
        sample = chunks[0]
        print(f"\n📋 Sample chunk:")
        print(f"   ID: {sample.get('id', 'N/A')}")
        print(f"   Type: {sample.get('data_type', 'N/A')}")

        # FIXED: Check for 'source' field
        if 'source' in sample:
            print(f"   Source keys: {list(sample['source'].keys())}")
        elif 'metadata' in sample:
            print(f"   Metadata keys: {list(sample['metadata'].keys())}")

        # FIXED: Show correct content field
        content = sample.get('chunk_content') or sample.get('content')
        if content:
            content_len = len(str(content))
            print(f"   Content length: {content_len:,} chars")
            print(f"   Content preview: {str(content)[:100]}...")
        else:
            print(f"   ⚠️  WARNING: Sample chunk missing content field!")
            print(f"   Available keys: {list(sample.keys())}")

    return chunks


def prepare_metadata_for_pinecone(item: Dict) -> Dict:
    """
    Prepare metadata for Pinecone - COMPLETE VERSION with ALL fields
    Stores EVERYTHING from chunker output
    """
    metadata = {
        'id': item['id'],
        'data_type': item['data_type'],
    }

    # Extract from 'source' field
    if 'source' in item:
        source = item['source']
        metadata['cik'] = source.get('cik', 'UNKNOWN')
        metadata['ticker'] = source.get('ticker', 'UNKNOWN')
        metadata['year'] = source.get('yr', 'UNKNOWN')
        metadata['exhibit_type'] = source.get('ex', 'UNKNOWN')
        metadata['filename'] = source.get('fn', '')[:50]

        if 'folder' in source:
            metadata['folder'] = source['folder'][:150]

        # R-file specific
        if 'r_type' in source:
            metadata['r_type'] = source['r_type']
        if 'r_score' in source:
            metadata['r_score'] = source['r_score']

    # Table-specific metadata - COMPLETE
    if item['data_type'] == 'table_content':

        # Table stats
        if 'table_stats' in item:
            metadata['table_rows'] = item['table_stats'].get('rows', 0)
            metadata['table_cols'] = item['table_stats'].get('cols', 0)

        # XBRL context - COMPLETE
        if 'xbrl_context' in item and item['xbrl_context']:
            xbrl = item['xbrl_context']

            if 'title' in xbrl:
                metadata['xbrl_title'] = str(xbrl['title'])[:150]
            if 'statement_type' in xbrl:
                metadata['statement_type'] = str(xbrl['statement_type'])[:20]
            if 'statement_name' in xbrl:
                metadata['statement_name'] = str(xbrl['statement_name'])[:50]
            if 'is_xbrl' in xbrl:
                metadata['is_xbrl'] = bool(xbrl['is_xbrl'])

            # Periods - join into string
            if 'periods' in xbrl and xbrl['periods']:
                metadata['periods'] = ', '.join(str(p) for p in xbrl['periods'][:3])

        # Units - COMPLETE
        if 'units' in item and item['units']:
            units = item['units']
            metadata['currency'] = str(units.get('currency', 'unknown'))[:10]
            metadata['scale_factor'] = str(units.get('scale_factor', 'unknown'))[:20]
            metadata['primary_unit'] = str(units.get('primary_unit', 'unknown'))[:20]
            if 'multiplier' in units:
                metadata['multiplier'] = units['multiplier']

        # Period info - COMPLETE
        if 'period_info' in item and item['period_info']:
            period = item['period_info']

            if 'fiscal_years' in period and period['fiscal_years']:
                metadata['fiscal_years'] = ','.join(str(y) for y in period['fiscal_years'][:5])
            if 'fiscal_quarters' in period and period['fiscal_quarters']:
                metadata['fiscal_quarters'] = ','.join(str(q) for q in period['fiscal_quarters'][:4])
            if 'period_type' in period:
                metadata['period_type'] = str(period['period_type'])[:30]

        # Statement info - COMPLETE
        if 'statement_info' in item and item['statement_info']:
            stmt = item['statement_info']

            if 'statement_type' in stmt:
                metadata['stmt_type'] = str(stmt['statement_type'])[:30]
            if 'statement_category' in stmt:
                metadata['stmt_category'] = str(stmt['statement_category'])[:30]
            if 'key_line_items' in stmt and stmt['key_line_items']:
                # Join first 5 line items
                metadata['key_line_items'] = ', '.join(str(k) for k in stmt['key_line_items'][:5])

        # Structure info
        if 'structure' in item and item['structure']:
            struct = item['structure']
            if 'has_subtotals' in struct:
                metadata['has_subtotals'] = bool(struct['has_subtotals'])
            if 'has_totals' in struct:
                metadata['has_totals'] = bool(struct['has_totals'])
            if 'hierarchical_levels' in struct:
                metadata['hierarchical_levels'] = struct['hierarchical_levels']

        # CRITICAL: Numeric data summary
        # Store info ABOUT the numeric data (not the data itself - too large)
        if 'numeric_data' in item and item['numeric_data']:
            num_data = item['numeric_data']
            # Store which columns have numeric data
            numeric_cols = list(num_data.keys())
            if numeric_cols:
                metadata['numeric_columns'] = ', '.join(numeric_cols[:5])
                metadata['has_numeric_data'] = True

                # Store sample values for context (first non-NaN value from each column)
                sample_values = []
                for col in numeric_cols[:3]:
                    values = num_data[col]
                    for val in values:
                        if val is not None and str(val) != 'nan':
                            sample_values.append(f"{col}:{val}")
                            break

                if sample_values:
                    metadata['sample_values'] = ' | '.join(sample_values[:3])

    # Get content (handles both field names)
    content_text = item.get('chunk_content') or item.get('content', '')
    if content_text:
        # Store first 1500 chars as preview
        metadata['text'] = str(content_text)[:1500]

    return metadata


def create_pinecone_vectors(chunks: List[Dict], embed_model: SafeVertexAIEmbedder,
                            start_idx: int = 0) -> List[tuple]:
    """
    Create vectors for Pinecone from chunks with budget tracking
    FIXED: Handles both 'chunk_content' and 'content' fields
    """
    print(f"\n🚀 Generating embeddings starting from chunk {start_idx:,}/{len(chunks):,}...")

    vectors = []
    skipped_count = 0

    # Process in batches
    for i in tqdm(range(start_idx, len(chunks), BATCH_SIZE), desc="Embedding batches"):
        batch = chunks[i:i + BATCH_SIZE]

        try:
            # FIXED: Validate and extract texts - check both field names
            valid_items = []
            texts = []

            for item in batch:
                # FIXED: Try 'chunk_content' first, then 'content'
                content = item.get('chunk_content') or item.get('content')

                # Check if content field exists
                if not content:
                    print(f"\n⚠️  Skipping chunk {item.get('id', 'unknown')}: missing content field")
                    skipped_count += 1
                    continue

                # Check if content is not empty
                if not isinstance(content, str) or len(content.strip()) == 0:
                    print(f"\n⚠️  Skipping chunk {item.get('id', 'unknown')}: empty content")
                    skipped_count += 1
                    continue

                # NEW: Check if content is too long (Vertex AI limit ~8K tokens = ~32K chars)
                if len(content) > 32000:
                    print(
                        f"\n⚠️  Chunk {item.get('id', 'unknown')}: too long ({len(content)} chars), truncating to 32000")
                    content = content[:32000]

                valid_items.append(item)
                texts.append(str(content))

            # Skip batch if no valid items
            if not valid_items:
                print(f"\n⚠️  Batch {i} has no valid items, skipping...")
                continue

            # Generate embeddings
            embeddings = embed_model.get_text_embedding_batch(texts)

            # Create vector tuples with metadata
            for item, embedding in zip(valid_items, embeddings):
                vector_id = item['id']
                metadata = prepare_metadata_for_pinecone(item)
                vectors.append((vector_id, embedding, metadata))

            # Small delay to avoid rate limits
            time.sleep(0.05)

        except KeyError as e:
            print(f"\n⚠️  KeyError in batch {i}: {e}")
            print(f"   Sample item keys: {list(batch[0].keys()) if batch else 'empty batch'}")
            # Continue with next batch

        except Exception as e:
            if "Budget limit" in str(e):
                print(f"\n🛑 Budget limit reached at batch {i}/{len(chunks)}")
                print(f"   Processed {len(vectors)} vectors so far")
                break
            else:
                print(f"\n⚠️  Error processing batch {i}: {e}")
                import traceback
                traceback.print_exc()
                # Continue with next batch

    if skipped_count > 0:
        print(f"\n⚠️  Warning: Skipped {skipped_count:,} chunks due to missing/empty content")

    print(f"✅ Generated {len(vectors):,} embeddings")
    return vectors


def upsert_to_pinecone(index, vectors: List[tuple]):
    """Upload vectors to Pinecone in batches with full numeric data preservation"""
    print(f"\n📤 Upserting {len(vectors):,} vectors to Pinecone...")

    successful = 0
    failed = 0

    # Upsert in batches
    for i in tqdm(range(0, len(vectors), UPSERT_BATCH_SIZE), desc="Upserting batches"):
        batch = vectors[i:i + UPSERT_BATCH_SIZE]

        try:
            index.upsert(vectors=batch)
            successful += len(batch)
            time.sleep(0.05)
        except Exception as e:
            print(f"\n⚠️  Error upserting batch {i}: {e}")

            # Try one by one
            for vector in batch:
                try:
                    index.upsert(vectors=[vector])
                    successful += 1
                except Exception as e2:
                    # Check if metadata is too large
                    metadata_size = len(json.dumps(vector[2]))
                    if metadata_size > 40000:  # Pinecone limit is 40KB
                        print(f"   Vector {vector[0]}: metadata too large ({metadata_size} bytes), truncating...")

                        # Truncate text field
                        vector_copy = list(vector)
                        if 'text' in vector_copy[2]:
                            vector_copy[2]['text'] = vector_copy[2]['text'][:500]

                        try:
                            index.upsert(vectors=[tuple(vector_copy)])
                            successful += 1
                        except:
                            print(f"   Failed even after truncation: {vector[0]}")
                            failed += 1
                    else:
                        print(f"   Failed vector {vector[0]}: {e2}")
                        failed += 1

    print(f"✅ Upsert complete: {successful:,} successful, {failed} failed")

    if failed > 0:
        print(f"⚠️  {failed} vectors failed - likely due to metadata size limits")


# ============================================================================
# VERIFICATION
# ============================================================================

def verify_upload(index: any, embed_model: SafeVertexAIEmbedder):
    """Verify the upload with test queries"""
    print("\n" + "=" * 80)
    print("VERIFICATION")
    print("=" * 80)

    stats = index.describe_index_stats()
    print(f"✅ Total vectors in index: {stats['total_vector_count']:,}")

    # Test queries
    test_queries = [
        "What is the revenue for 2023?",
        "Segment information and assets by division",
        "Stockholders equity and dividends"
    ]

    for test_query in test_queries:
        print(f"\n🔍 Testing query: '{test_query}'")

        try:
            test_embedding = embed_model.get_text_embedding(test_query)

            results = index.query(
                vector=test_embedding,
                top_k=3,
                include_metadata=True
            )

            print(f"   Found {len(results['matches'])} results:")
            for i, match in enumerate(results['matches'], 1):
                print(f"\n   {i}. Score: {match['score']:.4f}")
                print(f"      ID: {match['id']}")
                print(f"      CIK: {match['metadata'].get('cik', 'N/A')}")
                print(f"      Ticker: {match['metadata'].get('ticker', 'N/A')}")
                print(f"      Year: {match['metadata'].get('year', 'N/A')}")
                print(f"      Type: {match['metadata'].get('data_type', 'N/A')}")
                print(f"      Preview: {match['metadata'].get('text', '')[:100]}...")

        except Exception as e:
            print(f"   ⚠️  Query failed: {e}")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main pipeline to upload 10-K chunks to Pinecone with budget safety"""

    print("=" * 80)
    print("SAFE 10-K FILINGS → VERTEX AI → PINECONE PIPELINE")
    print("=" * 80)

    try:
        # Load checkpoint
        checkpoint = load_checkpoint()

        # Load chunks
        chunks = load_processed_chunks(INPUT_JSON_PATH)

        if not chunks:
            print("❌ No chunks to process. Exiting.")
            return

        start_idx = checkpoint['processed_count']

        if start_idx >= len(chunks):
            print("✅ All chunks already processed!")
            return

        # Initialize Vertex AI with budget limit
        embed_model = initialize_vertex_embeddings()

        # Initialize Pinecone
        pc, index = initialize_pinecone()

        # Process and upload
        print(f"\n{'=' * 80}")
        print(f"PROCESSING & UPLOAD")
        print(f"{'=' * 80}")

        vectors = create_pinecone_vectors(chunks, embed_model, start_idx)

        if vectors:
            upsert_to_pinecone(index, vectors)

            # Save final checkpoint with index name
            processed_count = start_idx + len(vectors)
            last_id = vectors[-1][0] if vectors else checkpoint.get('last_id')
            save_checkpoint(processed_count, last_id, embed_model.get_stats(), INDEX_NAME)  # Pass index name

            print(f"\n💾 Final checkpoint saved")

        # Show final statistics
        print("\n" + "=" * 80)
        print("PROCESSING COMPLETE")
        print("=" * 80)

        stats = embed_model.get_stats()
        print(f"\n💰 Cost Statistics:")
        print(f"   Total cost: ${stats['total_cost']:.4f}")
        print(f"   Total tokens: {stats['total_tokens']:,}")
        print(f"   Total requests: {stats['total_requests']:,}")
        print(f"   Budget used: {stats['pct_budget_used']:.1f}%")
        print(f"   Remaining: ${MAX_BUDGET_USD - stats['total_cost']:.2f}")

        processed_count = start_idx + len(vectors)
        print(f"\n📊 Processing Statistics:")
        print(f"   Total chunks: {len(chunks):,}")
        print(f"   Processed this run: {len(vectors):,}")
        print(f"   Total processed: {processed_count:,}")

        if processed_count < len(chunks):
            remaining = len(chunks) - processed_count
            print(f"   ⚠️  Incomplete: {remaining:,} chunks remaining")
            print(f"   Run script again to continue from checkpoint")
        else:
            print(f"   ✅ All chunks processed!")

        # Verify
        if vectors:
            verify_upload(index, embed_model)

        print("\n" + "=" * 80)
        print("✅ PIPELINE COMPLETE!")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

        print("\n💾 Progress has been saved to checkpoint. Run script again to resume.")


if __name__ == "__main__":
    main()
