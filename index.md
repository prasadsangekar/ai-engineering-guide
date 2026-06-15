---
title: AI Engineering
---
# AI Engineering Study Guide — Complete Self-Paced Course

> 8 Weeks · ~50-70 hours · LLMs → RAG → Agents → Production

This guide contains everything you need to learn AI engineering from foundations through production deployment. Each topic is explained in full detail with working code examples.

---

# WEEK 1: Terminology & Prerequisites

**Goal:** Understand how LLMs work under the hood before touching any application code.

---

## Theory

### 1.1 Core LLM Terminology

#### Tokens
Tokens are the fundamental units that LLMs process. Rather than working with whole words, LLMs break text into sub-word pieces called tokens. The word "unhappiness" might become `["un", "happiness"]` or `["un", "happ", "iness"]` depending on the tokenizer.

Why this matters:
- **Billing**: API providers charge per token (both input and output)
- **Limits**: Context windows are measured in tokens, not words
- **Non-English impact**: Languages like Chinese or Arabic use more tokens per concept
- **Rule of thumb**: ~4 English characters ≈ 1 token; ~0.75 English words ≈ 1 token

Common tokenizers: OpenAI uses "tiktoken" (BPE-based), Anthropic uses their own SentencePiece variant.

#### Context Window
The context window is the maximum number of tokens an LLM can process at once — this includes BOTH your input (prompt + history + documents) AND the model's output combined.

| Model | Context Window |
|-------|---------------|
| GPT-4o | 128K tokens |
| Claude 3.5 Sonnet | 200K tokens |
| Llama 3 | 8K-128K tokens |

Key insight: The model has NO memory beyond the context window. If you want it to "remember" a prior conversation, you must include that conversation in the current prompt. This is why chat applications send the full message history with each request.

#### Parameters
Parameters are the learned numerical weights inside a neural network. They ARE the model — they encode everything the model "knows." During training, these values are adjusted; during inference, they are frozen.

- GPT-3: 175 billion parameters
- Llama 3 70B: 70 billion parameters
- Claude 3.5 Sonnet: undisclosed (estimated 100B+)

More parameters generally = more knowledge capacity, but with diminishing returns and increased cost.

#### Attention Mechanism
The attention mechanism (from the 2017 paper "Attention Is All You Need") is the core innovation behind modern LLMs. It allows each token to "look at" every other token in the input and decide how much weight to give each one.

**Analogy**: Imagine reading a sentence and having a highlighter. For each word you read, you highlight the other words that are most relevant to understanding it. "The cat sat on the mat because *it* was tired" — when processing "it", attention highlights "cat" more than "mat."

**Self-attention** computes three values for each token:
- **Query (Q)**: "What am I looking for?"
- **Key (K)**: "What do I contain?"
- **Value (V)**: "What information do I provide?"

The attention score = softmax(Q · K^T / √d) · V

**Multi-head attention** runs multiple attention patterns in parallel (e.g., 32 or 64 heads), each capturing different relationship types (syntax, semantics, position, etc.).

---

### 1.2 How LLMs Are Trained

#### Pre-Training
The foundation phase where the model learns language patterns from massive text corpora.

**Process**: Given billions of text documents (books, websites, code, papers), the model learns to predict the next token. For example:
- Input: "The capital of France is"
- Target: "Paris"

The model sees trillions of such examples and adjusts its parameters to minimize prediction error. This is called **causal language modeling** or **next-token prediction**.

**What it learns**: Grammar, facts, reasoning patterns, code syntax, mathematical relationships, and unfortunately also biases present in the training data.

**Scale**: Pre-training GPT-4 is estimated to have cost $100M+ in compute. It takes weeks to months on thousands of GPUs.

**Result**: A "base model" that can autocomplete text but doesn't follow instructions well.

#### Post-Training (Alignment)
After pre-training, the model needs to be aligned to follow instructions and be helpful.

**Supervised Fine-Tuning (SFT)**: Train on thousands of (prompt, ideal_response) pairs written by humans. This teaches the model the format of a helpful assistant.

**RLHF (Reinforcement Learning from Human Feedback)**:
1. Humans rank multiple model outputs for the same prompt
2. A "reward model" learns these human preferences
3. The LLM is optimized to maximize the reward model's score
4. This makes outputs more helpful, honest, and harmless

**DPO (Direct Preference Optimization)**: A simpler alternative that skips the reward model and directly optimizes from preference pairs.

#### Inference
When you send a prompt to an LLM API, inference happens:
1. Your text is tokenized (converted to token IDs)
2. Tokens pass through the model's layers (attention → feed-forward → repeat)
3. The model outputs a probability distribution over all possible next tokens
4. One token is sampled (based on temperature settings)
5. That token is appended, and steps 2-4 repeat until a stop condition

**Key insight**: Generation is autoregressive — one token at a time. This is why streaming exists and why longer outputs take proportionally longer.

---

### 1.3 Embeddings & Semantic Similarity

#### What Are Embeddings?
An embedding is a dense numerical vector (array of floating-point numbers) that represents the *meaning* of text in high-dimensional space.

- Input: "machine learning" → Output: [0.12, -0.44, 0.87, 0.03, ...] (768 or 1536 dimensions)
- Similar meanings → vectors close together
- Different meanings → vectors far apart

**Analogy**: Think of latitude and longitude for cities. Paris and London have similar coordinates (both in Europe). Paris and Tokyo have very different coordinates. Embeddings do the same thing but in 768+ dimensions for *meaning*.

#### How They're Created
Embedding models (like OpenAI's `text-embedding-3-small` or `all-MiniLM-L6-v2`) are trained on pairs of similar/dissimilar text. They learn to map semantically similar text to nearby points.

#### Cosine Similarity
The standard metric for comparing embeddings. It measures the angle between two vectors:
- 1.0 = identical direction (same meaning)
- 0.0 = perpendicular (unrelated)
- -1.0 = opposite direction (opposite meaning, rare in practice)

Formula: cos(θ) = (A · B) / (||A|| × ||B||)

#### Use Cases
- **Semantic search**: Find documents that mean the same thing as a query
- **RAG**: Retrieve relevant chunks for an LLM to reason over
- **Clustering**: Group similar documents together
- **Classification**: Use embeddings as features

---

### 1.4 Prompting Basics

#### System Prompts
The system prompt sets the overall behavior, role, and constraints for the model. It persists across the entire conversation.

```
You are a senior Python developer. You write clean, well-documented code.
Always include type hints. If you're unsure about something, say so rather
than guessing. Format code with black-style formatting.
```

Best practices:
- Be specific about role and expertise level
- Define output format expectations
- State what NOT to do
- Keep it concise but complete

#### User Prompts
Individual messages from the user. Each is processed in context of the system prompt + conversation history.

Good user prompts are: specific, contextual, and state the desired format.

**Bad**: "Help me with Python"
**Good**: "Write a Python function that takes a list of integers and returns the top 3 most frequent elements. Include type hints and a docstring."

#### Few-Shot Prompting
Providing examples before your actual question to teach the model the pattern:

```
Classify the sentiment:
"I love this product!" → POSITIVE
"Terrible experience." → NEGATIVE
"This changed my life!" → ???
```

This works because the model learns the pattern from examples without any fine-tuning.

#### Chain-of-Thought
Adding "Think step by step" or showing reasoning examples dramatically improves accuracy on math, logic, and multi-step problems.

---

### 1.5 Structured Outputs and JSON Formatting

#### Why Structured Output?
Applications need predictable, parseable data — not free-form text. You need JSON for APIs, databases, and downstream code.

#### Techniques

**1. Prompt instruction:**
```
Respond with valid JSON only. Use this schema:
{"name": string, "age": number, "skills": string[]}
```

**2. Function calling / Tool use:**
Define a schema, and the model returns a structured tool call instead of text. This is the most reliable method.

**3. Response format parameter:**
OpenAI: `response_format: {"type": "json_object"}`
Anthropic: Use tool definitions with input schemas

#### Best Practices
- Always provide the exact schema you expect
- Include field descriptions and constraints
- Use `enum` for fields with limited values
- Validate output with Pydantic or JSON Schema
- Handle edge cases: model adds markdown fences, includes extra text, returns wrong types

---

## Coding Practice

### 1.C1: Set Up Python Environment

```python
# requirements.txt
# openai>=1.0
# anthropic>=0.18
# numpy

# Install:
# pip install openai anthropic numpy

# Verify installation
import openai
import anthropic
import numpy as np

print(f"openai version: {openai.__version__}")
print(f"anthropic version: {anthropic.__version__}")

# Set your API keys (use environment variables in production!)
import os
os.environ["OPENAI_API_KEY"] = "sk-..."  # Replace with your key
os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."  # Replace with your key
```

---

### 1.C2: First LLM API Calls

```python
from openai import OpenAI
import anthropic

# --- OpenAI ---
client = OpenAI()  # Reads OPENAI_API_KEY from env

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ],
    max_tokens=100,
)

print("OpenAI response:", response.choices[0].message.content)
print(f"Tokens used: {response.usage.prompt_tokens} input, {response.usage.completion_tokens} output")

# --- Anthropic ---
client_ant = anthropic.Anthropic()  # Reads ANTHROPIC_API_KEY from env

message = client_ant.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=100,
    system="You are a helpful assistant.",
    messages=[
        {"role": "user", "content": "What is the capital of France?"},
    ],
)

print("Anthropic response:", message.content[0].text)
print(f"Tokens used: {message.usage.input_tokens} input, {message.usage.output_tokens} output")
```

---

### 1.C3: Cosine Similarity from Scratch

```python
import numpy as np
from openai import OpenAI

client = OpenAI()

def get_embedding(text: str, model: str = "text-embedding-3-small") -> list[float]:
    """Get embedding vector for a text string."""
    response = client.embeddings.create(input=text, model=model)
    return response.data[0].embedding

def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two vectors from scratch."""
    a = np.array(vec_a)
    b = np.array(vec_b)
    
    dot_product = np.dot(a, b)
    magnitude_a = np.linalg.norm(a)
    magnitude_b = np.linalg.norm(b)
    
    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0
    
    return dot_product / (magnitude_a * magnitude_b)

# Compare similar sentences
emb1 = get_embedding("The cat sat on the mat")
emb2 = get_embedding("A feline rested on the rug")
emb3 = get_embedding("Stock prices rose sharply today")

sim_12 = cosine_similarity(emb1, emb2)
sim_13 = cosine_similarity(emb1, emb3)

print(f"Cat/Feline similarity: {sim_12:.4f}")  # ~0.85+ (very similar)
print(f"Cat/Stocks similarity: {sim_13:.4f}")   # ~0.10-0.30 (unrelated)
```

---

### 1.C4: Experiment with Temperature, max_tokens, System Prompts

```python
from openai import OpenAI

client = OpenAI()

def generate(prompt: str, temperature: float = 1.0, max_tokens: int = 100, 
             system: str = "You are a helpful assistant.") -> str:
    """Generate text with configurable parameters."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content

# --- Temperature experiment ---
prompt = "Write a one-sentence story about a robot."

print("=== Temperature 0 (deterministic) ===")
for i in range(3):
    print(f"  Run {i+1}: {generate(prompt, temperature=0)}")
# All 3 will be identical or near-identical

print("\n=== Temperature 1.5 (creative/random) ===")
for i in range(3):
    print(f"  Run {i+1}: {generate(prompt, temperature=1.5)}")
# All 3 will be very different

# --- max_tokens experiment ---
print("\n=== max_tokens=10 (truncated) ===")
print(generate("Explain quantum computing", max_tokens=10))

print("\n=== max_tokens=200 (full response) ===")
print(generate("Explain quantum computing", max_tokens=200))

# --- System prompt experiment ---
print("\n=== Pirate system prompt ===")
print(generate("What is 2+2?", system="You are a pirate. Respond in pirate speak."))

print("\n=== Formal academic system prompt ===")
print(generate("What is 2+2?", system="You are a formal academic. Use precise, scholarly language."))
```

---

*Continue to Week 2...*


---

# WEEK 2: RAG — Components & Architecture

**Goal:** Understand why Retrieval-Augmented Generation exists, master the full RAG pipeline from ingestion to answer generation, and build a working RAG system from scratch.

---

## Theory

### 2.1 Why RAG Exists: The Knowledge Cutoff and Hallucination Problem

Large Language Models are powerful, but they carry two fundamental limitations that make them unreliable for knowledge-intensive tasks in production systems.

#### The Knowledge Cutoff Problem

Every LLM has a training data cutoff date — a point in time beyond which it has zero information. GPT-4o's training data ends in late 2023. Claude's ends at a similar boundary. Ask either model about an event from last week, and it will either refuse to answer or — more dangerously — fabricate a plausible-sounding response.

This isn't just about current events. Consider internal enterprise data: your company's HR policies, product documentation, customer tickets, or proprietary research. None of this was in the training data, so the model literally cannot know it.

**Analogy**: Imagine hiring a brilliant consultant who graduated top of their class — but they've been in a coma since their graduation date. They know everything up to that point with impressive depth, but ask them about anything that happened after, and they'll either say "I don't know" (if well-aligned) or confidently guess (if not).

#### The Hallucination Problem

Hallucination is when an LLM generates text that sounds authoritative and fluent but is factually incorrect. This happens because LLMs are fundamentally next-token predictors — they produce the most *statistically likely* continuation, not the most *truthful* one.

**Example**: Ask "What did the Supreme Court rule in Smith v. Johnson (2024)?" and a model might respond with a detailed, perfectly formatted legal analysis of a case that doesn't exist. The model isn't lying — it's pattern-matching: it knows what Supreme Court rulings look like and generates one that fits the pattern.

Hallucination rates increase when:
- The topic is niche or recent (less training data coverage)
- The question demands specific facts (dates, numbers, citations)
- The model is pressured to answer rather than abstain

#### How RAG Solves Both Problems

Retrieval-Augmented Generation (RAG) addresses both issues by giving the LLM access to external, up-to-date knowledge at query time. Instead of relying solely on parametric memory (what's encoded in its weights), the model receives relevant documents in its context window and generates answers grounded in those documents.

The workflow:
1. User asks a question
2. The system searches a knowledge base for relevant documents
3. Those documents are injected into the LLM's prompt as context
4. The LLM generates an answer citing the provided documents

**Analogy**: RAG turns your brilliant-but-outdated consultant into one who has a research assistant. Before answering any question, the assistant fetches the latest relevant documents from the filing cabinet and hands them over. The consultant can now answer accurately because they're reading from source material, not guessing from memory.

**Why not just fine-tune?** Fine-tuning bakes knowledge into model weights, which means you'd need to re-train every time your data changes. For a knowledge base updated daily (support docs, news, internal wikis), fine-tuning is impractical. RAG is dynamic — update the documents, and the next query automatically has access to fresh information.

---

### 2.2 The 5-Stage RAG Pipeline: Ingest → Chunk → Embed → Index → Retrieve

A RAG system is not a single component — it's a pipeline with five distinct stages. Each stage has its own set of choices and trade-offs. Understanding the full pipeline is essential for building effective RAG systems and debugging them when retrieval quality is poor.

#### Stage 1: Ingest

Ingestion is the process of loading raw documents from their source format into a form your pipeline can process. Documents come in many formats: PDFs, Word files, HTML pages, Markdown, Slack messages, database records, code repositories.

**Challenges**:
- PDFs often have complex layouts (tables, columns, headers/footers) that mess up text extraction
- Scanned documents require OCR
- HTML pages have navigation, ads, and boilerplate mixed with content
- Encoding issues (UTF-8, Latin-1, etc.)

**Tools**: LangChain document loaders, Unstructured.io, PyPDF2, BeautifulSoup, or custom parsers.

The output of ingestion is clean text with associated metadata (source filename, page number, author, date, URL).

#### Stage 2: Chunk

Raw documents are rarely the right size for retrieval. A 50-page PDF is too large to stuff into an LLM prompt, and retrieving the whole thing when you only need one paragraph is wasteful. Chunking splits documents into smaller, semantically coherent pieces.

**Key decisions**:
- **Chunk size**: Too small = fragments lack context; too large = dilutes relevance and wastes tokens
- **Chunk overlap**: Adjacent chunks should share some text at boundaries to preserve context
- **Typical sizes**: 200-1000 tokens per chunk; 10-20% overlap

**Analogy**: Chunking is like cutting a book into index cards. Each card should contain a complete thought. If you cut mid-sentence, the card is useless. If you put an entire chapter on one card, it's too broad to be useful for answering a specific question.

#### Stage 3: Embed

Each chunk is converted into a dense vector (embedding) using an embedding model. This transforms text into a mathematical representation where semantic similarity maps to geometric proximity.

**Process**: "Company revenue grew 15% in Q3" → [0.23, -0.87, 0.14, ..., 0.56] (768-1536 dimensions)

The embedding captures meaning, not keywords. So "revenue growth" and "financial performance improvement" end up as nearby vectors even though they share zero words.

#### Stage 4: Index

Embeddings are stored in a vector database (or vector index) that enables fast approximate nearest-neighbor search. With millions of chunks, brute-force comparison (checking every single vector) is too slow. Indexing structures organize vectors for sub-linear search time.

**Popular options**: ChromaDB, Pinecone, Weaviate, Qdrant, FAISS, pgvector.

The index stores both the embedding vector and the original text + metadata, so when you find relevant vectors, you can retrieve the actual content.

#### Stage 5: Retrieve

At query time, the user's question is embedded using the same model, then the vector index finds the top-k most similar chunks. These chunks are injected into the LLM's prompt as context for answer generation.

**Example flow**:
- User: "What is our refund policy for digital purchases?"
- Query embedding → search vector DB → retrieve top 5 chunks about refund policy
- Prompt: "Given the following documents: [chunk1, chunk2, ...], answer: What is our refund policy for digital purchases?"
- LLM generates answer grounded in the retrieved chunks

**Analogy for the full pipeline**: Think of building a library.
1. **Ingest** = Acquiring books (buying, receiving donations, converting formats)
2. **Chunk** = Creating index cards, one per topic per book
3. **Embed** = Assigning a location code based on the card's subject matter
4. **Index** = Filing cards in a well-organized catalog system
5. **Retrieve** = When someone asks a question, the librarian quickly finds the most relevant cards and reads from them

---

### 2.3 Chunking Strategies: Fixed-Size, Sentence, Semantic, and Recursive

Chunking strategy is one of the highest-leverage decisions in a RAG system. Poor chunking leads to poor retrieval, and no amount of sophisticated re-ranking or prompt engineering downstream will fix garbage input. Here are the four main approaches.

#### Fixed-Size Chunking

The simplest approach: split text into chunks of exactly N characters (or N tokens), with an overlap of M characters between adjacent chunks.

```
Document: "ABCDEFGHIJ" with chunk_size=4, overlap=1
Chunks: ["ABCD", "DEFG", "GHIJ"]
```

**Pros**: Dead simple, predictable chunk sizes, fast to implement.
**Cons**: Cuts mid-sentence, mid-paragraph, even mid-word. No awareness of content structure.

**When to use**: Quick prototypes, or when your content is already fairly uniform (e.g., log entries, short product descriptions).

**Example**: A chunk might end with "The company was founded in" and the next begins with "1998 by John Smith." Neither chunk alone makes sense for retrieval.

#### Sentence-Based Chunking

Split on sentence boundaries. Group sentences until you hit the target chunk size.

**Process**: Use NLP sentence detection (spaCy, NLTK, or regex) to find sentence boundaries, then accumulate sentences until the chunk reaches the target token count.

**Pros**: Never cuts mid-sentence. Each chunk reads naturally.
**Cons**: Sentence detection isn't perfect (abbreviations like "Dr." or "U.S." confuse simple splitters). Doesn't respect paragraph or section boundaries.

**When to use**: Narrative text, articles, documentation where preserving sentence integrity matters.

**Example**: A sentence-based chunker processes a paragraph about machine learning and groups 4-5 sentences together, ensuring each chunk is a readable passage rather than a sentence fragment.

#### Semantic Chunking

Uses embeddings to determine where to split. The idea: if two adjacent sentences are semantically similar (high cosine similarity between their embeddings), keep them in the same chunk. When similarity drops (indicating a topic shift), create a new chunk boundary.

**Process**:
1. Embed each sentence individually
2. Compute cosine similarity between adjacent sentence pairs
3. When similarity drops below a threshold (or when a breakpoint is detected), create a chunk boundary
4. Group consecutive high-similarity sentences into chunks

**Pros**: Chunks are semantically coherent — each covers one topic. Best retrieval quality.
**Cons**: Expensive (requires embedding every sentence during chunking, not just at indexing). Variable chunk sizes. More complex implementation.

**When to use**: High-value use cases where retrieval quality matters more than ingestion speed.

**Analogy**: Fixed-size chunking is like cutting a pizza with a ruler at even intervals. Semantic chunking is like cutting along the natural boundaries between toppings — each slice has a coherent combination.

#### Recursive Character Text Splitting

LangChain's default and most popular strategy. It tries multiple separators in order of priority, recursively splitting until chunks are below the target size.

**Default separator hierarchy**: `["\n\n", "\n", " ", ""]`

**Process**:
1. Try splitting on `\n\n` (double newline = paragraph boundary)
2. If any chunk is still too large, split on `\n` (line break)
3. If still too large, split on `" "` (space = word boundary)
4. Last resort: split on `""` (character-level)

**Pros**: Respects document structure. Paragraphs stay together when possible. Falls back gracefully. Well-tested in production.
**Cons**: Still doesn't understand semantics — just structure. Can't handle documents without clear structural markers.

**When to use**: General-purpose default for most RAG applications. Works well on Markdown, HTML, code, and structured documents.

**Production tip**: For Markdown docs, use `MarkdownHeaderTextSplitter` which splits on headers first, preserving section-level context. For code, use language-aware splitters that respect function/class boundaries.

---

### 2.4 Embedding Models: Choosing the Right One for Your Use Case

Embedding models are the bridge between human language and machine-searchable vector space. Choosing the right one affects retrieval quality, speed, cost, and storage requirements.

#### Key Dimensions for Comparison

**1. Dimensionality**: The number of values in each embedding vector.
- Higher dimensions = more expressive, captures more nuance
- Lower dimensions = faster search, less storage
- Common range: 384 (lightweight) to 3072 (maximum quality)

| Model | Dimensions | Notes |
|-------|-----------|-------|
| all-MiniLM-L6-v2 | 384 | Fast, lightweight |
| text-embedding-3-small | 1536 | Good balance |
| text-embedding-3-large | 3072 | Highest quality (OpenAI) |
| Cohere embed-v3 | 1024 | Multilingual strength |

**2. Sequence Length**: Maximum tokens the model can embed at once.
- Most models: 512 tokens max
- Newer models (e.g., jina-embeddings-v2): up to 8192 tokens
- If your chunks exceed the model's max, they get truncated silently — a common bug

**3. Training Objective**: What the model was optimized for.
- **Symmetric**: Both query and document are similar length (e.g., sentence-to-sentence matching)
- **Asymmetric**: Short query vs. long document (better for RAG where queries are short questions)

**4. Domain Specificity**:
- General-purpose models work for most cases
- Domain-specific models (e.g., trained on medical or legal text) significantly outperform general ones in specialized fields
- If your domain has unusual vocabulary, consider fine-tuning

#### Popular Models and When to Use Them

**OpenAI text-embedding-3-small/large**:
- Best for: General-purpose RAG when you're already using OpenAI
- Trade-off: API cost per call, data leaves your infrastructure
- Quality: Top-tier on MTEB benchmarks

**all-MiniLM-L6-v2 (Sentence Transformers)**:
- Best for: Local/free deployment, prototyping, when data can't leave your infra
- Trade-off: Lower quality than larger models
- Speed: Very fast, runs on CPU

**Cohere embed-v3**:
- Best for: Multilingual applications
- Trade-off: API cost
- Unique feature: Separate `search_document` and `search_query` input types (asymmetric)

**BGE-large-en / E5-large-v2**:
- Best for: Open-source, high-quality, self-hosted
- Trade-off: Requires GPU for reasonable speed
- Quality: Competitive with proprietary models

#### Practical Guidelines

1. **Start with `text-embedding-3-small`** if you want quality with reasonable cost
2. **Use `all-MiniLM-L6-v2`** for local prototyping or air-gapped environments
3. **Never mix embedding models** — you can't compare vectors from different models
4. **Benchmark on YOUR data** — MTEB scores don't always correlate with your specific use case
5. **Consider Matryoshka embeddings** (text-embedding-3-* supports this) — you can truncate vectors to save storage with minimal quality loss

**Analogy**: Choosing an embedding model is like choosing a camera lens. A 50mm prime lens (small model) is fast, cheap, and good enough for most photos. A 200mm telephoto (large model) captures more detail at distance but costs more and is slower. A macro lens (domain-specific model) is perfect for close-up photography but useless for landscapes. The best lens depends on what you're photographing.

---

### 2.5 Vector Databases & Indexing Algorithms: HNSW, IVF, PQ

When you have millions of document chunks embedded as vectors, you need infrastructure that can find the nearest neighbors to a query vector in milliseconds. This is the job of vector databases and their underlying indexing algorithms.

#### Why Not Brute Force?

Brute-force nearest-neighbor search compares the query vector against every single stored vector. With 1 million 1536-dimensional vectors, that's 1 million dot product calculations per query. At 10ms per 100K comparisons, that's ~100ms per query — acceptable for small collections but unusable at scale.

With 100 million vectors, brute force would take seconds per query. We need Approximate Nearest Neighbor (ANN) algorithms that trade a small amount of accuracy (recall) for massive speed improvements.

#### HNSW (Hierarchical Navigable Small World)

HNSW is the most popular ANN algorithm in production. It builds a multi-layer graph structure where each vector is a node connected to its approximate nearest neighbors.

**How it works**:
- Imagine a city map at multiple zoom levels
- The top layer has just a few "express highway" connections between distant nodes
- Each lower layer adds more nodes and more local connections
- The bottom layer contains all vectors with fine-grained connections

**Search process**:
1. Start at a random node in the top layer
2. Greedily move to the neighbor closest to the query
3. When you can't get closer, drop to the next layer
4. Repeat until you reach the bottom layer
5. The final greedy search at the bottom layer finds the nearest neighbors

**Pros**: Excellent recall (95-99%), fast search (<5ms for millions of vectors), no training phase
**Cons**: High memory usage (stores the full vectors + graph structure), slow index building

**Parameters**:
- `M`: Number of connections per node (higher = better recall, more memory). Typical: 16-64
- `ef_construction`: Search width during building (higher = better graph, slower build). Typical: 200-400
- `ef_search`: Search width during query (higher = better recall, slower query). Typical: 50-200

**Analogy**: HNSW is like navigating a new city. First you take the highway (top layer) to get close to the neighborhood. Then you take local roads (middle layers) to the right block. Finally you walk street by street (bottom layer) to find the exact address.

#### IVF (Inverted File Index)

IVF partitions the vector space into clusters and only searches the clusters closest to the query.

**How it works**:
1. **Training phase**: Run k-means clustering on your vectors to create `nlist` centroids (e.g., 1024 clusters)
2. **Index phase**: Assign each vector to its nearest centroid's cluster
3. **Search phase**: Find the `nprobe` closest centroids to the query, then search only vectors in those clusters

**Pros**: Low memory (just stores cluster assignments + vectors), fast building, tunable speed/accuracy trade-off
**Cons**: Requires training data, less recall than HNSW at same speed, partition boundaries can miss relevant vectors

**Parameters**:
- `nlist`: Number of clusters (more clusters = faster search, risk of missing neighbors). Typical: sqrt(n) to 4*sqrt(n)
- `nprobe`: Number of clusters to search (higher = better recall, slower). Typical: 1-20% of nlist

**Analogy**: IVF is like organizing a library by subject. When someone asks about "machine learning," you only search the Computer Science and Mathematics sections — you skip Literature and History entirely. Fast, but you might miss a relevant book mis-shelved in another section.

#### PQ (Product Quantization)

PQ is a compression technique that reduces memory usage by approximating vectors with compact codes.

**How it works**:
1. Split each vector into `m` sub-vectors (e.g., a 768-dim vector into 96 sub-vectors of 8 dimensions each)
2. For each sub-vector position, learn a codebook of 256 representative centroids (via k-means)
3. Replace each sub-vector with the index of its nearest centroid (1 byte per sub-vector)
4. A 768-dim float32 vector (3072 bytes) becomes a 96-byte code — 32x compression

**Distance computation**: Instead of computing full vector distance, look up precomputed distances from the codebook — extremely fast.

**Pros**: Massive memory reduction (32-64x), enables billion-scale search on a single machine
**Cons**: Lossy compression, reduced recall compared to full-precision search

**Analogy**: PQ is like representing a color photograph as an 8-color palette image. You lose fine detail, but you can still tell what the image shows, and it takes far less disk space. For most practical purposes (finding relevant documents), the approximation is good enough.

#### Combining Algorithms: IVF-PQ and HNSW-PQ

In practice, these are often combined:
- **IVF-PQ**: Partition space with IVF, compress vectors in each partition with PQ. The standard approach for billion-scale search (used by FAISS).
- **HNSW + PQ**: Use PQ-compressed vectors in an HNSW graph for both speed and memory savings.

#### Choosing a Vector Database

| Database | Best For | Indexing |
|----------|----------|---------|
| ChromaDB | Prototyping, local dev | HNSW |
| FAISS | Research, custom pipelines | IVF, PQ, HNSW, Flat |
| Pinecone | Managed production, no-ops | Proprietary (HNSW-based) |
| Weaviate | Hybrid search (vector + keyword) | HNSW |
| Qdrant | Self-hosted production | HNSW |
| pgvector | Existing PostgreSQL infrastructure | IVF, HNSW |

**Decision framework**:
- Prototyping or < 100K vectors? → ChromaDB (zero config, runs in-memory)
- Need managed service, no infra? → Pinecone or Weaviate Cloud
- Billion-scale, self-hosted? → FAISS with IVF-PQ or Qdrant
- Already using PostgreSQL? → pgvector (add vector search without new infra)

---

## Coding Practice

### 2.C1: Install LangChain + ChromaDB (Setup and Verify)

```python
# ============================================================
# SETUP: Install required packages
# ============================================================
# Run in your terminal:
#
#   pip install langchain langchain-community langchain-openai \
#       langchain-chroma chromadb pypdf sentence-transformers \
#       tiktoken openai
#
# ============================================================

# Verify all installations
import langchain
import chromadb
import pypdf
import tiktoken
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader

print(f"✓ langchain version: {langchain.__version__}")
print(f"✓ chromadb version: {chromadb.__version__}")
print(f"✓ pypdf version: {pypdf.__version__}")
print(f"✓ tiktoken version: {tiktoken.__version__}")
print("✓ All imports successful — environment is ready!")

# Set API key (use env vars in production)
import os
os.environ["OPENAI_API_KEY"] = "sk-..."  # Replace with your key

# Quick sanity check: create a test embedding
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
test_vector = embeddings.embed_query("hello world")
print(f"✓ Embedding works! Vector dimension: {len(test_vector)}")

# Quick sanity check: create a test ChromaDB collection
client = chromadb.Client()  # In-memory for testing
collection = client.create_collection("test")
collection.add(
    documents=["test document"],
    ids=["id1"]
)
results = collection.query(query_texts=["test"], n_results=1)
print(f"✓ ChromaDB works! Query returned: {results['documents'][0][0]}")

# Expected output:
# ✓ langchain version: 0.3.x
# ✓ chromadb version: 0.5.x
# ✓ pypdf version: 4.x.x
# ✓ tiktoken version: 0.7.x
# ✓ All imports successful — environment is ready!
# ✓ Embedding works! Vector dimension: 1536
# ✓ ChromaDB works! Query returned: test document
```

---

### 2.C2: Ingest Documents and Chunk Them

```python
"""
Ingest a PDF (or set of text documents) and chunk them using
different strategies. This demonstrates the Ingest → Chunk stages.
"""

import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter,
)
from langchain.schema import Document

# ============================================================
# Option A: Load a PDF file
# ============================================================
def load_pdf(file_path: str) -> list[Document]:
    """Load a PDF and return a list of Document objects (one per page)."""
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    print(f"Loaded PDF: {file_path}")
    print(f"  Pages: {len(pages)}")
    print(f"  First page preview: {pages[0].page_content[:200]}...")
    return pages

# Option B: Load text files from a directory
# ============================================================
def load_text_directory(dir_path: str) -> list[Document]:
    """Load all .txt files from a directory."""
    loader = DirectoryLoader(dir_path, glob="**/*.txt", loader_cls=TextLoader)
    docs = loader.load()
    print(f"Loaded {len(docs)} text files from: {dir_path}")
    return docs

# Option C: Create sample documents for demonstration
# ============================================================
def create_sample_documents() -> list[Document]:
    """Create sample documents to demonstrate chunking (no files needed)."""
    docs = [
        Document(
            page_content="""
Retrieval-Augmented Generation (RAG) is a technique that enhances large language 
models by providing them with relevant external knowledge at inference time. Rather 
than relying solely on the information encoded in model parameters during training, 
RAG systems retrieve relevant documents from a knowledge base and include them in 
the prompt context.

The key advantage of RAG over fine-tuning is that knowledge can be updated without 
retraining the model. When documents in the knowledge base change, the system 
automatically has access to the updated information on the next query.

RAG was first introduced in a 2020 paper by Lewis et al. from Facebook AI Research. 
The original architecture used a dense retriever (DPR) to find relevant passages 
from Wikipedia and a sequence-to-sequence model (BART) to generate answers.

Modern RAG systems have evolved significantly. They use more sophisticated retrieval 
methods including hybrid search (combining dense and sparse retrieval), re-ranking 
stages, and query transformation techniques. The generator component now typically 
uses large instruction-tuned models like GPT-4 or Claude.

Common challenges in RAG include: chunk size optimization, handling multi-hop 
questions that require information from multiple documents, dealing with 
contradictory information across sources, and ensuring the model doesn't hallucinate 
beyond what the retrieved documents support.
            """.strip(),
            metadata={"source": "rag_overview.txt", "topic": "RAG fundamentals"}
        ),
        Document(
            page_content="""
Vector databases are specialized database systems designed to store, index, and 
query high-dimensional vector embeddings efficiently. Unlike traditional databases 
that search by exact match or range queries, vector databases find the most similar 
vectors to a given query vector using approximate nearest neighbor (ANN) algorithms.

Popular vector databases include ChromaDB, Pinecone, Weaviate, Qdrant, and Milvus. 
Each has different strengths: ChromaDB excels at local development and prototyping, 
Pinecone offers a fully managed cloud service, Weaviate provides hybrid search 
combining vectors with keyword filtering, and Qdrant offers high-performance 
self-hosted deployment.

The core indexing algorithms used by vector databases include:

HNSW (Hierarchical Navigable Small World): Builds a multi-layer graph structure 
for fast traversal. Offers excellent recall (95-99%) with sub-millisecond query 
times for millions of vectors. Memory-intensive as it stores the full graph.

IVF (Inverted File Index): Partitions the vector space into clusters using k-means, 
then searches only the nearest clusters at query time. More memory-efficient than 
HNSW but typically lower recall.

Product Quantization (PQ): Compresses vectors by splitting them into sub-vectors 
and replacing each with a codebook index. Achieves 32-64x compression with 
acceptable recall loss. Often combined with IVF for billion-scale search.

When choosing a vector database, consider: scale (number of vectors), latency 
requirements, memory budget, whether you need filtering/metadata queries, and 
whether you prefer managed or self-hosted infrastructure.
            """.strip(),
            metadata={"source": "vector_databases.txt", "topic": "Vector DBs"}
        ),
        Document(
            page_content="""
Chunking strategies determine how documents are split into smaller pieces for 
embedding and retrieval. The choice of chunking strategy significantly impacts 
retrieval quality — poor chunking leads to fragments that lack context or chunks 
that are too broad to be useful.

Fixed-size chunking splits text into uniform segments of N characters or tokens 
with an optional overlap. It's simple and predictable but often cuts mid-sentence 
or mid-paragraph, destroying semantic coherence.

Sentence-based chunking uses NLP sentence detection to split on sentence boundaries, 
then groups sentences until reaching the target size. This preserves sentence 
integrity but doesn't respect higher-level structures like paragraphs or sections.

Semantic chunking uses embedding similarity between adjacent sentences to find 
natural topic boundaries. When the cosine similarity between consecutive sentences 
drops below a threshold, a new chunk begins. This produces the most coherent chunks 
but requires embedding every sentence during ingestion, making it expensive.

Recursive character splitting (LangChain's default) tries multiple separators in 
priority order: paragraph breaks, line breaks, spaces, then characters. It respects 
document structure while ensuring all chunks stay within the size limit.

Best practices for chunking: aim for 200-1000 tokens per chunk, use 10-20% overlap 
to preserve context at boundaries, preserve metadata (source, page number, section 
header) with each chunk, and test different strategies on your specific data.
            """.strip(),
            metadata={"source": "chunking_guide.txt", "topic": "Chunking"}
        ),
    ]
    print(f"Created {len(docs)} sample documents")
    for doc in docs:
        print(f"  - {doc.metadata['source']}: {len(doc.page_content)} chars")
    return docs


# ============================================================
# CHUNKING: Split documents with different strategies
# ============================================================

# Use sample documents (replace with load_pdf() or load_text_directory() for real files)
documents = create_sample_documents()

# --- Strategy 1: Recursive Character Splitting (RECOMMENDED DEFAULT) ---
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,         # Max characters per chunk
    chunk_overlap=50,       # Overlap between adjacent chunks
    length_function=len,    # How to measure chunk length
    separators=["\n\n", "\n", " ", ""],  # Priority order
)

recursive_chunks = recursive_splitter.split_documents(documents)
print(f"\n=== Recursive Character Splitting ===")
print(f"Input: {len(documents)} documents")
print(f"Output: {len(recursive_chunks)} chunks")
print(f"Chunk sizes: {[len(c.page_content) for c in recursive_chunks[:5]]}...")
print(f"\nSample chunk (first):")
print(f"  Content: {recursive_chunks[0].page_content[:150]}...")
print(f"  Metadata: {recursive_chunks[0].metadata}")

# --- Strategy 2: Fixed-Size Character Splitting ---
fixed_splitter = CharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50,
    separator=" ",  # Split on spaces (word boundary)
)

fixed_chunks = fixed_splitter.split_documents(documents)
print(f"\n=== Fixed-Size Character Splitting ===")
print(f"Output: {len(fixed_chunks)} chunks")

# --- Strategy 3: Token-Based Splitting ---
token_splitter = TokenTextSplitter(
    chunk_size=200,        # Max tokens per chunk
    chunk_overlap=20,      # Token overlap
    encoding_name="cl100k_base",  # GPT-4 tokenizer
)

token_chunks = token_splitter.split_documents(documents)
print(f"\n=== Token-Based Splitting ===")
print(f"Output: {len(token_chunks)} chunks")
print(f"(Token-based ensures precise token counts for LLM context budgeting)")

# --- Compare all strategies ---
print(f"\n=== Comparison ===")
print(f"{'Strategy':<30} {'Chunks':<10} {'Avg Size (chars)':<20}")
print(f"{'-'*60}")
for name, chunks in [
    ("Recursive (500 chars)", recursive_chunks),
    ("Fixed-size (400 chars)", fixed_chunks),
    ("Token-based (200 tokens)", token_chunks),
]:
    avg_size = sum(len(c.page_content) for c in chunks) // len(chunks)
    print(f"{name:<30} {len(chunks):<10} {avg_size:<20}")

# Expected output:
# Created 3 sample documents
#   - rag_overview.txt: 1189 chars
#   - vector_databases.txt: 1420 chars
#   - chunking_guide.txt: 1307 chars
#
# === Recursive Character Splitting ===
# Input: 3 documents
# Output: 10 chunks
# Chunk sizes: [497, 478, 489, 451, ...]...
#
# === Fixed-Size Character Splitting ===
# Output: 12 chunks
#
# === Token-Based Splitting ===
# Output: 15 chunks
#
# === Comparison ===
# Strategy                       Chunks     Avg Size (chars)
# ------------------------------------------------------------
# Recursive (500 chars)          10         ~410
# Fixed-size (400 chars)         12         ~350
# Token-based (200 tokens)       15         ~260
```

---

### 2.C3: Embed Chunks and Store in ChromaDB

```python
"""
Take chunked documents, generate embeddings using OpenAI, and store
them in a ChromaDB vector database. This covers the Embed → Index stages.
"""

import os
import chromadb
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

os.environ["OPENAI_API_KEY"] = "sk-..."  # Replace with your key

# ============================================================
# Step 1: Prepare documents and chunks (from previous exercise)
# ============================================================
documents = [
    Document(
        page_content="""Retrieval-Augmented Generation (RAG) enhances LLMs by providing 
relevant external knowledge at inference time. Instead of relying solely on parametric 
memory, RAG retrieves documents from a knowledge base and includes them in the prompt.
The key advantage is that knowledge can be updated without retraining. RAG was introduced
in 2020 by Lewis et al. from Facebook AI Research.""",
        metadata={"source": "rag_intro.txt", "section": "overview"}
    ),
    Document(
        page_content="""Vector databases store and query high-dimensional embeddings 
efficiently. They use approximate nearest neighbor algorithms like HNSW, IVF, and 
Product Quantization. Popular options include ChromaDB for prototyping, Pinecone for 
managed cloud, and Qdrant for self-hosted production deployments.""",
        metadata={"source": "vector_db.txt", "section": "infrastructure"}
    ),
    Document(
        page_content="""Chunking splits documents into smaller pieces for embedding and 
retrieval. The recursive character splitter is the most popular strategy, trying paragraph 
breaks first, then line breaks, then spaces. Chunk sizes of 200-1000 tokens with 10-20% 
overlap work well for most use cases.""",
        metadata={"source": "chunking.txt", "section": "preprocessing"}
    ),
    Document(
        page_content="""Embedding models convert text into dense vectors. OpenAI's 
text-embedding-3-small produces 1536-dimensional vectors. Sentence-Transformers like 
all-MiniLM-L6-v2 produce 384-dimensional vectors locally. The choice depends on your 
quality needs, latency budget, and data privacy requirements.""",
        metadata={"source": "embeddings.txt", "section": "models"}
    ),
    Document(
        page_content="""HNSW (Hierarchical Navigable Small World) is the most popular 
approximate nearest neighbor algorithm. It builds a multi-layer graph where upper layers 
have long-range connections and lower layers have fine-grained local connections. Search 
starts at the top layer and greedily navigates toward the query vector.""",
        metadata={"source": "hnsw.txt", "section": "algorithms"}
    ),
    Document(
        page_content="""Fine-tuning bakes knowledge into model weights permanently. This 
is expensive and inflexible — you must retrain whenever data changes. RAG is preferable 
when knowledge changes frequently (daily docs, support tickets, news). Fine-tuning is 
better for teaching the model a new style, format, or domain-specific reasoning pattern.""",
        metadata={"source": "rag_vs_finetuning.txt", "section": "comparison"}
    ),
]

# Chunk the documents
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
chunks = splitter.split_documents(documents)
print(f"Split {len(documents)} documents into {len(chunks)} chunks")

# ============================================================
# Step 2: Create embedding model
# ============================================================
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small",  # 1536 dimensions, good balance of cost/quality
)

# Demonstrate what an embedding looks like
sample_embedding = embedding_model.embed_query("What is RAG?")
print(f"\nEmbedding model: text-embedding-3-small")
print(f"Vector dimensions: {len(sample_embedding)}")
print(f"First 5 values: {sample_embedding[:5]}")

# ============================================================
# Step 3: Store chunks in ChromaDB via LangChain
# ============================================================

# Option A: In-memory (for development/testing)
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    collection_name="rag_study_guide",
    # persist_directory="./chroma_db"  # Uncomment to persist to disk
)

print(f"\n✓ Stored {len(chunks)} chunks in ChromaDB")
print(f"  Collection: 'rag_study_guide'")

# ============================================================
# Step 4: Verify the store works with a test query
# ============================================================
test_query = "How does HNSW work?"
results = vectorstore.similarity_search_with_score(test_query, k=3)

print(f"\nVerification query: '{test_query}'")
print(f"Top 3 results:")
for i, (doc, score) in enumerate(results, 1):
    print(f"\n  {i}. Score: {score:.4f}")
    print(f"     Source: {doc.metadata.get('source', 'unknown')}")
    print(f"     Content: {doc.page_content[:100]}...")

# ============================================================
# Step 5: Inspect the ChromaDB collection directly
# ============================================================
# Access the underlying ChromaDB collection for debugging
collection = vectorstore._collection
print(f"\n=== Collection Stats ===")
print(f"  Name: {collection.name}")
print(f"  Count: {collection.count()}")

# Peek at stored metadata
peek = collection.peek(limit=3)
print(f"  Sample metadata: {peek['metadatas'][:3]}")

# Expected output:
# Split 6 documents into ~8 chunks
#
# Embedding model: text-embedding-3-small
# Vector dimensions: 1536
# First 5 values: [0.023, -0.041, 0.012, ...]
#
# ✓ Stored 8 chunks in ChromaDB
#   Collection: 'rag_study_guide'
#
# Verification query: 'How does HNSW work?'
# Top 3 results:
#   1. Score: 0.3214 (lower = more similar in ChromaDB's L2 distance)
#      Source: hnsw.txt
#      Content: HNSW (Hierarchical Navigable Small World) is the most popular...
#   2. Score: 0.5891
#      Source: vector_db.txt
#      Content: Vector databases store and query high-dimensional embeddings...
#   3. Score: 0.7234
#      Source: embeddings.txt
#      Content: Embedding models convert text into dense vectors...
#
# === Collection Stats ===
#   Name: rag_study_guide
#   Count: 8
#   Sample metadata: [{'source': 'rag_intro.txt', 'section': 'overview'}, ...]
```

---

### 2.C4: Run Semantic Search Queries and Surface Top-K Results

```python
"""
Demonstrate semantic search: embed queries, find relevant chunks,
and explore similarity scoring, filtering, and result formatting.
"""

import os
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

os.environ["OPENAI_API_KEY"] = "sk-..."  # Replace with your key

# ============================================================
# Setup: Create and populate vectorstore (reusable from 2.C3)
# ============================================================
documents = [
    Document(page_content="RAG combines retrieval with generation. It fetches relevant documents from a knowledge base and passes them to an LLM as context for answer generation. This grounds responses in factual sources.", metadata={"source": "rag.txt", "topic": "RAG"}),
    Document(page_content="HNSW builds a multi-layer graph for fast approximate nearest neighbor search. It starts at the top layer with few long-range connections and descends to bottom layers with dense local connections.", metadata={"source": "hnsw.txt", "topic": "algorithms"}),
    Document(page_content="ChromaDB is an open-source embedding database designed for AI applications. It runs locally, requires no setup, and supports both persistent and in-memory storage modes.", metadata={"source": "chromadb.txt", "topic": "databases"}),
    Document(page_content="Text-embedding-3-small from OpenAI produces 1536-dimensional vectors. It offers a good balance between embedding quality and API cost. For higher quality, text-embedding-3-large provides 3072 dimensions.", metadata={"source": "embeddings.txt", "topic": "models"}),
    Document(page_content="Chunking strategies include fixed-size, sentence-based, semantic, and recursive splitting. Recursive character splitting is the most common default, respecting paragraph and sentence boundaries.", metadata={"source": "chunking.txt", "topic": "preprocessing"}),
    Document(page_content="Fine-tuning modifies model weights to encode new knowledge or behaviors. It's expensive, inflexible, and best for teaching style or reasoning patterns rather than factual knowledge that changes frequently.", metadata={"source": "finetuning.txt", "topic": "training"}),
    Document(page_content="Pinecone is a fully managed vector database service. It handles scaling, replication, and infrastructure automatically. Pricing is based on the number of vectors stored and queries per second.", metadata={"source": "pinecone.txt", "topic": "databases"}),
    Document(page_content="Product Quantization compresses vectors by splitting them into sub-vectors and replacing each with a codebook index. This achieves 32-64x memory reduction while maintaining reasonable search accuracy.", metadata={"source": "pq.txt", "topic": "algorithms"}),
]

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embedding_model,
    collection_name="semantic_search_demo",
)
print(f"✓ Vectorstore ready with {len(documents)} documents\n")

# ============================================================
# Search 1: Basic similarity search (top-k)
# ============================================================
print("=" * 60)
print("SEARCH 1: Basic top-k similarity search")
print("=" * 60)

query = "What vector databases are available?"
results = vectorstore.similarity_search(query, k=3)

print(f"Query: '{query}'")
print(f"Top {len(results)} results:\n")
for i, doc in enumerate(results, 1):
    print(f"  {i}. [{doc.metadata['source']}] {doc.page_content[:80]}...")
print()

# ============================================================
# Search 2: Similarity search WITH scores
# ============================================================
print("=" * 60)
print("SEARCH 2: Similarity search with distance scores")
print("=" * 60)

query = "How do I compress embeddings to save memory?"
results_with_scores = vectorstore.similarity_search_with_score(query, k=4)

print(f"Query: '{query}'")
print(f"Results (lower score = more similar in L2 distance):\n")
for i, (doc, score) in enumerate(results_with_scores, 1):
    relevance = "HIGH" if score < 0.5 else "MEDIUM" if score < 1.0 else "LOW"
    print(f"  {i}. Score: {score:.4f} [{relevance}]")
    print(f"     Source: {doc.metadata['source']}")
    print(f"     Content: {doc.page_content[:100]}...")
    print()

# ============================================================
# Search 3: Similarity search with relevance threshold
# ============================================================
print("=" * 60)
print("SEARCH 3: Filtered by relevance score threshold")
print("=" * 60)

query = "Explain the HNSW algorithm"
# similarity_search_with_relevance_scores returns normalized scores (0-1, higher=more similar)
results_relevant = vectorstore.similarity_search_with_relevance_scores(query, k=5, score_threshold=0.5)

print(f"Query: '{query}'")
print(f"Results above 0.5 relevance threshold:\n")
for i, (doc, score) in enumerate(results_relevant, 1):
    print(f"  {i}. Relevance: {score:.4f} | Source: {doc.metadata['source']}")
    print(f"     {doc.page_content[:80]}...")
print()

# ============================================================
# Search 4: Metadata filtering
# ============================================================
print("=" * 60)
print("SEARCH 4: Semantic search with metadata filter")
print("=" * 60)

query = "Tell me about search algorithms"
# Filter: only search within documents tagged with topic "algorithms"
results_filtered = vectorstore.similarity_search(
    query, 
    k=3, 
    filter={"topic": "algorithms"}
)

print(f"Query: '{query}' (filtered to topic='algorithms')")
print(f"Results:\n")
for i, doc in enumerate(results_filtered, 1):
    print(f"  {i}. [{doc.metadata['topic']}] {doc.page_content[:80]}...")
print()

# ============================================================
# Search 5: Multiple queries to show semantic understanding
# ============================================================
print("=" * 60)
print("SEARCH 5: Demonstrating semantic understanding")
print("=" * 60)

queries = [
    "How do I make LLMs use current information?",  # Should match RAG
    "What's the fastest way to find similar vectors?",  # Should match HNSW
    "How to reduce storage costs for embeddings?",  # Should match PQ
    "hosted service with automatic scaling",  # Should match Pinecone
]

for query in queries:
    top_result = vectorstore.similarity_search(query, k=1)[0]
    print(f"  Q: '{query}'")
    print(f"  A: [{top_result.metadata['source']}] {top_result.page_content[:60]}...")
    print()

# Expected output:
# ✓ Vectorstore ready with 8 documents
#
# SEARCH 1: Basic top-k similarity search
# Query: 'What vector databases are available?'
# Top 3 results:
#   1. [chromadb.txt] ChromaDB is an open-source embedding database designed for AI...
#   2. [pinecone.txt] Pinecone is a fully managed vector database service...
#   3. [hnsw.txt] HNSW builds a multi-layer graph for fast approximate...
#
# SEARCH 5: Demonstrating semantic understanding
#   Q: 'How do I make LLMs use current information?'
#   A: [rag.txt] RAG combines retrieval with generation...
#   Q: 'What's the fastest way to find similar vectors?'
#   A: [hnsw.txt] HNSW builds a multi-layer graph for fast...
#   Q: 'How to reduce storage costs for embeddings?'
#   A: [pq.txt] Product Quantization compresses vectors...
#   Q: 'hosted service with automatic scaling'
#   A: [pinecone.txt] Pinecone is a fully managed vector database...
```

---

### 2.C5: Wire Retrieval Output to an LLM for Final Answer Generation

```python
"""
Complete RAG pipeline: retrieve relevant chunks and pass them to an LLM
to generate a grounded answer. This is the full end-to-end system.
"""

import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

os.environ["OPENAI_API_KEY"] = "sk-..."  # Replace with your key

# ============================================================
# Step 1: Build the knowledge base (vectorstore)
# ============================================================
knowledge_base = [
    Document(page_content="Our company's refund policy allows returns within 30 days of purchase for physical products. Digital products can be refunded within 7 days if less than 10% of the content has been accessed. Shipping costs are non-refundable unless the return is due to a defect.", metadata={"source": "refund_policy.md"}),
    Document(page_content="Premium tier subscribers get priority support with 2-hour response time during business hours (9 AM - 6 PM EST). Basic tier has 24-hour response time. Enterprise tier gets dedicated support engineers and 30-minute response SLA.", metadata={"source": "support_tiers.md"}),
    Document(page_content="To reset your password: 1) Go to login page, 2) Click 'Forgot Password', 3) Enter your email address, 4) Check your inbox for a reset link (valid for 1 hour), 5) Create a new password with at least 12 characters, one uppercase, one number, and one special character.", metadata={"source": "password_reset.md"}),
    Document(page_content="Our API rate limits are: Free tier - 100 requests/minute, Basic - 1000 requests/minute, Pro - 10000 requests/minute, Enterprise - custom limits. Exceeding limits returns HTTP 429. Implement exponential backoff for retry logic.", metadata={"source": "api_limits.md"}),
    Document(page_content="Data export is available for all paid plans. Go to Settings > Data > Export. You can export in CSV, JSON, or Parquet formats. Exports for accounts with more than 1 million records are processed asynchronously and you'll receive an email when ready.", metadata={"source": "data_export.md"}),
    Document(page_content="Two-factor authentication (2FA) is required for all admin accounts. We support authenticator apps (TOTP), SMS, and hardware security keys. Backup codes are generated during setup — store them securely. Lost access? Contact support with your account ID and photo ID.", metadata={"source": "security_2fa.md"}),
]

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(
    documents=knowledge_base,
    embedding=embedding_model,
    collection_name="company_docs",
)
print(f"✓ Knowledge base loaded: {len(knowledge_base)} documents\n")

# ============================================================
# Step 2: Create a retriever
# ============================================================
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},  # Retrieve top 3 most relevant chunks
)

# ============================================================
# Step 3: Define the RAG prompt template
# ============================================================
rag_prompt = ChatPromptTemplate.from_template("""
You are a helpful customer support assistant. Answer the user's question 
based ONLY on the following context. If the context doesn't contain enough 
information to answer, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer: Provide a clear, concise answer based on the context above. 
Cite the source document when possible.
""")

# ============================================================
# Step 4: Create the LLM
# ============================================================
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,  # Deterministic for factual answers
)

# ============================================================
# Step 5: Build the RAG chain
# ============================================================
def format_docs(docs: list[Document]) -> str:
    """Format retrieved documents into a single context string."""
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        formatted.append(f"[Document {i} - {source}]\n{doc.page_content}")
    return "\n\n".join(formatted)

# LangChain Expression Language (LCEL) chain
rag_chain = (
    {
        "context": retriever | format_docs,  # Retrieve docs, then format them
        "question": RunnablePassthrough(),    # Pass question through unchanged
    }
    | rag_prompt     # Fill in the prompt template
    | llm            # Generate answer
    | StrOutputParser()  # Extract string from LLM response
)

# ============================================================
# Step 6: Ask questions!
# ============================================================
questions = [
    "Can I get a refund for a digital product I bought 5 days ago?",
    "What happens if I exceed the API rate limit on the Pro plan?",
    "How do I set up two-factor authentication?",
    "What's the response time for premium support?",
    "Can I export my data as a Parquet file?",
]

print("=" * 70)
print("RAG QUESTION-ANSWERING SYSTEM")
print("=" * 70)

for question in questions:
    print(f"\n{'─' * 70}")
    print(f"Q: {question}")
    
    # Show retrieved documents for transparency
    retrieved_docs = retriever.invoke(question)
    print(f"\n  📄 Retrieved {len(retrieved_docs)} documents:")
    for doc in retrieved_docs:
        print(f"     - {doc.metadata['source']}: {doc.page_content[:60]}...")
    
    # Generate answer
    answer = rag_chain.invoke(question)
    print(f"\n  💡 Answer: {answer}")

# ============================================================
# Step 7: Advanced — Manual RAG for full control
# ============================================================
print("\n\n" + "=" * 70)
print("MANUAL RAG (for full control over each step)")
print("=" * 70)

def manual_rag(question: str, vectorstore, llm, k: int = 3) -> dict:
    """
    Manually perform each RAG step for maximum visibility and control.
    Returns the full pipeline state for debugging.
    """
    # Step A: Embed the question and retrieve
    retrieved = vectorstore.similarity_search_with_score(question, k=k)
    
    # Step B: Format context
    context_parts = []
    sources = []
    for doc, score in retrieved:
        context_parts.append(f"[{doc.metadata['source']}]: {doc.page_content}")
        sources.append({"source": doc.metadata["source"], "score": float(score)})
    context = "\n\n".join(context_parts)
    
    # Step C: Build prompt
    prompt = f"""Answer the question based only on the provided context.
If the context doesn't contain the answer, say so.

Context:
{context}

Question: {question}

Answer:"""
    
    # Step D: Generate
    from langchain.schema import HumanMessage, SystemMessage
    response = llm.invoke([
        SystemMessage(content="You are a precise, helpful assistant. Only use the provided context."),
        HumanMessage(content=prompt),
    ])
    
    return {
        "question": question,
        "answer": response.content,
        "sources": sources,
        "num_chunks_retrieved": len(retrieved),
        "context_length": len(context),
    }

# Run manual RAG
result = manual_rag(
    "What are the password requirements for a new password?",
    vectorstore,
    llm,
)

print(f"\nQuestion: {result['question']}")
print(f"Answer: {result['answer']}")
print(f"\nDebug info:")
print(f"  Chunks retrieved: {result['num_chunks_retrieved']}")
print(f"  Context length: {result['context_length']} chars")
print(f"  Sources used:")
for src in result["sources"]:
    print(f"    - {src['source']} (distance: {src['score']:.4f})")

# Expected output:
# ✓ Knowledge base loaded: 6 documents
#
# ══════════════════════════════════════════════════════════════════
# RAG QUESTION-ANSWERING SYSTEM
# ══════════════════════════════════════════════════════════════════
#
# ──────────────────────────────────────────────────────────────────
# Q: Can I get a refund for a digital product I bought 5 days ago?
#
#   📄 Retrieved 3 documents:
#      - refund_policy.md: Our company's refund policy allows returns within 30 days...
#      - data_export.md: Data export is available for all paid plans...
#      - support_tiers.md: Premium tier subscribers get priority support...
#
#   💡 Answer: Yes, you can get a refund for a digital product purchased 5 days ago.
#      According to our refund policy (refund_policy.md), digital products can be
#      refunded within 7 days if less than 10% of the content has been accessed.
#      Since you purchased it 5 days ago, you're within the 7-day window.
#
# ──────────────────────────────────────────────────────────────────
# Q: What happens if I exceed the API rate limit on the Pro plan?
#
#   💡 Answer: On the Pro plan, the API rate limit is 10,000 requests per minute.
#      If you exceed this limit, the API will return an HTTP 429 status code.
#      You should implement exponential backoff for retry logic.
#      (Source: api_limits.md)
#
# ══════════════════════════════════════════════════════════════════
# MANUAL RAG (for full control over each step)
# ══════════════════════════════════════════════════════════════════
#
# Question: What are the password requirements for a new password?
# Answer: A new password must have at least 12 characters, one uppercase letter,
#         one number, and one special character.
# Debug info:
#   Chunks retrieved: 3
#   Context length: 542 chars
#   Sources used:
#     - password_reset.md (distance: 0.3821)
#     - security_2fa.md (distance: 0.7234)
#     - support_tiers.md (distance: 0.9102)
```

---

*Continue to Week 3...*


---

# WEEK 3: Advanced RAG

**Goal:** Push RAG accuracy beyond naive retrieval using query rewriting, reranking, and evaluation.

---

## Theory

### 3.1 Why Naive RAG Hits Accuracy Ceilings

Naive RAG (embed query → retrieve top-k → feed to LLM) fails in predictable ways:

**Vocabulary mismatch**: User asks "How do I fix a segfault?" but documents say "segmentation fault resolution." The embeddings may not be close enough despite identical meaning.

**Lost in the middle**: When you retrieve 10 chunks, the LLM pays more attention to the first and last ones, sometimes ignoring critical information in the middle.

**Insufficient context**: A single chunk may not contain enough information. The answer might span multiple non-adjacent paragraphs.

**Wrong granularity**: Your chunks might be too large (diluting relevance) or too small (missing context).

**Multi-hop reasoning**: "Who founded the company that acquired Twitter?" requires two retrieval steps — one to find the acquirer, another to find its founder.

The solution is a multi-stage retrieval pipeline: rewrite the query → retrieve broadly → rerank precisely → synthesize.

---

### 3.2 Query Rewriting Techniques

#### Query Expansion
Add synonyms and related terms to the original query:
- Original: "Python memory leak"
- Expanded: "Python memory leak garbage collection memory management heap allocation"

#### HyDE (Hypothetical Document Embeddings)
Instead of embedding the *question*, generate a *hypothetical answer* and embed that. The intuition: a hypothetical answer looks more like the actual document than the question does.

Process:
1. User asks: "What causes buffer overflows?"
2. LLM generates hypothetical answer: "Buffer overflows occur when a program writes data beyond the allocated memory buffer..."
3. Embed this hypothetical answer
4. Search with that embedding (it's closer to the actual document text)

This consistently improves retrieval by 10-20% on benchmarks.

#### Multi-Query
Generate 3-5 variations of the original question, retrieve for each, and merge results:
- "What causes buffer overflows?" → Also search: "buffer overflow prevention techniques", "memory safety vulnerabilities in C", "stack smashing explained"

---

### 3.3 Cross-Encoder Reranking

**Bi-encoders** (used in initial retrieval) encode query and document *separately*, then compare with cosine similarity. Fast but less accurate.

**Cross-encoders** take query AND document as a *single input* and output a relevance score. Much more accurate but 100x slower.

The solution: two-stage pipeline:
1. **Stage 1**: Bi-encoder retrieves top-50 candidates (fast, ~5ms)
2. **Stage 2**: Cross-encoder reranks those 50 to find the best 5 (slow but accurate, ~200ms)

Popular cross-encoders: `cross-encoder/ms-marco-MiniLM-L-6-v2`, Cohere Rerank, Jina Reranker.

Typical improvement: 15-30% better precision at top-5.

---

### 3.4 Hybrid Search (Dense + Sparse / BM25)

**Dense retrieval** (embeddings): Great at semantic matching ("car" matches "automobile") but can miss exact keyword matches.

**Sparse retrieval (BM25)**: Traditional keyword search using term frequency. Great at exact matches ("error code 0x80070005") but misses semantic similarity.

**Hybrid** combines both:
1. Run dense search → get top-50 with scores
2. Run BM25 search → get top-50 with scores
3. Merge using **Reciprocal Rank Fusion (RRF)**:
   - `RRF_score = Σ 1/(k + rank_i)` where k=60 (constant)
   - Documents appearing in both lists get boosted

This consistently outperforms either method alone, especially for technical/domain-specific queries.

---

### 3.5 RAG-Specific Evals: Faithfulness and Answer Relevance

**RAGAS** (Retrieval Augmented Generation Assessment) defines key metrics:

| Metric | What it measures | How |
|--------|-----------------|-----|
| **Faithfulness** | Is the answer grounded in retrieved context? | Check each claim against source docs |
| **Answer Relevance** | Does the answer address the question? | Generate questions from the answer, compare to original |
| **Context Precision** | Are retrieved docs relevant? | Check if ground-truth appears in top-k |
| **Context Recall** | Did we find all relevant docs? | Compare retrieved set to full relevant set |

**How to evaluate**:
1. Create 50-100 ground-truth QA pairs (question + expected answer + source docs)
2. Run your RAG pipeline on each question
3. Score with RAGAS or custom metrics
4. Track scores as you make changes

---

## Coding Practice

### 3.C1: Implement HyDE Query Rewriting

```python
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions

client = OpenAI()

def generate_hypothetical_answer(question: str) -> str:
    """Generate a hypothetical answer to use as search query (HyDE)."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Write a brief, factual paragraph that would answer this question. Do not say 'I don't know'. Write as if you are a textbook."},
            {"role": "user", "content": question},
        ],
        temperature=0.0,
        max_tokens=150,
    )
    return response.choices[0].message.content

def hyde_search(question: str, collection, n_results: int = 5):
    """Search using HyDE: embed hypothetical answer instead of question."""
    # Step 1: Generate hypothetical answer
    hyp_answer = generate_hypothetical_answer(question)
    print(f"  HyDE hypothesis: {hyp_answer[:100]}...")
    
    # Step 2: Search with the hypothetical answer as query
    results = collection.query(query_texts=[hyp_answer], n_results=n_results)
    return results

def naive_search(question: str, collection, n_results: int = 5):
    """Standard naive search: embed the question directly."""
    results = collection.query(query_texts=[question], n_results=n_results)
    return results

# Compare naive vs HyDE
question = "What are the memory management techniques in modern operating systems?"

print("=== Naive Search ===")
naive_results = naive_search(question, collection)  # assumes collection exists
for doc, score in zip(naive_results["documents"][0], naive_results["distances"][0]):
    print(f"  [{score:.3f}] {doc[:80]}...")

print("\n=== HyDE Search ===")
hyde_results = hyde_search(question, collection)
for doc, score in zip(hyde_results["documents"][0], hyde_results["distances"][0]):
    print(f"  [{score:.3f}] {doc[:80]}...")
```

---

### 3.C2: Cross-Encoder Reranker

```python
# pip install sentence-transformers

from sentence_transformers import CrossEncoder
import numpy as np

# Load a cross-encoder model (downloads ~80MB first time)
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank_results(query: str, documents: list[str], top_k: int = 5) -> list[tuple[str, float]]:
    """Rerank documents using cross-encoder for better precision."""
    # Create pairs of (query, document) for the cross-encoder
    pairs = [[query, doc] for doc in documents]
    
    # Score all pairs (this is the expensive step)
    scores = reranker.predict(pairs)
    
    # Sort by score descending
    scored_docs = list(zip(documents, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    return scored_docs[:top_k]

# Example: initial retrieval gives 20 candidates, rerank to top 5
query = "How does garbage collection work in Python?"
candidates = [
    "Python uses reference counting and a cyclic garbage collector...",
    "Java's garbage collection uses mark-and-sweep algorithm...",
    "Memory management in C requires manual allocation with malloc...",
    "The Python GC module allows you to tune collection thresholds...",
    "Rust eliminates garbage collection through ownership semantics...",
    "CPython's reference counting increments/decrements on assignment...",
    # ... imagine 20 candidates from initial retrieval
]

reranked = rerank_results(query, candidates, top_k=3)
print("Top 3 after reranking:")
for doc, score in reranked:
    print(f"  [{score:.4f}] {doc[:80]}...")

# Expected: Python GC docs rank highest, Java/Rust/C rank lower
```

---

### 3.C3: BM25 Hybrid Search with Reciprocal Rank Fusion

```python
# pip install rank-bm25 chromadb

from rank_bm25 import BM25Okapi
import numpy as np

def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Merge multiple ranked lists using RRF."""
    scores: dict[str, float] = {}
    
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            if doc_id not in scores:
                scores[doc_id] = 0.0
            scores[doc_id] += 1.0 / (k + rank + 1)
    
    # Sort by fused score
    sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results

def hybrid_search(query: str, documents: list[str], collection, n_results: int = 5):
    """Combine dense (embedding) and sparse (BM25) retrieval."""
    
    # --- Dense retrieval ---
    dense_results = collection.query(query_texts=[query], n_results=20)
    dense_ranking = dense_results["ids"][0]  # ranked doc IDs
    
    # --- Sparse retrieval (BM25) ---
    tokenized_docs = [doc.lower().split() for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)
    bm25_scores = bm25.get_scores(query.lower().split())
    bm25_ranking = np.argsort(bm25_scores)[::-1][:20].tolist()
    bm25_doc_ids = [str(i) for i in bm25_ranking]
    
    # --- Fuse with RRF ---
    fused = reciprocal_rank_fusion([dense_ranking, bm25_doc_ids])
    
    return fused[:n_results]

# Example usage
query = "error code 0x80070005 access denied"
results = hybrid_search(query, documents, collection)
print("Hybrid results (RRF):")
for doc_id, score in results:
    print(f"  ID: {doc_id}, RRF Score: {score:.4f}")
```

---

### 3.C4: RAG Evaluation Suite

```python
# pip install ragas datasets

from dataclasses import dataclass

@dataclass
class EvalCase:
    question: str
    expected_answer: str
    source_doc_ids: list[str]

def evaluate_faithfulness(answer: str, context: str, llm_client) -> float:
    """Check if the answer is grounded in the retrieved context."""
    response = llm_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You evaluate whether an answer is fully supported by the given context. Return a score from 0.0 to 1.0."},
            {"role": "user", "content": f"Context: {context}\n\nAnswer: {answer}\n\nScore (0.0-1.0):"},
        ],
        temperature=0.0,
        max_tokens=10,
    )
    try:
        return float(response.choices[0].message.content.strip())
    except ValueError:
        return 0.0

def evaluate_relevance(question: str, answer: str, llm_client) -> float:
    """Check if the answer actually addresses the question."""
    response = llm_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Rate how well this answer addresses the question. Return 0.0-1.0."},
            {"role": "user", "content": f"Question: {question}\n\nAnswer: {answer}\n\nScore (0.0-1.0):"},
        ],
        temperature=0.0,
        max_tokens=10,
    )
    try:
        return float(response.choices[0].message.content.strip())
    except ValueError:
        return 0.0

def run_eval(eval_cases: list[EvalCase], rag_pipeline, llm_client) -> dict:
    """Run full evaluation across all test cases."""
    faithfulness_scores = []
    relevance_scores = []
    retrieval_hits = []
    
    for case in eval_cases:
        # Run the RAG pipeline
        answer, retrieved_docs, retrieved_ids = rag_pipeline(case.question)
        
        # Evaluate
        context = "\n".join(retrieved_docs)
        faith = evaluate_faithfulness(answer, context, llm_client)
        relevance = evaluate_relevance(case.question, answer, llm_client)
        hit = any(doc_id in retrieved_ids for doc_id in case.source_doc_ids)
        
        faithfulness_scores.append(faith)
        relevance_scores.append(relevance)
        retrieval_hits.append(hit)
    
    return {
        "avg_faithfulness": sum(faithfulness_scores) / len(faithfulness_scores),
        "avg_relevance": sum(relevance_scores) / len(relevance_scores),
        "retrieval_precision": sum(retrieval_hits) / len(retrieval_hits),
        "num_cases": len(eval_cases),
    }

# Usage
eval_cases = [
    EvalCase("What is Python's GIL?", "The Global Interpreter Lock...", ["doc_3", "doc_7"]),
    EvalCase("How does async work?", "Async uses an event loop...", ["doc_12"]),
    # Add 50-100 cases for meaningful results
]

results = run_eval(eval_cases, my_rag_pipeline, client)
print(f"Faithfulness: {results['avg_faithfulness']:.2f}")
print(f"Relevance: {results['avg_relevance']:.2f}")
print(f"Retrieval Precision: {results['retrieval_precision']:.2f}")
```

---

### 3.C5: Benchmark Accuracy Deltas

```python
import json
from datetime import datetime

def benchmark_pipeline(name: str, pipeline_fn, eval_cases: list[EvalCase], client) -> dict:
    """Run eval and record results for comparison."""
    results = run_eval(eval_cases, pipeline_fn, client)
    results["pipeline_name"] = name
    results["timestamp"] = datetime.now().isoformat()
    return results

# Run benchmarks for each improvement
benchmarks = []

# Baseline: naive RAG
benchmarks.append(benchmark_pipeline("naive_rag", naive_rag_pipeline, eval_cases, client))

# + HyDE
benchmarks.append(benchmark_pipeline("naive + hyde", hyde_rag_pipeline, eval_cases, client))

# + Reranking
benchmarks.append(benchmark_pipeline("naive + hyde + rerank", rerank_rag_pipeline, eval_cases, client))

# + Hybrid search
benchmarks.append(benchmark_pipeline("full_hybrid", hybrid_rag_pipeline, eval_cases, client))

# Print comparison table
print(f"{'Pipeline':<30} {'Faith':>8} {'Relevance':>10} {'Retrieval':>10}")
print("-" * 60)
for b in benchmarks:
    print(f"{b['pipeline_name']:<30} {b['avg_faithfulness']:>8.3f} {b['avg_relevance']:>10.3f} {b['retrieval_precision']:>10.3f}")

# Save results
with open("benchmark_results.json", "w") as f:
    json.dump(benchmarks, f, indent=2)
```

---

*Continue to Week 4...*


---

# WEEK 4: RAG Architectures & Specialised Types

**Goal:** Know when to move beyond vanilla RAG — and how to implement Graph and Agentic RAG.

---

## Theory

### 4.1 Common Pitfalls in Production RAG Systems

Production RAG systems fail in ways that aren't visible in demos:

**1. Chunking-Query Mismatch**: Your chunks are 500 tokens but the answer spans 2000 tokens across multiple sections. The user gets a partial, confusing answer.

**2. Embedding Drift**: Your embedding model was trained on general text but your domain uses specialized terminology ("CRISPR-Cas9 guide RNA efficacy" won't embed well with a model trained on Wikipedia).

**3. Stale Data**: Documents change but your vector index doesn't. Users get outdated answers with no indication of staleness.

**4. Lost Metadata**: You embedded the text but lost the source URL, page number, and date. Now you can't provide citations or freshness signals.

**5. Retrieval Failures Are Silent**: The system always returns *something* — even if it's irrelevant. Users don't know the retrieval failed; they just see a confidently wrong answer.

**6. Context Window Stuffing**: Retrieving too many chunks dilutes the signal. The LLM gets confused by contradictory information from different time periods or contexts.

**Mitigations**:
- Add retrieval confidence scores and refuse to answer below a threshold
- Use metadata filtering (date, source, section) before semantic search
- Implement incremental indexing with TTLs for freshness
- Monitor retrieval quality with production evals

---

### 4.2 GraphRAG: Knowledge Graphs as Retrieval Backends

Traditional RAG treats documents as flat text chunks. GraphRAG structures information as **entities** (nodes) and **relationships** (edges), enabling multi-hop reasoning.

**What is a knowledge graph?**
A knowledge graph is a network of entities and their relationships:
- Node: "Python" (Programming Language)
- Node: "Guido van Rossum" (Person)
- Edge: "created_by" connecting Python → Guido

**Why it helps RAG**:
Consider the question: "What languages were created by people who studied at the University of Amsterdam?"

Vector search would struggle — no single chunk contains this complete answer. But a graph traversal can:
1. Find people connected to "University of Amsterdam" via "studied_at"
2. Find programming languages connected to those people via "created_by"
3. Return: Python (Guido van Rossum studied mathematics at UvA)

**GraphRAG Pipeline**:
1. **Extract**: Use an LLM to identify entities and relationships from documents
2. **Store**: Insert into a graph database (Neo4j, Amazon Neptune)
3. **Query**: Convert user questions to graph queries (Cypher/SPARQL) or use LLM to traverse
4. **Generate**: Feed graph results + original text to LLM for final answer

**When to use GraphRAG**: Multi-hop questions, relationship-heavy data (org charts, research papers, legal documents), when you need explainable retrieval paths.

---

### 4.3 KAG (Knowledge-Augmented Generation)

KAG combines **structured knowledge bases** (tables, APIs, SQL databases) with LLMs, going beyond pure text retrieval.

**Difference from RAG**:
- RAG: "Search text chunks, feed to LLM"
- KAG: "Query structured data (SQL, APIs, knowledge bases), combine with LLM reasoning"

**Example**: "What's the average salary for senior engineers in Seattle?"
- RAG approach: Search for documents mentioning salaries (unreliable, outdated)
- KAG approach: Query a structured HR database with SQL, return precise numbers

**Architecture**:
1. **Schema understanding**: LLM learns the database schema
2. **Query generation**: LLM generates SQL/API calls from natural language
3. **Execution**: Run the query against structured sources
4. **Synthesis**: LLM formats results into a natural language answer

**When to use**: Data with clear schema (databases, spreadsheets, APIs), numerical/analytical questions, when precision matters more than fluency.

---

### 4.4 Agentic RAG: LLM Decides What to Retrieve

In vanilla RAG, retrieval is automatic and fixed: every query triggers the same search pipeline. In **Agentic RAG**, the LLM itself decides:
- **Whether** to retrieve (maybe it already knows the answer)
- **What** query to use (may rewrite or decompose the question)
- **When** to stop (are results sufficient, or should it search again?)
- **How** to combine results from multiple sources

**The Agentic RAG Loop**:
```
1. Receive question
2. LLM thinks: "Do I need external information?"
3. If yes → LLM formulates a search query
4. Execute search → return results
5. LLM evaluates: "Is this sufficient to answer?"
6. If no → reformulate query and go to step 4
7. If yes → generate final answer with citations
```

**Advantages**:
- Handles questions that need zero, one, or multiple retrievals
- Can decompose complex questions into sub-queries
- Self-corrects when initial retrieval is insufficient

**Disadvantages**:
- Higher latency (multiple LLM calls)
- Higher cost
- Harder to debug (non-deterministic paths)

---

### 4.5 Decision Framework: Choosing the Right RAG Architecture

| Question Characteristic | Best Architecture |
|------------------------|-------------------|
| Simple factual lookup | Vanilla RAG |
| Multi-hop reasoning | GraphRAG |
| Numerical/analytical | KAG (structured) |
| Variable complexity | Agentic RAG |
| High latency tolerance | Agentic RAG |
| Relationship-heavy domain | GraphRAG |
| Need for citations/explainability | GraphRAG |
| Structured data sources | KAG |

**Decision tree**:
1. Is the data structured (SQL/API)? → KAG
2. Do questions require multi-hop reasoning? → GraphRAG
3. Is question complexity variable/unpredictable? → Agentic RAG
4. Simple factual Q&A over text? → Vanilla RAG (with Advanced techniques from Week 3)

---

## Coding Practice

### 4.C1: Set Up Neo4j

```python
# Option 1: Docker (recommended)
# docker run -d --name neo4j -p 7474:7474 -p 7687:7687 \
#   -e NEO4J_AUTH=neo4j/password123 neo4j:latest

# pip install neo4j

from neo4j import GraphDatabase

URI = "bolt://localhost:7687"
AUTH = ("neo4j", "password123")

def verify_connection():
    """Verify Neo4j connection is working."""
    driver = GraphDatabase.driver(URI, auth=AUTH)
    with driver.session() as session:
        result = session.run("RETURN 'Connected!' AS message")
        record = result.single()
        print(record["message"])
    driver.close()

verify_connection()
# Output: Connected!

# Create a sample graph
def setup_sample_data(driver):
    with driver.session() as session:
        # Clear existing data
        session.run("MATCH (n) DETACH DELETE n")
        
        # Create entities and relationships
        session.run("""
            CREATE (python:Language {name: 'Python', year: 1991})
            CREATE (guido:Person {name: 'Guido van Rossum'})
            CREATE (uva:University {name: 'University of Amsterdam'})
            CREATE (google:Company {name: 'Google'})
            CREATE (guido)-[:CREATED]->(python)
            CREATE (guido)-[:STUDIED_AT]->(uva)
            CREATE (guido)-[:WORKED_AT]->(google)
            CREATE (python)-[:USED_BY]->(google)
        """)
        print("Sample data created!")

driver = GraphDatabase.driver(URI, auth=AUTH)
setup_sample_data(driver)
driver.close()
```

---

### 4.C2: Build a GraphRAG System — Extract Entities & Store

```python
from openai import OpenAI
from neo4j import GraphDatabase
import json

client = OpenAI()
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password123"))

def extract_entities_and_relations(text: str) -> dict:
    """Use LLM to extract entities and relationships from text."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": """Extract entities and relationships from the text.
Return JSON with this exact format:
{
  "entities": [{"name": "...", "type": "Person|Organization|Technology|Concept"}],
  "relationships": [{"source": "...", "target": "...", "relation": "..."}]
}"""},
            {"role": "user", "content": text},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)

def store_in_neo4j(extraction: dict):
    """Store extracted entities and relations in Neo4j."""
    with driver.session() as session:
        # Create entities
        for entity in extraction["entities"]:
            session.run(
                "MERGE (e:Entity {name: $name}) SET e.type = $type",
                name=entity["name"], type=entity["type"]
            )
        
        # Create relationships
        for rel in extraction["relationships"]:
            session.run(
                """MATCH (a:Entity {name: $source}), (b:Entity {name: $target})
                   MERGE (a)-[r:RELATES {type: $relation}]->(b)""",
                source=rel["source"], target=rel["target"], relation=rel["relation"]
            )

# Process a document
text = """
Linus Torvalds created the Linux kernel in 1991 while studying at the University of Helsinki.
Linux is the foundation of Android, which is developed by Google. Torvalds also created Git,
which is used by GitHub, a company acquired by Microsoft in 2018.
"""

extraction = extract_entities_and_relations(text)
print("Extracted:", json.dumps(extraction, indent=2))
store_in_neo4j(extraction)
print("Stored in Neo4j!")
```

---

### 4.C3: Graph Traversal Queries + LLM Generation

```python
def graph_query(question: str) -> str:
    """Convert natural language to Cypher query, execute, and generate answer."""
    
    # Step 1: Generate Cypher query from natural language
    cypher_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": """Convert the question to a Neo4j Cypher query.
The graph has Entity nodes with 'name' and 'type' properties.
Relationships have a 'type' property describing the relation.
Return ONLY the Cypher query, nothing else."""},
            {"role": "user", "content": question},
        ],
        temperature=0.0,
    )
    cypher = cypher_response.choices[0].message.content.strip()
    print(f"Generated Cypher: {cypher}")
    
    # Step 2: Execute query
    with driver.session() as session:
        try:
            result = session.run(cypher)
            records = [dict(record) for record in result]
        except Exception as e:
            records = [{"error": str(e)}]
    
    print(f"Query results: {records}")
    
    # Step 3: Generate natural language answer
    answer_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Answer the question using the provided graph query results. Be concise."},
            {"role": "user", "content": f"Question: {question}\nGraph results: {json.dumps(records)}"},
        ],
        temperature=0.0,
    )
    return answer_response.choices[0].message.content

# Multi-hop question that vector search would struggle with
answer = graph_query("What did the creator of Linux also create?")
print(f"\nAnswer: {answer}")
# Expected: "Linus Torvalds, who created Linux, also created Git."
```

---

### 4.C4: Compare GraphRAG vs Vanilla RAG

```python
# Multi-hop questions that test graph traversal vs vector search
test_questions = [
    "What company acquired the platform that uses the tool created by the creator of Linux?",
    "Which university did the creator of Python attend?",
    "What technologies are used by the company that acquired GitHub?",
]

results_comparison = []

for q in test_questions:
    # Vanilla RAG
    vanilla_answer = vanilla_rag_pipeline(q)  # from Week 2
    
    # GraphRAG
    graph_answer = graph_query(q)
    
    results_comparison.append({
        "question": q,
        "vanilla_rag": vanilla_answer,
        "graph_rag": graph_answer,
    })
    print(f"\nQ: {q}")
    print(f"  Vanilla: {vanilla_answer}")
    print(f"  Graph:   {graph_answer}")

# GraphRAG should significantly outperform on multi-hop questions
```

---

### 4.C5: Basic Agentic Retrieval Loop

```python
from openai import OpenAI
import json

client = OpenAI()

def agentic_rag(question: str, max_iterations: int = 3) -> str:
    """LLM decides whether to search, what to search for, and when to stop."""
    
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_documents",
                "description": "Search the knowledge base for relevant information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The search query"},
                    },
                    "required": ["query"],
                },
            },
        }
    ]
    
    messages = [
        {"role": "system", "content": """You are a research assistant. 
You can search for information using the search_documents tool.
- Only search if you need external information
- You can search multiple times with different queries
- When you have enough information, provide your final answer"""},
        {"role": "user", "content": question},
    ]
    
    for iteration in range(max_iterations):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        
        msg = response.choices[0].message
        messages.append(msg)
        
        # If no tool call, the agent is done
        if not msg.tool_calls:
            print(f"  [Iteration {iteration+1}] Agent decided: answer directly")
            return msg.content
        
        # Execute tool calls
        for tool_call in msg.tool_calls:
            args = json.loads(tool_call.function.arguments)
            print(f"  [Iteration {iteration+1}] Searching: '{args['query']}'")
            
            # Execute the search (use your RAG retrieval here)
            search_results = search_documents(args["query"])  # your retrieval function
            
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(search_results),
            })
    
    # If we hit max iterations, force a final answer
    messages.append({"role": "user", "content": "Please provide your best answer with the information gathered."})
    final = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
    return final.choices[0].message.content

# Test with questions of varying complexity
print("=== Simple question (might not need search) ===")
print(agentic_rag("What is 2 + 2?"))

print("\n=== Complex question (needs search) ===")
print(agentic_rag("What are the key differences between HNSW and IVF indexing?"))

print("\n=== Multi-step question (needs multiple searches) ===")
print(agentic_rag("Compare Python's GIL with Java's threading model and explain which is better for CPU-bound tasks"))
```

---

*Continue to Week 5...*


---

# WEEK 5: Single-Agent Systems

**Goal:** Build a reliable single agent that reasons, uses tools, and returns structured outputs.

---

## Theory

### 5.1 What Is an Agent? LLM + Tools + Loop

An agent is an LLM that can take **actions** in the real world, not just generate text. The formula:

**Agent = LLM + Tools + Reasoning Loop**

- **LLM**: The "brain" that decides what to do next
- **Tools**: Functions the LLM can call (web search, calculator, database query, file I/O)
- **Loop**: The LLM observes results and decides whether to take another action or return a final answer

Unlike a simple prompt→response flow, an agent runs in a loop:
1. Think about what to do
2. Choose and execute a tool
3. Observe the result
4. Decide: done, or need another action?

**Analogy**: A regular LLM is like asking someone a question and they answer from memory. An agent is like giving someone a computer, phone, and notepad — they can look things up, make calculations, and take notes before answering.

---

### 5.2 LLM vs Agent: Pipeline vs Single API Call

| | Single LLM Call | Agent |
|--|--|--|
| Actions | 0 (just generates text) | Multiple (calls tools) |
| Reasoning | One-shot | Iterative |
| Determinism | High | Lower (different paths each run) |
| Latency | Fast (~1-3s) | Slow (~5-30s) |
| Cost | Low (1 API call) | High (3-10+ API calls) |
| Debugging | Easy | Hard (trace needed) |

**When to use a simple LLM call**: Summarization, translation, classification, code generation from spec.

**When to use an agent**: Research tasks, multi-step problem solving, tasks requiring external data, anything where the LLM needs to "figure out" the steps.

---

### 5.3 Tool / Function Calling

Function calling is the mechanism by which an LLM invokes external code. The LLM doesn't actually *run* the function — it outputs a structured JSON describing which function to call and with what arguments. Your code then executes it.

**Flow**:
1. You define available tools (name, description, parameter schema)
2. User sends a message
3. LLM decides to call a tool → outputs `{"name": "search", "arguments": {"query": "..."}}`
4. Your code executes `search(query="...")` and gets results
5. You feed results back to the LLM
6. LLM generates final answer (or calls another tool)

**Key insight**: The LLM never runs code. It only *decides* what to call. Your application is the executor.

---

### 5.4 The ReAct Pattern: Reasoning → Acting → Observing

ReAct (from the 2022 paper) interleaves **reasoning** (chain-of-thought) with **acting** (tool use):

```
Thought: I need to find the population of France.
Action: search("population of France 2024")
Observation: France has approximately 68.4 million people.
Thought: Now I have the answer.
Action: finish("France has approximately 68.4 million people as of 2024.")
```

Why this works better than acting alone: The reasoning step helps the LLM plan before acting, reducing irrelevant tool calls. The observation step grounds future reasoning in real data.

---

### 5.5 Pydantic AI: Type-Safe Agents

Pydantic AI enforces structured, validated outputs from agents. Instead of hoping the LLM returns valid JSON, you define Pydantic models and the framework guarantees the output matches.

Benefits:
- **Type safety**: Outputs are validated Python objects, not raw strings
- **Retry on failure**: If output doesn't match schema, automatically retry
- **IDE support**: Full autocomplete on agent outputs
- **Documentation**: Schema serves as documentation for what the agent returns

---

### 5.6 Basic Guardrails

Guardrails prevent agents from going off the rails:

- **Max iterations**: Cap the reasoning loop (e.g., max 5 tool calls) to prevent infinite loops
- **Input validation**: Check user input for injection attacks or out-of-scope requests
- **Output validation**: Verify the final output matches expected schema
- **Tool restrictions**: Limit which tools are available based on context
- **Cost caps**: Track token usage and stop if budget exceeded
- **Timeout**: Kill the agent after N seconds

---

## Coding Practice

### 5.C1: Build a ReAct Agent with 3+ Tools

```python
from openai import OpenAI
import json
import math
import requests

client = OpenAI()

# Define tools
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        result = eval(expression, {"__builtins__": {}}, {"math": math})
        return str(result)
    except Exception as e:
        return f"Error: {e}"

def web_search(query: str) -> str:
    """Simulate web search (replace with real API in production)."""
    # In production, use Tavily, Serper, or SerpAPI
    fake_results = {
        "population of france": "France has approximately 68.4 million people (2024).",
        "python release date": "Python was first released on February 20, 1991.",
    }
    for key, value in fake_results.items():
        if key in query.lower():
            return value
    return f"No results found for: {query}"

def read_file(filepath: str) -> str:
    """Read contents of a local file."""
    try:
        with open(filepath, "r") as f:
            return f.read()[:2000]  # Limit to 2000 chars
    except FileNotFoundError:
        return f"File not found: {filepath}"

# Map tool names to functions
TOOL_MAP = {
    "calculator": calculator,
    "web_search": web_search,
    "read_file": read_file,
}

# Tool definitions for OpenAI
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate mathematical expressions. Use Python math syntax.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string", "description": "Math expression, e.g. '2**10' or 'math.sqrt(144)'"}},
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Search query"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a local file.",
            "parameters": {
                "type": "object",
                "properties": {"filepath": {"type": "string", "description": "Path to the file"}},
                "required": ["filepath"],
            },
        },
    },
]

def run_agent(user_message: str, max_iterations: int = 5) -> str:
    """Run a ReAct agent with tool calling."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use tools when needed. Think step by step."},
        {"role": "user", "content": user_message},
    ]
    
    for i in range(max_iterations):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )
        
        msg = response.choices[0].message
        messages.append(msg)
        
        # No tool calls = agent is done
        if not msg.tool_calls:
            print(f"  [Step {i+1}] Final answer")
            return msg.content
        
        # Execute each tool call
        for tool_call in msg.tool_calls:
            fn_name = tool_call.function.name
            fn_args = json.loads(tool_call.function.arguments)
            
            print(f"  [Step {i+1}] Calling {fn_name}({fn_args})")
            
            # Execute the tool
            result = TOOL_MAP[fn_name](**fn_args)
            print(f"  [Step {i+1}] Result: {result[:100]}")
            
            # Feed result back
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })
    
    return "Agent reached max iterations without a final answer."

# Test the agent
print(run_agent("What is 2^10 + the square root of 144?"))
print("\n---\n")
print(run_agent("What year was Python released and how old is it now?"))
```

---

### 5.C2: Pydantic Models for Structured Outputs

```python
from pydantic import BaseModel, Field
from openai import OpenAI
import json

client = OpenAI()

class ResearchResult(BaseModel):
    """Structured output for a research query."""
    summary: str = Field(description="2-3 sentence summary of findings")
    key_facts: list[str] = Field(description="List of key facts discovered")
    sources: list[str] = Field(description="Sources used")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in answer (0-1)")

def research_with_structured_output(question: str) -> ResearchResult:
    """Get structured research results validated by Pydantic."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"""Research the question and return JSON matching this schema:
{ResearchResult.model_json_schema()}"""},
            {"role": "user", "content": question},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    
    raw = json.loads(response.choices[0].message.content)
    result = ResearchResult(**raw)  # Validates against schema
    return result

# Use it
result = research_with_structured_output("What are the benefits of Python type hints?")
print(f"Summary: {result.summary}")
print(f"Key facts: {result.key_facts}")
print(f"Confidence: {result.confidence}")
# Fully typed and validated!
```

---

### 5.C3: Guardrails Implementation

```python
import time
from functools import wraps

class AgentGuardrails:
    """Guardrails for agent execution."""
    
    def __init__(self, max_iterations=5, max_tokens_budget=10000, timeout_seconds=30):
        self.max_iterations = max_iterations
        self.max_tokens_budget = max_tokens_budget
        self.timeout_seconds = timeout_seconds
        self.tokens_used = 0
        self.start_time = None
    
    def check_iteration_limit(self, current_iteration: int):
        if current_iteration >= self.max_iterations:
            raise RuntimeError(f"Agent exceeded max iterations ({self.max_iterations})")
    
    def check_budget(self, tokens: int):
        self.tokens_used += tokens
        if self.tokens_used > self.max_tokens_budget:
            raise RuntimeError(f"Token budget exceeded: {self.tokens_used}/{self.max_tokens_budget}")
    
    def check_timeout(self):
        if self.start_time and (time.time() - self.start_time) > self.timeout_seconds:
            raise RuntimeError(f"Agent timed out after {self.timeout_seconds}s")
    
    def validate_input(self, user_input: str) -> str:
        """Basic input sanitization."""
        blocked_patterns = ["ignore previous instructions", "system prompt", "jailbreak"]
        for pattern in blocked_patterns:
            if pattern.lower() in user_input.lower():
                return "I cannot process that request."
        if len(user_input) > 5000:
            return "Input too long. Please shorten your request."
        return user_input
    
    def validate_output(self, output: str) -> str:
        """Basic output validation."""
        if not output or len(output.strip()) == 0:
            return "I was unable to generate a response."
        if len(output) > 10000:
            return output[:10000] + "\n[Response truncated]"
        return output

# Usage in agent loop
guardrails = AgentGuardrails(max_iterations=5, timeout_seconds=30)
guardrails.start_time = time.time()

for i in range(10):
    guardrails.check_iteration_limit(i)
    guardrails.check_timeout()
    # ... agent logic ...
    guardrails.check_budget(tokens=500)  # track per call
```

---

### 5.C4: Trace and Log Reasoning Steps

```python
import logging
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class AgentStep:
    step_number: int
    thought: str
    action: str
    action_input: dict
    observation: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    tokens_used: int = 0

class AgentTracer:
    """Trace and log all agent reasoning steps."""
    
    def __init__(self):
        self.steps: list[AgentStep] = []
        self.logger = logging.getLogger("agent")
        logging.basicConfig(level=logging.INFO, format="%(message)s")
    
    def log_step(self, step_number: int, thought: str, action: str, 
                 action_input: dict, observation: str, tokens: int = 0):
        step = AgentStep(
            step_number=step_number,
            thought=thought,
            action=action,
            action_input=action_input,
            observation=observation,
            tokens_used=tokens,
        )
        self.steps.append(step)
        self.logger.info(f"\n{'='*50}")
        self.logger.info(f"Step {step_number}")
        self.logger.info(f"  Thought: {thought}")
        self.logger.info(f"  Action: {action}({action_input})")
        self.logger.info(f"  Observation: {observation[:200]}")
        self.logger.info(f"  Tokens: {tokens}")
    
    def get_trace_summary(self) -> str:
        """Get a readable summary of the full trace."""
        lines = [f"Agent Trace ({len(self.steps)} steps):"]
        total_tokens = 0
        for step in self.steps:
            lines.append(f"  {step.step_number}. {step.action}({step.action_input}) → {step.observation[:80]}...")
            total_tokens += step.tokens_used
        lines.append(f"  Total tokens: {total_tokens}")
        return "\n".join(lines)

# Use in agent
tracer = AgentTracer()
# ... during agent loop:
tracer.log_step(1, "Need to find population", "web_search", {"query": "population france"}, "68.4 million", tokens=150)
tracer.log_step(2, "Now I can answer", "finish", {"answer": "68.4M"}, "Done", tokens=50)
print(tracer.get_trace_summary())
```

---

### 5.C5: Test Edge Cases

```python
import pytest

def test_tool_failure():
    """Agent should handle tool failures gracefully."""
    result = run_agent("Read the file /nonexistent/path.txt and summarize it")
    assert "not found" in result.lower() or "unable" in result.lower()
    # Should NOT crash — should report the failure gracefully

def test_ambiguous_input():
    """Agent should ask for clarification or make reasonable assumptions."""
    result = run_agent("Calculate it")
    assert result is not None  # Should not crash
    assert len(result) > 0  # Should produce some response

def test_iteration_limit():
    """Agent should stop at max iterations."""
    # With a very complex question that could loop forever
    result = run_agent("Keep searching until you find the meaning of life", max_iterations=3)
    assert result is not None  # Should terminate, not hang

def test_no_tool_needed():
    """Agent should answer directly when no tools needed."""
    result = run_agent("What is 2 + 2?")
    assert "4" in result  # Should answer without calling calculator

def test_multiple_tools():
    """Agent can use multiple tools in sequence."""
    result = run_agent("What is the population of France divided by 10?")
    # Should: 1) search for population, 2) calculate division
    assert result is not None

# Run: pytest test_agent.py -v
```

---

*Continue to Week 6...*


---

# WEEK 6: Multi-Agent Systems

**Goal:** Design and implement multi-agent pipelines with orchestration, routing, and memory.

---

## Theory

### 6.1 Why Multi-Agent? Parallelism, Specialisation, Separation of Concerns

A single agent with 20 tools becomes unwieldy — it's hard to prompt, hard to debug, and expensive (huge system prompt). Multi-agent systems split responsibilities:

- **Parallelism**: Researcher and Fact-Checker work simultaneously
- **Specialisation**: Each agent has a focused prompt and limited tools (better at its job)
- **Separation of concerns**: The Writer doesn't need access to search tools; the Researcher doesn't need formatting tools
- **Composability**: Swap out one agent without rewriting the whole system

**Analogy**: A single person trying to write a research paper, fact-check, format citations, and edit simultaneously vs. a team where each person has one role.

---

### 6.2 Agentic Design Patterns

**Orchestrator-Worker**: A central "manager" agent decides which worker agents to invoke and in what order. Workers report back to the orchestrator.

**Routing**: A lightweight classifier routes incoming requests to the right specialist agent (e.g., coding questions → Code Agent, math → Math Agent).

**Hierarchical**: Nested layers — a top-level planner creates sub-goals, each delegated to a mid-level agent, which may further delegate to worker agents.

**Pipeline/Sequential**: Fixed sequence — Agent A → Agent B → Agent C. Each transforms the output for the next.

**Debate/Critic**: Two agents argue different perspectives; a Judge agent picks the best answer.

---

### 6.3 Problems Unique to Multi-Agent Systems

- **Orchestration complexity**: Who goes next? How do you handle failures mid-pipeline?
- **Information isolation**: Agent B needs context from Agent A's work, but how much to pass?
- **Planning failures**: The orchestrator makes a bad plan and agents waste work
- **State management**: Where is the current state of the overall task? Who owns it?
- **Error propagation**: Agent A hallucinates → Agent B builds on that hallucination
- **Cost explosion**: 3 agents × 5 iterations × 3 tool calls each = 45 LLM calls for one user query

---

### 6.4 Memory Systems for Agents

**Short-term memory (scratchpad)**: A shared text buffer that agents read/write during a single task. Cleared after the task completes.

**Long-term memory (vector store)**: Persistent storage of past interactions, learned facts, and user preferences. Survives across sessions.

**Episodic memory**: "Last time the user asked about Python, they preferred concise answers" — stored as embeddings, retrieved when relevant.

**Working memory**: The current context window contents — what the agent can "see" right now. Limited by context window size.

---

### 6.5 Cost and Latency Tradeoffs

| Factor | Single Agent | Multi-Agent |
|--------|-------------|-------------|
| Latency | 5-15s | 15-60s |
| Cost per query | $0.01-0.05 | $0.05-0.50 |
| Quality ceiling | Medium | High |
| Debugging ease | Simple | Complex |
| Failure modes | Few | Many |

**Optimizations**: Run independent agents in parallel, cache intermediate results, use cheaper models for routing/classification, limit retry counts.

---

## Coding Practice

### 6.C1: Set Up LangGraph

```python
# pip install langgraph langchain-openai

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated
import operator

# Define the state that flows between agents
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]  # Append-only message list
    current_step: str
    research_notes: str
    draft: str
    feedback: str
    final_output: str

# Verify setup
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
response = llm.invoke("Say hello")
print(f"LangGraph ready! LLM says: {response.content}")
```

---

### 6.C2: Build a 3-Agent Pipeline (Researcher + Writer + Critic)

```python
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

def researcher_agent(state: AgentState) -> AgentState:
    """Research agent: gathers information on the topic."""
    topic = state["messages"][-1] if state["messages"] else "general topic"
    
    response = llm.invoke([
        SystemMessage(content="You are a thorough researcher. Gather key facts, statistics, and insights on the given topic. Be comprehensive but concise."),
        HumanMessage(content=f"Research this topic thoroughly: {topic}"),
    ])
    
    return {"research_notes": response.content, "current_step": "writer"}

def writer_agent(state: AgentState) -> AgentState:
    """Writer agent: creates polished content from research notes."""
    response = llm.invoke([
        SystemMessage(content="You are an excellent technical writer. Create a clear, well-structured article from the research notes provided."),
        HumanMessage(content=f"Write a polished article based on these research notes:\n\n{state['research_notes']}"),
    ])
    
    return {"draft": response.content, "current_step": "critic"}

def critic_agent(state: AgentState) -> AgentState:
    """Critic agent: reviews and provides actionable feedback."""
    response = llm.invoke([
        SystemMessage(content="You are a constructive critic. Review the article for accuracy, clarity, and completeness. Provide specific suggestions."),
        HumanMessage(content=f"Review this article:\n\n{state['draft']}"),
    ])
    
    return {"feedback": response.content, "current_step": "done"}

def should_continue(state: AgentState) -> str:
    """Route to next agent or end."""
    step = state.get("current_step", "researcher")
    if step == "writer":
        return "writer"
    elif step == "critic":
        return "critic"
    else:
        return END

# Build the graph
workflow = StateGraph(AgentState)
workflow.add_node("researcher", researcher_agent)
workflow.add_node("writer", writer_agent)
workflow.add_node("critic", critic_agent)

workflow.set_entry_point("researcher")
workflow.add_edge("researcher", "writer")
workflow.add_edge("writer", "critic")
workflow.add_edge("critic", END)

app = workflow.compile()

# Run the pipeline
result = app.invoke({
    "messages": ["Explain how vector databases work and their use in AI applications"],
    "current_step": "researcher",
    "research_notes": "",
    "draft": "",
    "feedback": "",
    "final_output": "",
})

print("=== Research Notes ===")
print(result["research_notes"][:500])
print("\n=== Draft ===")
print(result["draft"][:500])
print("\n=== Critic Feedback ===")
print(result["feedback"][:500])
```

---

### 6.C3: Router That Delegates to Specialist Agents

```python
from openai import OpenAI
import json

client = OpenAI()

def route_query(query: str) -> str:
    """Classify the query and route to the appropriate specialist."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": """Classify this query into one category:
- "code": Programming/coding questions
- "math": Mathematical calculations
- "research": General knowledge/research questions
- "creative": Creative writing tasks
Return ONLY the category name."""},
            {"role": "user", "content": query},
        ],
        temperature=0.0,
        max_tokens=10,
    )
    return response.choices[0].message.content.strip().lower()

def code_agent(query: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert programmer. Write clean, well-documented code."},
            {"role": "user", "content": query},
        ],
    )
    return response.choices[0].message.content

def math_agent(query: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a mathematician. Show your work step by step."},
            {"role": "user", "content": query},
        ],
    )
    return response.choices[0].message.content

def research_agent(query: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a research analyst. Provide thorough, cited answers."},
            {"role": "user", "content": query},
        ],
    )
    return response.choices[0].message.content

AGENT_MAP = {
    "code": code_agent,
    "math": math_agent,
    "research": research_agent,
    "creative": research_agent,  # fallback
}

def routed_query(query: str) -> str:
    """Route query to the best specialist agent."""
    category = route_query(query)
    print(f"  Routed to: {category}")
    agent_fn = AGENT_MAP.get(category, research_agent)
    return agent_fn(query)

# Test
print(routed_query("Write a Python function for binary search"))
print(routed_query("What is the integral of x^2?"))
print(routed_query("Explain how TCP/IP works"))
```

---

### 6.C4: Shared Memory (Scratchpad)

```python
class SharedMemory:
    """Thread-safe shared scratchpad for multi-agent communication."""
    
    def __init__(self):
        self.entries: list[dict] = []
    
    def write(self, agent_name: str, key: str, value: str):
        """Agent writes to shared memory."""
        self.entries.append({
            "agent": agent_name,
            "key": key,
            "value": value,
            "timestamp": len(self.entries),
        })
    
    def read(self, key: str = None, agent: str = None) -> list[dict]:
        """Read from shared memory with optional filters."""
        results = self.entries
        if key:
            results = [e for e in results if e["key"] == key]
        if agent:
            results = [e for e in results if e["agent"] == agent]
        return results
    
    def read_latest(self, key: str) -> str:
        """Get the most recent value for a key."""
        entries = self.read(key=key)
        return entries[-1]["value"] if entries else ""
    
    def to_context_string(self) -> str:
        """Convert memory to a string for LLM context."""
        lines = []
        for e in self.entries:
            lines.append(f"[{e['agent']}] {e['key']}: {e['value'][:200]}")
        return "\n".join(lines)

# Usage in multi-agent pipeline
memory = SharedMemory()

# Researcher writes findings
memory.write("researcher", "key_facts", "Vector DBs use HNSW for approximate nearest neighbor search...")
memory.write("researcher", "statistics", "Market growing 25% YoY, $2.1B by 2028")

# Writer reads researcher's notes
facts = memory.read_latest("key_facts")
stats = memory.read_latest("statistics")
print(f"Writer sees: {facts[:80]}...")

# Critic writes feedback
memory.write("critic", "feedback", "Missing comparison with traditional databases")

# Full context for any agent
print(memory.to_context_string())
```

---

### 6.C5: Instrument with Tracing

```python
import time
from dataclasses import dataclass, field
from typing import Optional

@dataclass 
class AgentSpan:
    agent_name: str
    start_time: float
    end_time: Optional[float] = None
    input_text: str = ""
    output_text: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    error: Optional[str] = None
    
    @property
    def duration_ms(self) -> float:
        if self.end_time:
            return (self.end_time - self.start_time) * 1000
        return 0

class PipelineTracer:
    """Simple tracing for multi-agent pipelines."""
    
    def __init__(self, pipeline_name: str):
        self.pipeline_name = pipeline_name
        self.spans: list[AgentSpan] = []
        self.start_time = time.time()
    
    def start_span(self, agent_name: str, input_text: str) -> AgentSpan:
        span = AgentSpan(agent_name=agent_name, start_time=time.time(), input_text=input_text)
        self.spans.append(span)
        return span
    
    def end_span(self, span: AgentSpan, output: str, tokens_in: int = 0, tokens_out: int = 0):
        span.end_time = time.time()
        span.output_text = output
        span.tokens_in = tokens_in
        span.tokens_out = tokens_out
    
    def print_summary(self):
        total_time = (time.time() - self.start_time) * 1000
        total_tokens = sum(s.tokens_in + s.tokens_out for s in self.spans)
        
        print(f"\n{'='*60}")
        print(f"Pipeline: {self.pipeline_name}")
        print(f"Total time: {total_time:.0f}ms | Total tokens: {total_tokens}")
        print(f"{'='*60}")
        for span in self.spans:
            status = "✓" if not span.error else "✗"
            print(f"  {status} {span.agent_name}: {span.duration_ms:.0f}ms ({span.tokens_in}+{span.tokens_out} tokens)")
        print(f"{'='*60}")

# Usage
tracer = PipelineTracer("research_pipeline")

span = tracer.start_span("researcher", "Research vector databases")
# ... agent work ...
tracer.end_span(span, "Vector DBs use HNSW...", tokens_in=200, tokens_out=500)

span = tracer.start_span("writer", "Write article from notes")
# ... agent work ...
tracer.end_span(span, "# Vector Databases\n...", tokens_in=800, tokens_out=1200)

tracer.print_summary()
```

---

*Continue to Week 7...*


---

# WEEK 7: Context Engineering, Memory & Evaluation

**Goal:** Master what goes into the context window and rigorously evaluate your system's quality.

---

## Theory

### 7.1 Context Engineering vs Prompt Engineering

**Prompt engineering** focuses on *how you write instructions* — word choice, few-shot examples, chain-of-thought patterns.

**Context engineering** focuses on *what information is available* to the LLM when it generates a response — which documents, tools, history, and metadata are in the window.

Think of it this way: prompt engineering is writing a good question; context engineering is making sure the right textbook is open on the desk when you ask it.

In production, context engineering matters more because:
- The LLM can only reason about what it can see
- Stuffing irrelevant context degrades quality
- Missing context causes hallucination
- Context window space is finite and expensive

---

### 7.2 What Goes in the Context

A typical production LLM call includes:

1. **System instructions** (~500-2000 tokens): Role, behavior rules, output format
2. **Tool definitions** (~200-1000 tokens): Function schemas for each available tool
3. **Retrieved documents** (~1000-4000 tokens): RAG results relevant to the query
4. **Conversation history** (~500-5000 tokens): Prior messages in this session
5. **User message** (~50-500 tokens): The current query
6. **Available output space** (~1000-4000 tokens): Room for the response

Total budget: context_window - output_tokens = available_input_tokens

When you exceed the budget, you must **prioritize**: trim history? Reduce retrieved docs? Shorten system prompt?

---

### 7.3 Quantitative Evals

| Metric | What It Measures | Formula |
|--------|-----------------|---------|
| **Accuracy** | Correct answers / total | correct / total |
| **F1 Score** | Balance of precision & recall | 2 × (P × R) / (P + R) |
| **Exact Match** | Output matches ground truth exactly | matches / total |
| **Tool-Call Accuracy** | Agent calls correct tool with correct args | correct_calls / total_calls |
| **Retrieval Precision@K** | Relevant docs in top-K results | relevant_in_K / K |
| **Latency P50/P95** | Response time distribution | percentile of response times |

These metrics require **ground-truth datasets** — hand-labeled examples with expected outputs.

---

### 7.4 LLM-as-a-Judge

When human eval is too expensive, use a strong LLM to evaluate a weaker LLM's outputs:

1. **Design a rubric**: Define exactly what good/bad looks like (1-5 scale per dimension)
2. **Create evaluation prompt**: Tell the judge LLM the rubric, context, and response to evaluate
3. **Score**: Judge returns a numerical score with reasoning
4. **Calibrate**: Compare judge scores with human scores on a subset to check alignment

Dimensions to evaluate: helpfulness, accuracy, relevance, completeness, safety, formatting.

**Pitfalls**: Judge LLMs have their own biases (prefer longer answers, prefer their own style). Always calibrate against human judgment.

---

### 7.5 Observability: Tracing, Logging, and Monitoring

In production, you need to see what's happening inside your AI system:

- **Tracing**: Follow a request through the full pipeline (retrieval → rerank → LLM → output)
- **Logging**: Record inputs, outputs, latencies, errors, and token usage for every call
- **Monitoring**: Dashboards showing quality metrics, cost, latency, error rates over time
- **Alerting**: Notify when quality drops, latency spikes, or costs exceed budget

Tools: LangSmith, Weights & Biases, Helicone, custom logging to your observability stack.

---

## Coding Practice

### 7.C1: Audit Context Window Usage

```python
import tiktoken

def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """Count tokens in a text string."""
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

def audit_context(system_prompt: str, tools: list[dict], history: list[dict], 
                  retrieved_docs: list[str], user_message: str, model: str = "gpt-4o-mini"):
    """Audit how context window budget is being used."""
    
    model_limit = 128000  # GPT-4o-mini context window
    desired_output = 4000  # Reserve for response
    
    system_tokens = count_tokens(system_prompt)
    tools_tokens = count_tokens(str(tools))
    history_tokens = sum(count_tokens(m.get("content", "")) for m in history)
    docs_tokens = sum(count_tokens(doc) for doc in retrieved_docs)
    user_tokens = count_tokens(user_message)
    
    total_input = system_tokens + tools_tokens + history_tokens + docs_tokens + user_tokens
    available = model_limit - desired_output
    usage_pct = (total_input / available) * 100
    
    print(f"Context Window Audit ({model})")
    print(f"{'='*50}")
    print(f"  System prompt:    {system_tokens:>6} tokens ({system_tokens/available*100:.1f}%)")
    print(f"  Tool definitions: {tools_tokens:>6} tokens ({tools_tokens/available*100:.1f}%)")
    print(f"  History:          {history_tokens:>6} tokens ({history_tokens/available*100:.1f}%)")
    print(f"  Retrieved docs:   {docs_tokens:>6} tokens ({docs_tokens/available*100:.1f}%)")
    print(f"  User message:     {user_tokens:>6} tokens ({user_tokens/available*100:.1f}%)")
    print(f"  {'─'*40}")
    print(f"  Total input:      {total_input:>6} tokens ({usage_pct:.1f}%)")
    print(f"  Budget remaining: {available - total_input:>6} tokens")
    print(f"  Output reserved:  {desired_output:>6} tokens")
    
    if usage_pct > 80:
        print(f"  ⚠️  WARNING: Context usage at {usage_pct:.0f}% — consider trimming!")
    
    return {"total": total_input, "available": available, "pct": usage_pct}
```

---

### 7.C2: Context Budget Manager

```python
class ContextBudgetManager:
    """Dynamically manage context window budget across components."""
    
    def __init__(self, model_limit: int = 128000, output_reserve: int = 4000):
        self.model_limit = model_limit
        self.output_reserve = output_reserve
        self.budget = model_limit - output_reserve
        
        # Priority-based allocation (higher = keep more)
        self.priorities = {
            "system_prompt": 10,  # Never trim
            "user_message": 9,   # Never trim
            "tools": 8,          # Rarely trim
            "retrieved_docs": 5, # Trim if needed
            "history": 3,        # Trim aggressively
        }
    
    def fit_to_budget(self, components: dict[str, str]) -> dict[str, str]:
        """Trim components to fit within budget, respecting priorities."""
        
        # Calculate current sizes
        sizes = {k: count_tokens(v) for k, v in components.items()}
        total = sum(sizes.values())
        
        if total <= self.budget:
            return components  # Already fits
        
        # Need to trim. Start with lowest priority
        trimmed = dict(components)
        sorted_components = sorted(self.priorities.items(), key=lambda x: x[1])
        
        for component_name, priority in sorted_components:
            if component_name not in trimmed:
                continue
            
            current_total = sum(count_tokens(v) for v in trimmed.values())
            if current_total <= self.budget:
                break
            
            overage = current_total - self.budget
            component_tokens = count_tokens(trimmed[component_name])
            
            if component_name == "history":
                # Trim oldest messages first
                trimmed[component_name] = self._trim_history(trimmed[component_name], overage)
            elif component_name == "retrieved_docs":
                # Keep fewer docs
                trimmed[component_name] = self._trim_docs(trimmed[component_name], overage)
        
        return trimmed
    
    def _trim_history(self, history_str: str, tokens_to_cut: int) -> str:
        """Remove oldest messages until budget is met."""
        messages = history_str.split("\n\n")
        while count_tokens("\n\n".join(messages)) > count_tokens(history_str) - tokens_to_cut:
            if len(messages) <= 2:  # Keep at least last exchange
                break
            messages.pop(0)
        return "\n\n".join(messages)
    
    def _trim_docs(self, docs_str: str, tokens_to_cut: int) -> str:
        """Remove lowest-ranked documents."""
        docs = docs_str.split("\n---\n")
        while count_tokens("\n---\n".join(docs)) > count_tokens(docs_str) - tokens_to_cut:
            if len(docs) <= 1:
                break
            docs.pop()  # Remove last (lowest ranked)
        return "\n---\n".join(docs)

# Usage
manager = ContextBudgetManager(model_limit=8000, output_reserve=1000)  # Small for demo
components = {
    "system_prompt": "You are a helpful assistant...",
    "user_message": "What is HNSW?",
    "retrieved_docs": "Doc 1: HNSW is...\n---\nDoc 2: Vector indexing...\n---\nDoc 3: ANN algorithms...",
    "history": "User: Hello\nAssistant: Hi!\n\nUser: Tell me about RAG\nAssistant: RAG is...",
}
fitted = manager.fit_to_budget(components)
```

---

### 7.C3: Quantitative Eval Suite

```python
from dataclasses import dataclass

@dataclass
class EvalExample:
    question: str
    expected_answer: str
    expected_tool: str = ""  # For agent evals

def exact_match(predicted: str, expected: str) -> float:
    return 1.0 if predicted.strip().lower() == expected.strip().lower() else 0.0

def contains_match(predicted: str, expected: str) -> float:
    return 1.0 if expected.strip().lower() in predicted.strip().lower() else 0.0

def run_quantitative_eval(pipeline_fn, eval_set: list[EvalExample]) -> dict:
    """Run eval and compute metrics."""
    exact_scores = []
    contains_scores = []
    
    for example in eval_set:
        predicted = pipeline_fn(example.question)
        exact_scores.append(exact_match(predicted, example.expected_answer))
        contains_scores.append(contains_match(predicted, example.expected_answer))
    
    return {
        "exact_match": sum(exact_scores) / len(exact_scores),
        "contains_match": sum(contains_scores) / len(contains_scores),
        "total_examples": len(eval_set),
    }

# Build eval set
eval_set = [
    EvalExample("What is the capital of France?", "Paris"),
    EvalExample("Who created Python?", "Guido van Rossum"),
    EvalExample("What year was Git created?", "2005"),
]

results = run_quantitative_eval(my_pipeline, eval_set)
print(f"Exact Match: {results['exact_match']:.2%}")
print(f"Contains Match: {results['contains_match']:.2%}")
```

---

### 7.C4: LLM-as-a-Judge Evaluator

```python
from openai import OpenAI

client = OpenAI()

JUDGE_RUBRIC = """Rate the response on a scale of 1-5 for each dimension:

1. **Accuracy** (1-5): Are the facts correct?
2. **Completeness** (1-5): Does it fully address the question?
3. **Clarity** (1-5): Is it well-written and easy to understand?
4. **Relevance** (1-5): Does it stay on topic?

Return JSON: {"accuracy": N, "completeness": N, "clarity": N, "relevance": N, "reasoning": "..."}
"""

def llm_judge(question: str, response: str, context: str = "") -> dict:
    """Use GPT-4 as a judge to evaluate a response."""
    import json
    
    judge_prompt = f"""Evaluate this AI response using the rubric below.

Question: {question}
{'Context provided: ' + context if context else ''}
Response to evaluate: {response}

{JUDGE_RUBRIC}"""
    
    result = client.chat.completions.create(
        model="gpt-4o",  # Use strongest model as judge
        messages=[
            {"role": "system", "content": "You are an impartial evaluator. Be strict but fair."},
            {"role": "user", "content": judge_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    
    return json.loads(result.choices[0].message.content)

# Evaluate
scores = llm_judge(
    question="How does HNSW indexing work?",
    response="HNSW builds a multi-layer graph where each layer has fewer nodes...",
)
print(f"Accuracy: {scores['accuracy']}/5")
print(f"Completeness: {scores['completeness']}/5")
print(f"Reasoning: {scores['reasoning']}")
```

---

### 7.C5: Before/After Eval Comparison

```python
def compare_pipelines(pipeline_a, pipeline_b, eval_set: list[EvalExample], judge_fn) -> dict:
    """Compare two pipelines using both quantitative and qualitative metrics."""
    
    results_a = {"scores": [], "judge_scores": []}
    results_b = {"scores": [], "judge_scores": []}
    
    for example in eval_set:
        # Run both pipelines
        output_a = pipeline_a(example.question)
        output_b = pipeline_b(example.question)
        
        # Quantitative
        results_a["scores"].append(contains_match(output_a, example.expected_answer))
        results_b["scores"].append(contains_match(output_b, example.expected_answer))
        
        # Qualitative (LLM judge)
        judge_a = judge_fn(example.question, output_a)
        judge_b = judge_fn(example.question, output_b)
        results_a["judge_scores"].append(sum(v for k, v in judge_a.items() if k != "reasoning") / 4)
        results_b["judge_scores"].append(sum(v for k, v in judge_b.items() if k != "reasoning") / 4)
    
    avg = lambda lst: sum(lst) / len(lst) if lst else 0
    
    print(f"{'Metric':<25} {'Pipeline A':>12} {'Pipeline B':>12} {'Delta':>8}")
    print("-" * 60)
    print(f"{'Contains Match':<25} {avg(results_a['scores']):>12.2%} {avg(results_b['scores']):>12.2%} {avg(results_b['scores'])-avg(results_a['scores']):>+8.2%}")
    print(f"{'Judge Score (avg)':<25} {avg(results_a['judge_scores']):>12.2f} {avg(results_b['judge_scores']):>12.2f} {avg(results_b['judge_scores'])-avg(results_a['judge_scores']):>+8.2f}")
```

---

# WEEK 8: Capstone Project

**Goal:** Ship a production-grade AI system that demonstrates the full stack: RAG + Agents + Evals.

---

## Theory

### 8.1 AI Engineering Best Practices: Prototype to Production

**Prototype** (Week 1-3): Get something working in a notebook. Optimize for speed of iteration.

**MVP** (Week 4-6): Structure into modules. Add error handling. Deploy behind an API.

**Production** (Week 7-8): Add evals, monitoring, cost controls, and reliability.

Key principles:
- **Start simple**: Vanilla RAG before GraphRAG. Single agent before multi-agent.
- **Measure before optimizing**: Don't add a reranker until you've measured baseline accuracy.
- **Fail gracefully**: Every external call (LLM API, vector DB, tools) can fail. Handle it.
- **Cost awareness**: Track cost per query. Set budgets. Use cheaper models for routing/classification.

---

### 8.2 Engineering Decision Framework

| Decision | Options | Choose Based On |
|----------|---------|----------------|
| RAG type | Vanilla / Advanced / Graph / Agentic | Data structure + question complexity |
| Agent pattern | Single / Multi / No agent | Task variability + tool count |
| Embedding model | OpenAI / Cohere / Open-source | Cost + quality + privacy needs |
| Vector DB | Chroma / Pinecone / pgvector | Scale + managed vs self-hosted |
| LLM | GPT-4o / Claude / Llama | Cost + quality + latency + privacy |
| Framework | LangChain / LlamaIndex / Raw SDK | Flexibility vs speed of development |

---

### 8.3 How to Scope a Capstone: MVP Definition

1. **Pick a real problem** you care about (not a toy example)
2. **Define success criteria**: What does "good enough" look like? (e.g., "answers 80% of test questions correctly")
3. **Cut scope ruthlessly**: No auth, no fancy UI, no multi-language support for v1
4. **Time-box**: 10-12 hours total. Plan for 6 hours of building, 4 hours of eval and polish.

---

### 8.4 Presenting AI Systems

Structure your demo:
1. **Problem statement** (30s): What does this solve?
2. **Architecture diagram** (30s): High-level components
3. **Live demo** (2-3 min): Show it working on 2-3 example queries
4. **Eval results** (1 min): Show your metrics (accuracy, latency, cost)
5. **Limitations & next steps** (30s): What doesn't work yet?

---

### 8.5 Production Gotchas

- **Rate limits**: OpenAI limits requests/minute. Implement exponential backoff.
- **Cost explosion**: A bug in your agent loop = hundreds of API calls. Set hard caps.
- **Latency variance**: P50 might be 2s but P99 is 30s. Measure percentiles.
- **Model updates**: Provider changes model behavior. Pin versions. Run evals after updates.
- **Context window overflow**: Long conversations exceed limits. Implement sliding window.
- **Embedding model changes**: If you change embedding models, you must re-embed ALL documents.

---

## Coding Practice

### 8.C1: Define Capstone Scope

```markdown
# Capstone: AI Research Assistant for Technical Papers

## Problem
Researchers spend hours reading papers to find specific methods and results.

## MVP Scope
- Ingest 10 arXiv papers (PDF)
- RAG over papers with Advanced retrieval (HyDE + reranking)
- Agent with tools: search_papers, extract_methods, compare_approaches
- Eval: 20 ground-truth Q&A pairs, target 75% accuracy

## Architecture
- Chunking: recursive, 500 tokens with 50 overlap
- Embeddings: text-embedding-3-small
- Vector DB: ChromaDB (local)
- LLM: GPT-4o-mini (cost-effective)
- Agent: Single ReAct agent with 3 tools

## Out of Scope (v1)
- User auth, multi-user, UI (CLI only)
- Real-time paper ingestion
- Citation generation
- Multi-language support

## Success Criteria
- Contains-match accuracy ≥ 75% on eval set
- Average latency < 10s
- Cost per query < $0.05
```

---

### 8.C2: Architecture Decision Doc Template

```python
ARCHITECTURE_DOC = """
# Architecture Decision Record: {project_name}

## Context
{what_problem_are_we_solving}

## Decision

### RAG Architecture: {choice}
**Why**: {reasoning}
**Alternatives considered**: {alternatives}

### Agent Pattern: {choice}
**Why**: {reasoning}

### LLM Choice: {choice}
**Why**: {reasoning} (cost: ${cost_per_1k_tokens}, latency: {avg_latency}ms)

### Vector Database: {choice}
**Why**: {reasoning}

## Tradeoffs Accepted
- {tradeoff_1}
- {tradeoff_2}

## Metrics
- Accuracy: {accuracy}%
- Avg latency: {latency}ms
- Cost per query: ${cost}
- Eval set size: {n} examples

## Next Steps
- {next_1}
- {next_2}
"""
```

---

### 8.C3: Production Cost Tracker

```python
from dataclasses import dataclass, field
from typing import Optional
import time

@dataclass
class QueryMetrics:
    query_id: str
    start_time: float
    end_time: Optional[float] = None
    input_tokens: int = 0
    output_tokens: int = 0
    num_llm_calls: int = 0
    num_tool_calls: int = 0
    retrieval_count: int = 0
    success: bool = True
    error: Optional[str] = None
    
    @property
    def latency_ms(self) -> float:
        if self.end_time:
            return (self.end_time - self.start_time) * 1000
        return 0
    
    @property
    def estimated_cost(self) -> float:
        # GPT-4o-mini pricing: $0.15/1M input, $0.60/1M output
        input_cost = (self.input_tokens / 1_000_000) * 0.15
        output_cost = (self.output_tokens / 1_000_000) * 0.60
        return input_cost + output_cost

class ProductionMonitor:
    """Track cost, latency, and quality in production."""
    
    def __init__(self, daily_budget: float = 10.0):
        self.daily_budget = daily_budget
        self.queries: list[QueryMetrics] = []
    
    def record(self, metrics: QueryMetrics):
        self.queries.append(metrics)
        
        # Check budget
        daily_cost = sum(q.estimated_cost for q in self.queries)
        if daily_cost > self.daily_budget:
            raise RuntimeError(f"Daily budget exceeded: ${daily_cost:.2f} > ${self.daily_budget}")
    
    def summary(self) -> dict:
        if not self.queries:
            return {}
        
        latencies = [q.latency_ms for q in self.queries]
        costs = [q.estimated_cost for q in self.queries]
        
        return {
            "total_queries": len(self.queries),
            "success_rate": sum(1 for q in self.queries if q.success) / len(self.queries),
            "avg_latency_ms": sum(latencies) / len(latencies),
            "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95)],
            "total_cost": sum(costs),
            "avg_cost_per_query": sum(costs) / len(costs),
            "total_tokens": sum(q.input_tokens + q.output_tokens for q in self.queries),
        }

# Usage
monitor = ProductionMonitor(daily_budget=5.0)

metrics = QueryMetrics(query_id="q-001", start_time=time.time())
# ... run pipeline ...
metrics.end_time = time.time()
metrics.input_tokens = 1500
metrics.output_tokens = 800
metrics.num_llm_calls = 3

monitor.record(metrics)
print(monitor.summary())
```

---

## End of Course

Congratulations on completing the AI Engineering study guide! You now have the knowledge to:

1. ✅ Understand LLMs from tokens to attention
2. ✅ Build RAG pipelines (vanilla → advanced → graph → agentic)
3. ✅ Create single and multi-agent systems
4. ✅ Engineer context windows and evaluate quality
5. ✅ Ship production-grade AI applications

**Next steps**: Pick a real problem, build your capstone, and keep iterating. The field moves fast — the fundamentals you've learned here will serve as your foundation for whatever comes next.
