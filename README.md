🚀 In 2 weeks built a multi-agent RAG system for Fortune 500 10-K financial intelligence.

Built on 5 years of S&P 500 filings (2M+ chunks, 768D embeddings) - it's an autonomous financial analyst.

🤖 How it works:
• Agents decompose complex financial queries
• Metadata-aware filtering (ticker, year, statement type) intelligently narrows search space
• OpenAI O1 reasoning model synthesizes comprehensive analysis
• RAGAS evaluation ensures answer quality with citation tracking

💻 Tech Stack:
Streamlit • Pinecone • LangChain/LlamaIndex • MLflow • Vertex AI • FastAPI
Example query: "How does AMD's R&D spending compare to Intel over the last 3 years?"

→ Agent breaks it down → Searches specific financial statements → O1 analyzes trends → Returns cited answer with hyperlinks to source docs
Built with full observability (MLflow traces), production-grade vector search, and LLM-as-judge evaluation.
