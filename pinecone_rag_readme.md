# Pinecone RAG Pipeline with LangChain, Ollama, and Python

This project demonstrates a minimal **Retrieval-Augmented Generation (RAG)** workflow using:
- **Ollama** for LLM inference and embeddings
- **Pinecone** as a vector database
- **LangChain** for composing ingestion and retrieval chains

It includes two scripts:
1. `ingestion.py` – loads and embeds documents into Pinecone.
2. `main.py` – runs a RAG pipeline to answer questions using retrieved context.

---

## 1. Project Structure
```
.
├── main.py               # RAG query pipeline
├── ingestion.py          # Vector ingestion script
├── mediumblog.txt        # Sample document to ingest
├── .env                  # Contains Pinecone INDEX_NAME and API keys
└── README.md
```

---

## 2. Prerequisites
Ensure the following are installed and available:

### Python Packages
```
pip install langchain langchain-community langchain-ollama \
            langchain-pinecone python-dotenv
```

### Services & Tools Required
- **Ollama** running locally (for LLM + embeddings)
- **Pinecone account** and an index created
- **.env file** containing:
```
PINECONE_API_KEY=your-key
INDEX_NAME=your-index-name
```

---

## 3. Ingestion Pipeline (`ingestion.py`)
This script:
1. Loads a text file using `TextLoader`.
2. Splits the document into chunks via `CharacterTextSplitter`.
3. Generates embeddings using `OllamaEmbeddings`.
4. Inserts the vectors into Pinecone using `PineconeVectorStore.from_documents()`.

Run ingestion:
```
python ingestion.py
```

Key components:
- Embedding Model: `nomic-embed-text`
- Chunk size: 1000 characters
- Storage: Pinecone index defined in `.env`

After running, your Pinecone index will contain the embedded chunks.

---

## 4. RAG Pipeline (`main.py`)
This script performs the actual RAG query.

### Workflow:
1. Initialize embeddings and the LLM (`gemma3:1b`).
2. Create a Pinecone retriever.
3. Define a RAG prompt template.
4. Use LangChain Expression Language (LCEL) to build the pipeline:
   - Pass user input
   - Retrieve context from Pinecone
   - Apply prompt template
   - Run query through Ollama LLM

Run the RAG query:
```
python main.py
```

Output example:
```
=== ANSWER ===
<model-generated answer>
```

---

## 5. Key Concepts
### Vector Ingestion
Convert Documents → Chunks → Embeddings → Pinecone Index.

### Retrieval
Given a query, similar vectors are fetched from Pinecone.

### Generation
The retrieved context is injected into a structured prompt and processed by the LLM.

---

## 6. Customization
### Change the embedding or LLM model
Update:
```
embeddings = OllamaEmbeddings(model="nomic-embed-text")
llm = ChatOllama(model="gemma3:1b")
```

### Use custom document sources
Replace the file path in `ingestion.py`:
```
loader = TextLoader("/path/to/your/document.txt")
```

### Change chunking behavior
Modify:
```
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
```

---

## 7. Troubleshooting
### Pinecone errors
- Ensure index exists and `INDEX_NAME` matches.
- Confirm Pinecone API key is valid.

### Ollama model not found
Run:
```
ollama pull gemma3:1b
ollama pull nomic-embed-text
```

### Environment variables not loading
Verify:
```
load_dotenv()
print(os.environ)
```

---

## 8. Summary
This project is a minimal, production-style skeleton for building Retrieval-Augmented Generation applications using:
- **LangChain** for orchestration
- **Ollama** for local AI inference
- **Pinecone** for vector search

It is suitable as a starter template for more advanced RAG systems, chatbots, or document assistants.

---

