# NLP Agent Pipeline

An integrated system comprising a specialized sentiment analysis API and an agentic layer that utilizes this API as a tool for autonomous reasoning and text analysis.

## System Architecture

### 1. Text Analysis API
A production-ready microservice that provides multiple forms of text intelligence.
- **Core:** Built with Flask and HuggingFace Transformers.
- **Sentiment Model:** DistilBERT fine-tuned on SST-2 (`distilbert-base-uncased-finetuned-sst-2-english`).
- **Topic Classification:** Zero-shot topic classification using a cross-encoder model (`facebook/bart-large-mnli`) for labeling text into categories such as business, technology, sports, and politics.
- **Keyword Extraction:** KeyBERT-based keyword and phrase extraction for identifying salient terms in a document or input string.
- **Deployment:** Containerized via Docker and scalable using Gunicorn.

### 2. Agentic Layer
An implementation of AI agents that treat the text analysis API as an external tool.
- **Orchestration:** Agents use LLMs to determine when to invoke the API endpoints to analyze text.
- **Providers:** Experimental implementations using Groq (cloud-based) and Ollama (local).
- **Workflows:** Autonomous decision-making loops where the agent fetches sentiment, topic, and keyword data before generating a final synthesis or response.
- **Integrated agent endpoint:** `POST /agent` accepts a question and text, then uses a Groq-backed agent to call the sentiment, topic-classification, and keyword-extraction endpoints as needed.

## Project Structure
- `app/`: Model loading and text analysis logic.
- `app/predict.py`: Sentiment prediction logic.
- `app/classifier.py`: Topic classification logic.
- `app/keywords.py`: Keyword extraction logic.
- `app/agent.py`: Groq-backed agent loop that selects and processes analysis tools.
- `app/tools.py`: Tool definitions and functions that call the text analysis endpoints.
- `model/`: Trained model artifacts.
- `server.py`: Flask API implementation exposing sentiment, keyword, classification, and agent endpoints.
- `experiments/`: Agentic implementations:
    - `01_groq_hello.py` & `02_groq_tool.py`: Groq-based agent tool integration.
    - `03_ollama_tool.py` & `05_ollama_real_tool.py`: Local agent integration via Ollama.
    - `04_real_tool.py`: Integration of the live API as a functional tool.
- `Dockerfile`: API container configuration.
- `requirements.txt`: System dependencies.

## Setup and Execution

### Installation
```bash
pip install -r requirements.txt
```

### Running the API
**Local:**
```bash
python server.py
```
**Production:**
```bash
gunicorn --bind 0.0.0.0:5000 --workers 4 server:app
```
**Docker:**
```bash
docker build -t sentiment-api .
docker run -p 5000:5000 sentiment-api
```

### Agent Execution Setup

**1. Groq Setup:**
- Obtain a Groq API key from [Groq Console](https://console.groq.com/).
- Set it as an environment variable:
  ```bash
  export GROQ_API_KEY='your_api_key_here'

**2. Ollama Setup (for local agents):**
- Install Ollama from [ollama.com](https://ollama.com)
- Start the Ollama server:
  ```bash
  ollama serve
  ```
- Pull the required model (in a separate terminal):
  ```bash
  ollama pull llama3.1
  ```


### Agent Execution
To run the agentic experiments, ensure the API is running first.

**Groq-based Agents:**
```bash
python experiments/02_groq_tool.py
python experiments/04_real_tool.py
```

**Ollama-based Agents (Local):**
```bash
python experiments/03_ollama_tool.py
python experiments/05_ollama_real_tool.py
```

**Integrated agent endpoint:** Start the API with `python server.py`, set `GROQ_API_KEY`, then send a request:
```bash
curl -X POST http://localhost:5001/agent \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the sentiment and main topics of this text?","text":"The new iPhone has an amazing camera but the battery life is disappointing"}'
```


## API Reference

### Predict Sentiment
- **Endpoint:** `POST /predict`
- **Request:** `{"text": "string"}`
- **Response:** `{"text": "string", "sentiment": "positive/negative", "confidence": float}`

### Extract Keywords
- **Endpoint:** `POST /keywords`
- **Request:** `{"text": "string"}`
- **Response:** `{"text": "string", "keywords": [{"keyword": "string", "score": float}]}`

### Classify Topic
- **Endpoint:** `POST /classify`
- **Request:** `{"text": "string"}`
- **Response:** `{"text": "string", "label": "string", "score": float}`

### Agent Analysis
- **Endpoint:** `POST /agent`
- **Request:** `{"question": "string", "text": "string"}`
- **Response:** `{"response": "string", "tools_called": ["tool_name"]}`
- The Groq-backed agent may call sentiment prediction, topic classification, and keyword extraction tools to answer the question.

### Health Check
- **Endpoint:** `GET /health`
- **Response:** `{"status": "ok"}`
