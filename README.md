# 🤖 AI-Powered CV Analyzer & Job Recruiting Assistant

An intelligent recruitment assistant that leverages AI and NLP to analyze CVs, match candidates with job descriptions, and provide actionable feedback for both candidates and recruiters.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Workflow](#workflow)
- [Installation](#installation)
- [Usage](#usage)
- [Agent Details](#agent-details)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Requirements](#requirements)

---

## 🎯 Overview

This application is a dual-purpose recruiting tool that serves both **job candidates** and **recruiters**:

- **For Candidates**: Upload your CV to receive personalized improvement suggestions tailored to your target role
- **For Recruiters**: Match candidate CVs against job descriptions using semantic similarity scoring and generate comprehensive candidate summaries

The system uses local LLM models via Ollama for privacy-focused, offline AI processing.

---

## ✨ Features

### Candidate Features
- **CV Upload & Parsing**: Extract text from PDF resumes
- **Intelligent Feedback**: Get AI-powered suggestions to improve your CV
- **Role-Specific Tailoring**: Receive recommendations customized for your target job role

### Recruiter Features
- **Job Description Matching**: Calculate semantic similarity between CVs and JDs
- **Scoring Metrics**: View maximum and average match scores
- **Candidate Summaries**: Generate comprehensive summaries highlighting:
  - Skills and competencies
  - Experience overview
  - Notable achievements
  - Potential concerns or red flags

---

## 🏗️ Architecture

The application follows a **multi-agent architecture** where specialized agents handle specific tasks:

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Dashboard                       │
│                    (dashboard.py)                            │
└───────────────┬─────────────────────────────────────────────┘
                │
                ├─────► CVParserAgent
                │       • PDF text extraction
                │       • Text chunking
                │
                ├─────► JDMatcherAgent
                │       • Embedding generation
                │       • Cosine similarity scoring
                │
                ├─────► FeedbackAgent
                │       • CV analysis
                │       • Improvement suggestions
                │
                └─────► SummaryAgent
                        • Candidate summarization
                        • Skills extraction
```

---

## 🔄 Workflow

### Candidate Workflow

1. **Upload CV**: User uploads a PDF resume through the Streamlit interface
2. **CV Parsing**: `CVParserAgent` extracts and chunks the text
3. **Display Analysis**: Parsed content is displayed (truncated preview)
4. **Request Feedback**: User clicks "Get Improvement Suggestions"
5. **AI Analysis**: `FeedbackAgent` analyzes the CV using LLaMA 3.2
6. **Receive Suggestions**: Personalized recommendations are displayed

### Recruiter Workflow

1. **Input JD**: Recruiter pastes the job description
2. **Upload Candidate CV**: Upload the candidate's PDF resume
3. **CV Parsing**: `CVParserAgent` processes the PDF
4. **Semantic Matching**: `JDMatcherAgent` computes similarity scores
5. **Display Scores**: Max and average match scores are shown
6. **Generate Summary** (optional): `SummaryAgent` creates a detailed candidate profile
7. **Review Summary**: Comprehensive candidate analysis is displayed

---

## 📦 Installation

### Prerequisites

- **Python 3.8+**
- **Ollama** installed and running locally
- Required Ollama models:
  - `llama3.2` (for text generation)
  - `nomic-embed-text` (for embeddings)

### Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/aa055/ai-cv-analyzer.git
   cd ai-cv-analyzer
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Ollama models**:
   ```bash
   ollama pull llama3.2
   ollama pull nomic-embed-text
   ```

4. **Run the application**:
   ```bash
   streamlit run dashboard.py
   ```

5. **Access the dashboard**:
   Open your browser to `http://localhost:8501`

---

## 🚀 Usage

### For Candidates

1. Navigate to the **"Candidate View"** tab
2. Upload your CV in PDF format
3. (Optional) Enter your target job role (e.g., "Data Scientist")
4. Click **"Get Improvement Suggestions"**
5. Review the AI-generated recommendations

### For Recruiters

1. Navigate to the **"Recruiter View"** tab
2. Paste the job description in the text area
3. Upload the candidate's CV (PDF)
4. Review the **Match Score** metrics:
   - **Max Score**: Highest similarity between any CV chunk and the JD
   - **Avg Score**: Average similarity across all CV chunks
5. Click **"Generate Summary"** for detailed candidate analysis

---

## 🤖 Agent Details

### 1. CVParserAgent (`cv_parser_agent.py`)

**Purpose**: Extract and preprocess CV content from PDF files

**Key Components**:
- `PyPDFLoader`: Loads and extracts text from PDF documents
- `RecursiveCharacterTextSplitter`: Splits text into chunks (500 chars, 50 overlap)

**Methods**:
- `parse_cv(pdf_path)`: Returns dictionary with full text and chunked text

**Output Structure**:
```python
{
    "text": "full CV text...",
    "chunks": ["chunk1", "chunk2", ...]
}
```

---

### 2. JDMatcherAgent (`jd_matcher_agent.py`)

**Purpose**: Calculate semantic similarity between CV and job descriptions

**Key Components**:
- `OllamaEmbeddings`: Generates vector embeddings using `nomic-embed-text` model
- `cosine_similarity`: Computes similarity scores between vectors

**Methods**:
- `match(cv_chunks, job_description)`: Returns similarity metrics

**Algorithm**:
1. Generate embeddings for each CV chunk
2. Generate embedding for the job description
3. Calculate cosine similarity for each chunk-JD pair
4. Compute maximum and average scores

**Output Structure**:
```python
{
    "similarity_scores": [0.75, 0.82, ...],
    "max_score": 0.89,
    "avg_score": 0.76
}
```

---

### 3. FeedbackAgent (`feedback_agent.py`)

**Purpose**: Provide AI-powered CV improvement suggestions

**Key Components**:
- `Ollama`: LLaMA 3.2 model for text generation

**Methods**:
- `suggest_improvements(raw_cv_text, target_role)`: Returns improvement suggestions

**Prompt Engineering**:
- Analyzes CV clarity and formatting
- Provides role-specific tailoring advice
- Suggests concrete improvements

---

### 4. SummaryAgent (`summary_agent.py`)

**Purpose**: Generate comprehensive candidate summaries for recruiters

**Key Components**:
- `Ollama`: LLaMA 3.2 model for text generation

**Methods**:
- `generate_summary(cv_text)`: Returns structured candidate summary

**Summary Includes**:
- Skills and competencies
- Experience summary
- Notable highlights and achievements
- Potential red flags or concerns

---

## 🛠️ Technology Stack

### Core Framework
- **Streamlit**: Web-based dashboard interface

### AI/ML Libraries
- **LangChain**: Orchestration framework for LLM applications
  - `langchain-core`: Core abstractions
  - `langchain-ollama`: Ollama integration
  - `langchain-community`: Community integrations
  - `langchain_text_splitters`: Text chunking utilities

### NLP & Embeddings
- **Ollama**: Local LLM inference
  - Models: `llama3.2`, `nomic-embed-text`
- **sentence-transformers**: Embedding models

### Data Processing
- **scikit-learn**: Cosine similarity calculations
- **numpy**: Numerical operations
- **pypdf**: PDF text extraction

### Utilities
- **tqdm**: Progress bars
- **python-magic-bin**: File type detection (Windows)

---

## 📁 Project Structure

```
ai-cv-analyzer/
│
├── dashboard.py              # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                # Project documentation
│
└── agents/                  # Agent modules
    ├── __init__.py          # Agent exports
    ├── cv_parser_agent.py   # PDF parsing & text extraction
    ├── jd_matcher_agent.py  # CV-JD similarity matching
    ├── feedback_agent.py    # CV improvement suggestions
    └── summary_agent.py     # Candidate summarization
```

---

## 📝 Requirements

See `requirements.txt` for the complete list of dependencies:

---

## 🔮 Future Enhancements

- **Multi-CV Ranking**: Rank multiple candidates against a single JD
- **Skill Extraction**: Automatic skill tagging and categorization
- **ATS Compatibility Check**: Analyze CV compatibility with ATS systems
- **Export Reports**: Generate PDF/Word reports of analysis
- **Database Integration**: Store and track candidate analyses
- **Custom Model Support**: Allow users to choose different LLM models
- **Batch Processing**: Process multiple CVs simultaneously

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📄 License

This project is open source and available under the MIT License.

---

## 👤 Author

**aa055**

- GitHub: [@aa055](https://github.com/aa055)

---

## 🙏 Acknowledgments

- **Ollama** for local LLM infrastructure
- **LangChain** for LLM orchestration framework
- **Streamlit** for rapid web app development
