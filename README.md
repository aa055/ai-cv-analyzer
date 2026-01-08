# AI-Powered CV Analyzer & Job Recruiting Assistant

An intelligent recruitment assistant that leverages AI and NLP to analyze CVs, match candidates with job descriptions, and provide actionable feedback for both candidates and recruiters.

## Table of Contents

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

## Overview

This application is a dual-purpose recruiting tool that serves both **job candidates** and **recruiters**:

- **For Candidates**: Upload your CV to receive personalized improvement suggestions tailored to your target role
- **For Recruiters**: Match candidate CVs against job descriptions using semantic similarity scoring and generate comprehensive candidate summaries

The system uses local LLM models via Ollama for privacy-focused, offline AI processing.

---

## Features

### Candidate Portal
- **CV Upload & Parsing**: Extract text from PDF resumes with intelligent chunking
- **AI-Powered Feedback**: Get comprehensive improvement suggestions tailored to your target role
- **ATS Score Analysis**: Check your CV's compatibility with Applicant Tracking Systems
  - Keyword analysis and match rate
  - Format compatibility scoring
  - Parsing issue identification
  - Top 5 ATS optimization actions
- **Skills Analysis**: Detailed skills assessment including:
  - Technical and soft skills inventory
  - Skills gap analysis against job requirements
  - Skills organization recommendations
  - Suggested skills section rewrite
- **Downloadable Reports**: Export feedback, ATS analysis, and skills reports as TXT files

### Recruiter Dashboard
- **Single CV Mode**:
  - Job description matching with semantic similarity scoring
  - Visual match score gauges (max and average scores)
  - Match interpretation and confidence scoring
  - Detailed chunk-by-chunk similarity analysis
  - Comprehensive candidate report generation
- **Multi-CV Mode** (Candidate Ranking):
  - Upload multiple CVs to rank against a single job description
  - Automatic candidate ranking table with:
    - Extracted candidate info (name, email, LinkedIn, GitHub)
    - Years of experience
    - Max, average, and overall rating scores
  - Best candidate highlighting
  - AI-generated final verdict with hiring recommendation
  - Downloadable multi-candidate reports (TXT, JSON)

### Analytics Dashboard
- **Analysis Tracking**: View total analyses, CV reviews, and match analyses count
- **Performance Metrics**: Track average match scores across candidates
- **Activity Timeline**: Visualize analysis activity over time
- **Recent Activity Table**: Quick access to recent analyses
- **Cache Management**: Clear history and cached results

---

## Architecture

The application follows a **multi-agent architecture** where specialized agents handle specific tasks:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Streamlit Dashboard                              │
│                         (dashboard.py)                                   │
├─────────────────────┬──────────────────────┬────────────────────────────┤
│   Candidate Portal  │  Recruiter Dashboard │   Analytics Dashboard      │
└─────────┬───────────┴──────────┬───────────┴─────────────┬──────────────┘
          │                      │                         │
          │                      │                         └─► Session State
          │                      │                             • Analysis history
          │                      │                             • Cache management
          │                      │
          ├──────────────────────┼─────► CVParserAgent
          │                      │       • PDF text extraction
          │                      │       • Semantic chunking by CV sections
          │                      │       • Structured info extraction
          │                      │         (name, email, LinkedIn, GitHub, experience)
          │                      │
          │                      ├─────► JDMatcherAgent
          │                      │       • Embedding generation (nomic-embed-text)
          │                      │       • Cosine similarity scoring
          │                      │       • Multi-candidate ranking
          │                      │
          ├──────────────────────┼─────► FeedbackAgent
          │                      │       • CV improvement suggestions
          │                      │       • ATS score analysis
          │                      │       • Skills gap analysis
          │                      │
          └──────────────────────┴─────► SummaryAgent
                                        • Candidate summarization
                                        • Final verdict generation
                                        • Multi-candidate comparison
```

---

## Workflow

### Candidate Workflow

1. **Upload CV**: User uploads a PDF resume through the Streamlit interface
2. **CV Parsing**: `CVParserAgent` extracts text with semantic chunking
3. **Preview**: Parsed content displayed with word/character count and chunk stats
4. **Choose Analysis Type**:
   - **AI Feedback**: Click "Get AI Feedback" for improvement suggestions
   - **ATS Score**: Click "Check ATS Score" (requires job description) for ATS compatibility analysis
   - **Skills Analysis**: Click "Skills Analysis" (requires job description) for detailed skills assessment
5. **Review Results**: Analysis displayed with downloadable reports

### Recruiter Workflow (Single CV Mode)

1. **Input JD**: Paste the job description
2. **Upload CV**: Upload the candidate's PDF resume
3. **Generate Match**: Click "Generate Match Analysis"
4. **View Results**:
   - Visual gauge charts for max/avg match scores
   - Match interpretation (Excellent/Good/Poor)
   - Confidence and consistency metrics
5. **Generate Report** (optional): Create comprehensive candidate evaluation
6. **Export**: Download reports in TXT or JSON format

### Recruiter Workflow (Multi-CV Mode)

1. **Enable Multi-CV Mode**: Toggle "Multiple CVs Mode"
2. **Input JD**: Paste the job description for matching all candidates
3. **Upload CVs**: Upload multiple candidate PDFs
4. **Analyze**: Click "Analyze All Candidates"
5. **Review Rankings**: View candidate ranking table with:
   - Extracted contact info (name, email, LinkedIn, GitHub)
   - Experience years and match scores
   - Best candidate highlighted in green
6. **Generate Verdict**: Click "Generate Final Verdict" for AI hiring recommendation
7. **Export**: Download comprehensive multi-candidate report

---

## Installation

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

## Usage

### For Candidates

1. Navigate to the **"Candidate Portal"** tab
2. Upload your CV in PDF format
3. Enter your target job role (e.g., "Senior Data Scientist")
4. Paste the job description (required for ATS Score and Skills Analysis)
5. Choose your analysis:
   - **Get AI Feedback**: General CV improvement suggestions
   - **Check ATS Score**: ATS compatibility analysis (requires JD)
   - **Skills Analysis**: Detailed skills assessment (requires JD)
6. Download reports for offline review

### For Recruiters (Single Candidate)

1. Navigate to the **"Recruiter Dashboard"** tab
2. Ensure "Multiple CVs Mode" is OFF
3. Paste the job description
4. Upload the candidate's CV (PDF)
5. Click **"Generate Match Analysis"**
6. Review match scores with visual gauges
7. Click **"Generate Full Report"** for comprehensive evaluation
8. Export as TXT or JSON

### For Recruiters (Multiple Candidates)

1. Navigate to the **"Recruiter Dashboard"** tab
2. Toggle **"Multiple CVs Mode"** ON
3. Paste the job description
4. Upload multiple candidate CVs
5. Click **"Analyze All Candidates"**
6. Review the ranking table (best candidate highlighted)
7. Click **"Generate Final Verdict"** for AI hiring recommendation
8. Download the multi-candidate report

---

## Agent Details

### 1. CVParserAgent (`cv_parser_agent.py`)

**Purpose**: Extract, preprocess, and structure CV content from PDF files

**Key Components**:
- `PdfReader`: Loads and extracts text from PDF documents
- `RecursiveCharacterTextSplitter`: Configurable text chunking (default: 500 chars, 50 overlap)

**Methods**:
- `parse_cv(pdf_path, use_semantic_chunking)`: Parse CV with optional semantic chunking
- `extract_structured_info(text)`: Extract contact info, skills, experience years
- `extract_name_from_text(text, email)`: Multi-strategy name extraction
- `create_semantic_chunks(text)`: Section-aware chunking (Summary, Experience, Education, Skills, Projects)
- `update_chunk_settings(chunk_size, chunk_overlap)`: Dynamic chunk configuration

**Output Structure**:
```python
{
    "text": "full CV text...",
    "chunks": ["[EXPERIENCE]\nchunk1...", "[SKILLS]\nchunk2...", ...],
    "structured_info": {
        "name": "John Doe",
        "email": "john@example.com",
        "phone": "+1-234-567-8900",
        "linkedin": "linkedin.com/in/johndoe",
        "github": "github.com/johndoe",
        "skills": ["Python", "React", "AWS", ...],
        "experience_years": 5
    },
    "num_pages": 2,
    "chunk_method": "semantic"
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

**Purpose**: Provide comprehensive CV analysis including improvements, ATS optimization, and skills assessment

**Key Components**:
- `OllamaLLM`: LLaMA 3.2 model for text generation (from `langchain-ollama`)

**Methods**:
- `suggest_improvements(raw_cv_text, target_role)`: General CV improvement suggestions
- `check_ats_score(raw_cv_text, target_role, job_description)`: ATS compatibility analysis
- `analyze_skills(raw_cv_text, target_role, job_description)`: Detailed skills assessment
- `update_model(model_name)`: Switch LLM model dynamically

**Analysis Categories**:

*AI Feedback*:
- Content quality scoring
- Professional summary suggestions
- Experience section improvements
- Top 5 priority actions
- Industry-specific recommendations

*ATS Score Analysis*:
- Overall ATS score (X/100)
- Keyword analysis and match rate
- Format compatibility scoring
- Content structure assessment
- Parsing issue identification
- Job description alignment

*Skills Analysis*:
- Skills inventory (technical, soft, tools)
- Skills gap analysis against job requirements
- Outdated skills identification
- Skills organization recommendations
- Suggested skills section rewrite

---

### 4. SummaryAgent (`summary_agent.py`)

**Purpose**: Generate comprehensive candidate evaluations and hiring recommendations

**Key Components**:
- `OllamaLLM`: LLaMA 3.2 model for text generation (from `langchain-ollama`)

**Methods**:
- `generate_summary(cv_text)`: Returns structured candidate evaluation
- `update_model(model_name)`: Switch LLM model dynamically

**Evaluation Report Includes**:
- Candidate Overview (name, title, experience, education, location)
- Core Competencies (technical skills, soft skills, domain expertise)
- Key Achievements & Impact (top 3 with metrics)
- Professional Experience Analysis (career progression, tenure, industries)
- Education & Certifications
- Strengths (top 3 with evidence)
- Areas of Concern / Gaps (red flags, skills gaps)
- Cultural Fit Indicators
- Market Positioning (seniority, salary expectations, competitive advantage)
- Best Fit Roles (primary, alternative, stretch)
- Interview Focus Areas
- Overall Rating (X/10)
- Recruiter Notes (quick win, negotiation points, reference check focus)

---

## Technology Stack

### Core Framework
- **Streamlit**: Web-based dashboard interface with session state management

### AI/ML Libraries
- **LangChain**: Orchestration framework for LLM applications
  - `langchain-core`: Core abstractions
  - `langchain-ollama`: Ollama LLM integration
  - `langchain-community`: Community integrations (embeddings)
  - `langchain_text_splitters`: Text chunking utilities

### NLP & Embeddings
- **Ollama**: Local LLM inference (privacy-focused, offline processing)
  - Models: `llama3.2` (text generation), `nomic-embed-text` (embeddings)

### Data Processing & Visualization
- **pandas**: Data manipulation and table handling
- **Plotly**: Interactive visualizations (gauge charts, bar charts, scatter plots)
- **scikit-learn**: Cosine similarity calculations
- **pypdf**: PDF text extraction

---

## Project Structure

```
ai-cv-analyzer/
│
├── dashboard.py              # Main Streamlit application (3 tabs: Candidate, Recruiter, Analytics)
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
│
└── agents/                   # Agent modules
    ├── __init__.py           # Agent exports
    ├── cv_parser_agent.py    # PDF parsing, semantic chunking, structured info extraction
    ├── jd_matcher_agent.py   # CV-JD similarity matching with embeddings
    ├── feedback_agent.py     # AI feedback, ATS analysis, skills assessment
    └── summary_agent.py      # Candidate evaluation, hiring recommendations
```

