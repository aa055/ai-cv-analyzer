import streamlit as st
from agents.cv_parser_agent import CVParserAgent
from agents.jd_matcher_agent import JDMatcherAgent
from agents.feedback_agent import FeedbackAgent
from agents.summary_agent import SummaryAgent
import tempfile
import plotly.graph_objects as go
import plotly.express as px
import time
import json
import pandas as pd
from datetime import datetime

# Initialize agents
@st.cache_resource
def init_agents():
    """Initialize agents once and cache them"""
    return {
        'cv_parser': CVParserAgent(),
        'jd_matcher': JDMatcherAgent(),
        'feedback_agent': FeedbackAgent(),
        'summary_agent': SummaryAgent()
    }

# Page configuration
st.set_page_config(
    page_title="AI CV Analyzer Pro", 
    layout="wide",
    page_icon="🤖",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        background-color: #f0f2f6;
        border-left: 4px solid #1f77b4;
    }
    div[data-testid="metric-container"] {
        background-color: #f0f2f6;
        border: 1px solid #cccccc;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .uploadedFile {
        background-color: #e8f4f8;
        border-radius: 10px;
        padding: 10px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: #f0f2f6;
        border-radius: 5px 5px 0px 0px;
        color: black;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #5a9fd4;
        color: white;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: white !important;
    }
    /* Warning message styling */
    div[data-testid="stAlert"] p,
    div[role="alert"] p,
    .stAlert p {
        color: black !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'parsed_cv_cache' not in st.session_state:
    st.session_state.parsed_cv_cache = {}
if 'feedback_cache' not in st.session_state:
    st.session_state.feedback_cache = {}
if 'summary_cache' not in st.session_state:
    st.session_state.summary_cache = {}
if 'ats_cache' not in st.session_state:
    st.session_state.ats_cache = {}
if 'skills_cache' not in st.session_state:
    st.session_state.skills_cache = {}
if 'recruiter_match_result' not in st.session_state:
    st.session_state.recruiter_match_result = None
if 'recruiter_parsed' not in st.session_state:
    st.session_state.recruiter_parsed = None
if 'recruiter_cache_key' not in st.session_state:
    st.session_state.recruiter_cache_key = None
if 'recruiter_cv_name' not in st.session_state:
    st.session_state.recruiter_cv_name = None
# Multi-CV mode session state
if 'multi_cv_mode' not in st.session_state:
    st.session_state.multi_cv_mode = False
if 'multi_cv_candidates' not in st.session_state:
    st.session_state.multi_cv_candidates = []  # List of candidate data dicts
if 'multi_cv_jd' not in st.session_state:
    st.session_state.multi_cv_jd = None
if 'multi_cv_final_verdict' not in st.session_state:
    st.session_state.multi_cv_final_verdict = None

# Load agents
agents = init_agents()

# Header with better styling
st.markdown("""
<h1 style='text-align: center; color: #1f77b4;'>
    🤖 AI-Powered CV Analyzer Pro
</h1>
<p style='text-align: center; font-size: 18px; color: #666;'>
    Smart Resume Analysis & Job Matching Platform
</p>
<hr style='margin: 20px 0; border: 1px solid #e0e0e0;'>
""", unsafe_allow_html=True)

# Sidebar for configuration
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    st.markdown("#### Model Settings")
    model_choice = st.selectbox(
        "LLM Model",
        ["llama3.2", "llama2", "mistral", "codellama"],
        help="Select the Ollama model to use for analysis"
    )
    
    chunk_size = st.slider(
        "Text Chunk Size",
        min_value=200,
        max_value=1000,
        value=500,
        step=50,
        help="Size of text chunks for processing"
    )
    
    chunk_overlap = st.slider(
        "Chunk Overlap",
        min_value=0,
        max_value=200,
        value=50,
        step=10,
        help="Overlap between text chunks"
    )
    
    st.markdown("#### Display Settings")
    show_raw_scores = st.checkbox("Show Raw Similarity Scores", value=False)
    enable_caching = st.checkbox("Enable Result Caching", value=True)
    
    st.markdown("---")
    
    # Analysis history
    if st.session_state.analysis_history:
        st.markdown("### 📊 Recent Analyses")
        for idx, item in enumerate(st.session_state.analysis_history[-5:][::-1]):
            with st.expander(f"{item['type']} - {item['timestamp'][:19]}"):
                st.write(f"**File:** {item['filename']}")
                if item['type'] == 'Match Score':
                    st.write(f"**Score:** {item['score']:.2%}")

def create_score_gauge(score, title="Match Score"):
    """Create a gauge chart for match scores"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = score * 100,
        title = {'text': title},
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 40], 'color': "#ff4444"},
                {'range': [40, 70], 'color': "#ffaa00"},
                {'range': [70, 100], 'color': "#00ff88"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    fig.update_layout(height=250)
    return fig

def create_similarity_chart(scores, labels):
    """Create a bar chart for similarity scores"""
    colors = ['#ff4444' if s < 0.4 else '#ffaa00' if s < 0.7 else '#00ff88' for s in scores]
    
    fig = px.bar(
        x=scores, 
        y=labels, 
        orientation='h',
        title="Similarity Scores by Section",
        labels={'x': 'Similarity Score', 'y': 'CV Section'},
        color=scores,
        color_continuous_scale=["red", "yellow", "green"],
        range_color=[0, 1]
    )
    fig.update_layout(
        height=400,
        showlegend=False,
        xaxis_range=[0, 1],
        xaxis_tickformat='.0%'
    )
    return fig

def safe_parse_cv(pdf_path, cache_key=None):
    """Safely parse CV with error handling"""
    try:
        # Check cache first
        if cache_key and cache_key in st.session_state.parsed_cv_cache:
            return st.session_state.parsed_cv_cache[cache_key]
        
        with st.spinner("📄 Parsing CV..."):
            parsed = agents['cv_parser'].parse_cv(pdf_path)
            
            # Cache the result
            if cache_key and enable_caching:
                st.session_state.parsed_cv_cache[cache_key] = parsed
            
            return parsed
    except Exception as e:
        st.error(f"❌ Error parsing CV: {str(e)}")
        return None

def safe_get_feedback(cv_text, target_role, cache_key=None):
    """Safely get feedback with error handling"""
    try:
        # Check cache first
        if cache_key and cache_key in st.session_state.feedback_cache:
            return st.session_state.feedback_cache[cache_key]
        
        with st.spinner("🤔 Analyzing CV and generating suggestions..."):
            feedback = agents['feedback_agent'].suggest_improvements(cv_text, target_role)
            
            # Cache the result
            if cache_key and enable_caching:
                st.session_state.feedback_cache[cache_key] = feedback
            
            # Add to history
            st.session_state.analysis_history.append({
                'type': 'Feedback',
                'filename': 'CV Analysis',
                'timestamp': datetime.now().isoformat(),
                'role': target_role or 'General'
            })
            
            return feedback
    except Exception as e:
        st.error(f"❌ Error generating feedback: {str(e)}")
        st.info("💡 Make sure Ollama is running with the required model (llama3.2)")
        return None

def safe_match_jd(cv_chunks, job_description):
    """Safely match JD with error handling"""
    try:
        with st.spinner("🔍 Calculating match scores..."):
            result = agents['jd_matcher'].match(cv_chunks, job_description)
            return result
    except Exception as e:
        st.error(f"❌ Error matching CV with JD: {str(e)}")
        st.info("💡 Make sure Ollama is running with the embedding model (nomic-embed-text)")
        return None

def safe_generate_summary(cv_text, cache_key=None):
    """Safely generate summary with error handling"""
    try:
        # Check cache first
        if cache_key and cache_key in st.session_state.summary_cache:
            return st.session_state.summary_cache[cache_key]

        with st.spinner("📊 Generating comprehensive candidate summary..."):
            summary = agents['summary_agent'].generate_summary(cv_text)

            # Cache the result
            if cache_key and enable_caching:
                st.session_state.summary_cache[cache_key] = summary

            return summary
    except Exception as e:
        st.error(f"❌ Error generating summary: {str(e)}")
        return None

def safe_check_ats_score(cv_text, target_role, job_description, cache_key=None):
    """Safely check ATS score with error handling"""
    try:
        # Check cache first
        if cache_key and cache_key in st.session_state.ats_cache:
            return st.session_state.ats_cache[cache_key]

        with st.spinner("🔍 Analyzing ATS compatibility..."):
            ats_analysis = agents['feedback_agent'].check_ats_score(cv_text, target_role, job_description)

            # Cache the result
            if cache_key and enable_caching:
                st.session_state.ats_cache[cache_key] = ats_analysis

            # Add to history
            st.session_state.analysis_history.append({
                'type': 'ATS Score',
                'filename': 'CV Analysis',
                'timestamp': datetime.now().isoformat(),
                'role': target_role or 'General'
            })

            return ats_analysis
    except Exception as e:
        st.error(f"❌ Error checking ATS score: {str(e)}")
        st.info("💡 Make sure Ollama is running with the required model (llama3.2)")
        return None

def safe_analyze_skills(cv_text, target_role, job_description, cache_key=None):
    """Safely analyze skills with error handling"""
    try:
        # Check cache first
        if cache_key and cache_key in st.session_state.skills_cache:
            return st.session_state.skills_cache[cache_key]

        with st.spinner("📊 Analyzing skills..."):
            skills_analysis = agents['feedback_agent'].analyze_skills(cv_text, target_role, job_description)

            # Cache the result
            if cache_key and enable_caching:
                st.session_state.skills_cache[cache_key] = skills_analysis

            # Add to history
            st.session_state.analysis_history.append({
                'type': 'Skills Analysis',
                'filename': 'CV Analysis',
                'timestamp': datetime.now().isoformat(),
                'role': target_role or 'General'
            })

            return skills_analysis
    except Exception as e:
        st.error(f"❌ Error analyzing skills: {str(e)}")
        st.info("💡 Make sure Ollama is running with the required model (llama3.2)")
        return None

# Main tabs
tab1, tab2, tab3 = st.tabs(["🎯 Candidate Portal", "👔 Recruiter Dashboard", "📈 Analytics"])

# --- Candidate View ---
with tab1:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 📤 Upload Your CV")
        uploaded_cv = st.file_uploader(
            "Select your CV file (PDF format)",
            type=["pdf"],
            key="cv_upload",
            help="Upload your CV in PDF format for analysis"
        )

        st.markdown("### 📝 Job Description")
        job_description = st.text_area(
            "Paste the job description (Required for ATS Score & Skills Analysis)",
            height=150,
            placeholder="Paste the job description here to get more accurate ATS scoring and skills analysis...",
            help="Required for ATS Score and Skills Analysis. Optional for AI Feedback."
        )

    with col2:
        st.markdown("### 🎯 Target Role")
        target_role = st.text_input(
            "Enter your target position",
            placeholder="e.g., Senior Data Scientist",
            help="Specify the role you're applying for to get tailored feedback"
        )

        analysis_type = st.radio(
            "Analysis Type",
            ["Quick Review", "Detailed Analysis"],
            help="Quick Review provides key points, Detailed Analysis gives comprehensive feedback"
        )

    if uploaded_cv:
        # Create cache key
        cache_key = f"{uploaded_cv.name}_{uploaded_cv.size}"
        
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_cv.read())
            tmp_path = tmp_file.name

        # Parse CV
        parsed = safe_parse_cv(tmp_path, cache_key)
        
        if parsed:
            # Display CV preview in an expander
            with st.expander("📄 CV Content Preview", expanded=False):
                # Show first 1000 characters
                preview_text = parsed["text"][:1000]
                st.text(preview_text + "..." if len(parsed["text"]) > 1000 else preview_text)
                
                # Word count and other stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Word Count", len(parsed["text"].split()))
                with col2:
                    st.metric("Character Count", len(parsed["text"]))
                with col3:
                    st.metric("Chunks", len(parsed["chunks"]))

            # Action buttons
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🚀 Get AI Feedback", type="primary", use_container_width=True):
                    feedback_key = f"{cache_key}_{target_role}_{analysis_type}"
                    suggestions = safe_get_feedback(
                        parsed["text"],
                        target_role,
                        feedback_key if enable_caching else None
                    )
                    
                    if suggestions:
                        st.markdown("---")
                        st.markdown("### 💡 AI-Powered Improvement Suggestions")
                        
                        # Display feedback in a nice container
                        st.info(suggestions)
                        
                        # Download button for feedback
                        st.download_button(
                            label="📥 Download Feedback Report",
                            data=suggestions,
                            file_name=f"cv_feedback_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain"
                        )
            
            with col2:
                if st.button("🔍 Check ATS Score", use_container_width=True):
                    if not job_description or not job_description.strip():
                        st.warning("⚠️ Please provide a job description for ATS Score analysis. This is required to accurately match your CV against the job requirements.")
                    else:
                        ats_key = f"{cache_key}_{target_role}_{hash(job_description)}_ats"
                        ats_analysis = safe_check_ats_score(
                            parsed["text"],
                            target_role,
                            job_description,
                            ats_key if enable_caching else None
                        )

                        if ats_analysis:
                            st.markdown("---")
                            st.markdown("### 🔍 ATS Compatibility Analysis")

                            # Display ATS analysis in a nice container
                            st.info(ats_analysis)

                            # Download button for ATS analysis
                            st.download_button(
                                label="📥 Download ATS Report",
                                data=ats_analysis,
                                file_name=f"ats_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                mime="text/plain"
                            )

            with col3:
                if st.button("📊 Skills Analysis", use_container_width=True):
                    if not job_description or not job_description.strip():
                        st.warning("⚠️ Please provide a job description for Skills Analysis. This is required to accurately assess your skills against the job requirements.")
                    else:
                        skills_key = f"{cache_key}_{target_role}_{hash(job_description)}_skills"
                        skills_analysis = safe_analyze_skills(
                            parsed["text"],
                            target_role,
                            job_description,
                            skills_key if enable_caching else None
                        )

                        if skills_analysis:
                            st.markdown("---")
                            st.markdown("### 📊 Skills Analysis")

                            # Display skills analysis in a nice container
                            st.info(skills_analysis)

                            # Download button for skills analysis
                            st.download_button(
                                label="📥 Download Skills Report",
                                data=skills_analysis,
                                file_name=f"skills_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                mime="text/plain"
                            )

# --- Recruiter View ---
with tab2:
    # Header with Clear All button
    header_col1, header_col2 = st.columns([3, 1])
    with header_col1:
        st.markdown("### 📋 Job Matching Dashboard")
    with header_col2:
        if st.button("🗑️ Clear All", key="clear_all_recruiter", use_container_width=True):
            st.session_state.recruiter_match_result = None
            st.session_state.recruiter_parsed = None
            st.session_state.recruiter_cache_key = None
            st.session_state.recruiter_cv_name = None
            st.session_state.multi_cv_candidates = []
            st.session_state.multi_cv_jd = None
            st.session_state.multi_cv_final_verdict = None
            st.rerun()

    # Toggle for Single/Multiple CV mode
    cv_mode = st.toggle("Multiple CVs Mode", value=st.session_state.multi_cv_mode,
                        help="Toggle to switch between single CV analysis and multiple candidates ranking")
    st.session_state.multi_cv_mode = cv_mode

    if not cv_mode:
        # ============ SINGLE CV MODE ============
        col1, col2 = st.columns([1, 1])

        with col1:
            jd_input = st.text_area(
                "📝 Job Description",
                height=200,
                placeholder="Paste the job description here...",
                help="Enter the complete job description for matching",
                key="single_cv_jd"
            )

        with col2:
            uploaded_cv = st.file_uploader(
                "📄 Candidate CV (PDF)",
                type=["pdf"],
                key="recruiter_cv",
                help="Upload the candidate's CV for matching"
            )

            if uploaded_cv:
                st.success(f"✅ File uploaded: {uploaded_cv.name}")

        # Add button to trigger match analysis
        if uploaded_cv and jd_input:
            if st.button("🎯 Generate Match Analysis", type="primary", use_container_width=True):
                # Create cache key
                cache_key = f"{uploaded_cv.name}_{uploaded_cv.size}"

                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_cv.read())
                    tmp_path = tmp_file.name

                parsed = safe_parse_cv(tmp_path, cache_key)

                if parsed:
                    # Store parsed data in session state for later use
                    st.session_state.recruiter_parsed = parsed
                    st.session_state.recruiter_cache_key = cache_key
                    st.session_state.recruiter_jd = jd_input

                    # Match scores
                    result = safe_match_jd(parsed["chunks"], jd_input)

                    if result:
                        st.session_state.recruiter_match_result = result
                        st.session_state.recruiter_cv_name = uploaded_cv.name

        # Display results if available
        if 'recruiter_match_result' in st.session_state and st.session_state.recruiter_match_result:
            result = st.session_state.recruiter_match_result
            parsed = st.session_state.recruiter_parsed
            cache_key = st.session_state.recruiter_cache_key

            st.markdown("---")

            # Display match scores with visualizations
            st.markdown("### 📊 Match Analysis")

            col1, col2, col3 = st.columns([1, 1, 2])

            with col1:
                # Max score gauge
                fig_max = create_score_gauge(result['max_score'], "Max Match Score")
                st.plotly_chart(fig_max, use_container_width=True)

            with col2:
                # Average score gauge
                fig_avg = create_score_gauge(result['avg_score'], "Avg Match Score")
                st.plotly_chart(fig_avg, use_container_width=True)

            with col3:
                # Match interpretation
                st.markdown("#### 🎯 Match Interpretation")

                if result['max_score'] >= 0.8:
                    st.success("**Excellent Match!** Strong alignment with job requirements.")
                elif result['max_score'] >= 0.6:
                    st.warning("**Good Match.** Moderate alignment, some gaps to address.")
                else:
                    st.error("**Poor Match.** Significant gaps in requirements.")

                # Detailed metrics
                st.metric("Confidence Score", f"{(result['max_score'] + result['avg_score'])/2:.1%}")
                st.metric("Consistency", f"{result['avg_score']/result['max_score']:.1%}" if result['max_score'] > 0 else "0%")

            # Show detailed scores if enabled
            if show_raw_scores:
                with st.expander("📊 Detailed Chunk Scores"):
                    # Create labels for chunks
                    labels = [f"Chunk {i+1}" for i in range(len(result['similarity_scores']))]

                    # Create and display chart
                    fig_details = create_similarity_chart(result['similarity_scores'], labels)
                    st.plotly_chart(fig_details, use_container_width=True)

                    # Show top matching chunks
                    st.markdown("#### Top Matching Sections")
                    sorted_scores = sorted(
                        enumerate(result['similarity_scores']),
                        key=lambda x: x[1],
                        reverse=True
                    )[:3]

                    for idx, score in sorted_scores:
                        st.write(f"**Chunk {idx+1}** - Score: {score:.2%}")
                        st.text(parsed["chunks"][idx][:200] + "...")

            # Action buttons
            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("📝 Generate Full Report", use_container_width=True):
                    summary_key = f"{cache_key}_summary"
                    summary = safe_generate_summary(
                        parsed["text"],
                        summary_key if enable_caching else None
                    )

                    if summary:
                        st.markdown("---")
                        st.markdown("### 📄 Comprehensive Candidate Report")

                        # Display summary in a nice container
                        with st.container():
                            st.markdown(summary)

                        # Export options
                        col1, col2 = st.columns(2)
                        with col1:
                            # Download as text
                            report_data = f"""
CANDIDATE EVALUATION REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
================================

MATCH SCORES:
- Maximum Score: {result['max_score']:.2%}
- Average Score: {result['avg_score']:.2%}

DETAILED ANALYSIS:
{summary}
"""
                            st.download_button(
                                label="📥 Download Report (TXT)",
                                data=report_data,
                                file_name=f"candidate_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                mime="text/plain"
                            )

                        with col2:
                            # Download as JSON
                            cv_name = st.session_state.get('recruiter_cv_name', 'unknown')
                            json_data = json.dumps({
                                'timestamp': datetime.now().isoformat(),
                                'candidate_file': cv_name,
                                'scores': {
                                    'max': result['max_score'],
                                    'avg': result['avg_score'],
                                    'all': result['similarity_scores']
                                },
                                'summary': summary
                            }, indent=2)

                            st.download_button(
                                label="📥 Download Report (JSON)",
                                data=json_data,
                                file_name=f"candidate_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json"
                            )

            with col2:
                if st.button("🔄 Compare Candidates", use_container_width=True):
                    st.session_state.multi_cv_mode = True
                    st.rerun()

            with col3:
                if st.button("📧 Schedule Interview", use_container_width=True):
                    st.info("Interview scheduling feature coming soon!")

    else:
        # ============ MULTIPLE CVs MODE ============
        st.markdown("---")
        st.markdown("#### 🏆 Multi-Candidate Ranking")
        st.info("Upload multiple CVs to rank candidates against a single job description. The best candidate will be highlighted.")

        # Job Description input
        multi_jd_input = st.text_area(
            "📝 Job Description",
            height=150,
            placeholder="Paste the job description here...",
            help="Enter the complete job description for matching all candidates",
            key="multi_cv_jd_input",
            value=st.session_state.multi_cv_jd if st.session_state.multi_cv_jd else ""
        )

        # Store JD in session state
        if multi_jd_input:
            st.session_state.multi_cv_jd = multi_jd_input

        # CV Upload section with plus button
        upload_col1, upload_col2 = st.columns([4, 1])

        with upload_col1:
            uploaded_cvs = st.file_uploader(
                "📄 Upload Candidate CVs (PDF)",
                type=["pdf"],
                key="multi_cv_upload",
                accept_multiple_files=True,
                help="Upload multiple candidate CVs for ranking"
            )

        with upload_col2:
            st.markdown("<br>", unsafe_allow_html=True)
            add_more = st.button("➕ Add More", key="add_more_cv", use_container_width=True,
                                help="Click to add more CVs to the existing list")

        # Process uploaded CVs
        if uploaded_cvs and multi_jd_input:
            if st.button("🎯 Analyze All Candidates", type="primary", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()

                new_candidates = []
                total_files = len(uploaded_cvs)

                for i, cv_file in enumerate(uploaded_cvs):
                    status_text.text(f"Processing {cv_file.name}... ({i+1}/{total_files})")
                    progress_bar.progress((i + 1) / total_files)

                    # Check if already processed
                    existing_names = [c.get('file_name') for c in st.session_state.multi_cv_candidates]
                    if cv_file.name in existing_names and not add_more:
                        continue

                    # Create cache key
                    cache_key = f"{cv_file.name}_{cv_file.size}"

                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(cv_file.read())
                        tmp_path = tmp_file.name

                    # Parse CV
                    parsed = safe_parse_cv(tmp_path, cache_key)

                    if parsed:
                        # Match scores
                        result = safe_match_jd(parsed["chunks"], multi_jd_input)

                        if result:
                            # Extract structured info
                            struct_info = parsed.get("structured_info", {})

                            # Calculate overall rating (weighted average)
                            overall_rating = (result['max_score'] * 0.6 + result['avg_score'] * 0.4)

                            # Get candidate name with fallback to filename
                            candidate_name = struct_info.get('name')
                            if not candidate_name:
                                # Try to extract name from filename (e.g., "John_Doe_Resume.pdf")
                                filename_base = cv_file.name.rsplit('.', 1)[0]  # Remove extension
                                # Clean up common suffixes
                                for suffix in ['_resume', '_cv', '_Resume', '_CV', '-resume', '-cv', ' resume', ' cv']:
                                    filename_base = filename_base.replace(suffix, '')
                                # Replace underscores/dashes with spaces and title case
                                filename_name = filename_base.replace('_', ' ').replace('-', ' ').strip()
                                if filename_name and len(filename_name) > 2:
                                    candidate_name = filename_name.title()
                                else:
                                    candidate_name = '-'

                            candidate_data = {
                                'file_name': cv_file.name,
                                'name': candidate_name,
                                'email': struct_info.get('email') or '-',
                                'linkedin': struct_info.get('linkedin') or '-',
                                'github': struct_info.get('github') or '-',
                                'experience_years': struct_info.get('experience_years') or '-',
                                'max_score': result['max_score'],
                                'avg_score': result['avg_score'],
                                'overall_rating': overall_rating,
                                'parsed_data': parsed,
                                'match_result': result
                            }
                            new_candidates.append(candidate_data)

                # Add to existing candidates or replace
                if add_more:
                    st.session_state.multi_cv_candidates.extend(new_candidates)
                else:
                    st.session_state.multi_cv_candidates = new_candidates

                # Clear the final verdict when new candidates are added
                st.session_state.multi_cv_final_verdict = None

                status_text.text("✅ Analysis complete!")
                time.sleep(1)
                st.rerun()

        # Display candidates table if available
        if st.session_state.multi_cv_candidates:
            st.markdown("---")

            # Table header with download button
            table_header_col1, table_header_col2 = st.columns([3, 1])
            with table_header_col1:
                st.markdown("### 📊 Candidate Rankings")
            with table_header_col2:
                # PDF Download will be handled after table display
                pass

            # Sort candidates by overall rating
            sorted_candidates = sorted(
                st.session_state.multi_cv_candidates,
                key=lambda x: x['overall_rating'],
                reverse=True
            )

            # Find the best candidate
            best_rating = max(c['overall_rating'] for c in sorted_candidates)

            # Build table data
            table_data = []
            for idx, candidate in enumerate(sorted_candidates, 1):
                # Convert experience_years to string to avoid mixed type issues
                exp_years = candidate['experience_years']
                exp_str = str(exp_years) if exp_years and exp_years != '-' else '-'

                row = {
                    '#': idx,
                    'Candidate Name': candidate['name'],
                    'Email': candidate['email'],
                    'LinkedIn': candidate['linkedin'],
                    'GitHub': candidate['github'],
                    'Experience (Yrs)': exp_str,
                    'Max Score': f"{candidate['max_score']:.1%}",
                    'Avg Score': f"{candidate['avg_score']:.1%}",
                    'Overall Rating': f"{candidate['overall_rating']:.1%}"
                }
                table_data.append(row)

            df = pd.DataFrame(table_data)

            # Custom CSS for highlighting best candidate
            def highlight_best(row):
                # Check if this row has the best overall rating
                rating_str = row['Overall Rating'].rstrip('%')
                try:
                    rating = float(rating_str) / 100
                    if abs(rating - best_rating) < 0.001:  # Best candidate
                        return ['background-color: #5a9c5c'] * len(row)  # green
                except:
                    pass
                return [''] * len(row)

            styled_df = df.style.apply(highlight_best, axis=1)

            # Display the table
            st.dataframe(styled_df, width='stretch', hide_index=True)

            # Add more CVs button (alternative placement)
            st.markdown("")

            # Action buttons row
            action_col1, action_col2, action_col3 = st.columns([1, 1, 1])

            with action_col1:
                # Generate Final Verdict button
                if st.button("🏆 Generate Final Verdict", type="primary", use_container_width=True):
                    with st.spinner("🤔 Analyzing candidates and generating verdict..."):
                        # Build summary of all candidates for LLM
                        candidates_summary = []
                        for idx, candidate in enumerate(sorted_candidates, 1):
                            summary = f"""
Candidate {idx}: {candidate['name']}
- Experience: {candidate['experience_years']} years
- Max Match Score: {candidate['max_score']:.1%}
- Average Match Score: {candidate['avg_score']:.1%}
- Overall Rating: {candidate['overall_rating']:.1%}
- Email: {candidate['email']}
- LinkedIn: {candidate['linkedin']}
- GitHub: {candidate['github']}
"""
                            candidates_summary.append(summary)

                        all_candidates_text = "\n".join(candidates_summary)

                        verdict_prompt = f"""Based on the job description and candidate analysis below, provide a final hiring recommendation.

JOB DESCRIPTION:
{st.session_state.multi_cv_jd}

CANDIDATES ANALYSIS:
{all_candidates_text}

Please provide:
1. Your top recommendation for the position
2. Brief justification for your choice
3. Any notable strengths and concerns for the top candidate
4. Runner-up candidate (if applicable)
"""
                        try:
                            verdict = agents['summary_agent'].generate_summary(verdict_prompt)
                            st.session_state.multi_cv_final_verdict = verdict
                        except Exception as e:
                            st.error(f"Error generating verdict: {str(e)}")

            with action_col2:
                # Download PDF Report
                if st.button("📥 Download Report (PDF)", use_container_width=True):
                    # Generate comprehensive report content
                    report_content = f"""
MULTI-CANDIDATE EVALUATION REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*60}

JOB DESCRIPTION:
{'-'*40}
{st.session_state.multi_cv_jd}

{'='*60}
CANDIDATE RANKINGS:
{'='*60}
"""
                    for idx, candidate in enumerate(sorted_candidates, 1):
                        is_best = "⭐ BEST CANDIDATE" if abs(candidate['overall_rating'] - best_rating) < 0.001 else ""
                        report_content += f"""
{'-'*40}
Rank #{idx} {is_best}
{'-'*40}
Name: {candidate['name']}
Email: {candidate['email']}
LinkedIn: {candidate['linkedin']}
GitHub: {candidate['github']}
Experience: {candidate['experience_years']} years

MATCH SCORES:
- Maximum Score: {candidate['max_score']:.1%}
- Average Score: {candidate['avg_score']:.1%}
- Overall Rating: {candidate['overall_rating']:.1%}
"""

                    if st.session_state.multi_cv_final_verdict:
                        report_content += f"""
{'='*60}
FINAL VERDICT:
{'='*60}
{st.session_state.multi_cv_final_verdict}
"""

                    st.download_button(
                        label="📥 Confirm Download",
                        data=report_content,
                        file_name=f"multi_candidate_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        key="download_multi_report"
                    )

            with action_col3:
                # Remove selected / Clear candidates
                if st.button("🗑️ Clear Candidates", use_container_width=True):
                    st.session_state.multi_cv_candidates = []
                    st.session_state.multi_cv_final_verdict = None
                    st.rerun()

            # Display Final Verdict if available
            if st.session_state.multi_cv_final_verdict:
                st.markdown("---")
                st.markdown("### 🏆 Final Verdict")
                # Display verdict with proper markdown rendering
                with st.container(border=True):
                    st.markdown(st.session_state.multi_cv_final_verdict)

# --- Analytics Tab ---
with tab3:
    st.markdown("### 📈 Analytics Dashboard")
    
    if st.session_state.analysis_history:
        # Create metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Analyses", len(st.session_state.analysis_history))
        
        with col2:
            feedback_count = len([x for x in st.session_state.analysis_history if x['type'] == 'Feedback'])
            st.metric("CV Reviews", feedback_count)
        
        with col3:
            match_count = len([x for x in st.session_state.analysis_history if x['type'] == 'Match Score'])
            st.metric("Match Analyses", match_count)
        
        with col4:
            if match_count > 0:
                avg_score = sum([x['score'] for x in st.session_state.analysis_history if x['type'] == 'Match Score']) / match_count
                st.metric("Avg Match Score", f"{avg_score:.1%}")
        
        st.markdown("---")
        
        # Analysis timeline
        st.markdown("#### 📅 Analysis Timeline")
        
        # Convert to dataframe for visualization
        if len(st.session_state.analysis_history) > 0:
            df = pd.DataFrame(st.session_state.analysis_history)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Create timeline chart
            fig = px.scatter(
                df,
                x='timestamp',
                y='type',
                color='type',
                title="Analysis Activity Over Time",
                hover_data=['filename'] if 'filename' in df.columns else None
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Recent activity table
            st.markdown("#### 📊 Recent Activity")
            st.dataframe(
                df[['timestamp', 'type', 'filename']].tail(10).sort_values('timestamp', ascending=False),
                width='stretch',
                hide_index=True
            )
        
        # Clear history button
        if st.button("🗑️ Clear Analysis History"):
            st.session_state.analysis_history = []
            st.session_state.parsed_cv_cache = {}
            st.session_state.feedback_cache = {}
            st.session_state.summary_cache = {}
            st.session_state.ats_cache = {}
            st.session_state.skills_cache = {}
            st.session_state.recruiter_match_result = None
            st.session_state.recruiter_parsed = None
            st.session_state.recruiter_cache_key = None
            st.session_state.recruiter_cv_name = None
            st.session_state.multi_cv_candidates = []
            st.session_state.multi_cv_jd = None
            st.session_state.multi_cv_final_verdict = None
            st.success("✅ History cleared!")
            st.rerun()
    else:
        st.info("📊 No analysis data available yet. Start by analyzing some CVs!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; padding: 20px;'>
    <p>Powered by Ollama & LangChain | Made using Streamlit</p>
    <p style='font-size: 12px;'>Ensure Ollama is running with llama3.2 and nomic-embed-text models</p>
</div>
""", unsafe_allow_html=True)