import streamlit as st
from agents.cv_parser_agent import CVParserAgent
from agents.jd_matcher_agent import JDMatcherAgent
from agents.feedback_agent import FeedbackAgent
from agents.summary_agent import SummaryAgent
import tempfile

# Initialize agents
cv_parser = CVParserAgent()
jd_matcher = JDMatcherAgent()
feedback_agent = FeedbackAgent()
summary_agent = SummaryAgent()

st.set_page_config(page_title="AI Recruiter Assistant", layout="wide")
st.title("🤖 AI-Powered Job Recruiting Assistant")

# Tabs for Candidate and Recruiter
tab1, tab2 = st.tabs(["🎯 Candidate View", "📄 Recruiter View"])

# --- Candidate View ---
with tab1:
    st.header("Upload Your CV")
    uploaded_cv = st.file_uploader("Upload your CV as PDF", type=["pdf"], key="cv_upload")
    target_role = st.text_input("Target Job Role (Optional)", placeholder="e.g., Data Scientist")

    if uploaded_cv:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_cv.read())
            tmp_path = tmp_file.name

        parsed = cv_parser.parse_cv(tmp_path)
        st.subheader("🔍 CV Analysis")
        # st.write(parsed["text"][:1000] + "...")

        if st.button("✍️ Get Improvement Suggestions"):
            suggestions = feedback_agent.suggest_improvements(parsed["text"], target_role)
            st.subheader("📌 Suggestions")
            st.write(suggestions)

# --- Recruiter View ---
with tab2:
    st.header("Match Candidates with Job Description")
    jd_input = st.text_area("Paste the Job Description")
    uploaded_cv = st.file_uploader("Upload Candidate CV (PDF)", type=["pdf"], key="recruiter_cv")

    if jd_input and uploaded_cv:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_cv.read())
            tmp_path = tmp_file.name

        parsed = cv_parser.parse_cv(tmp_path)

        st.subheader("📈 Match Score")
        result = jd_matcher.match(parsed["chunks"], jd_input)
        st.metric("Max Score", f"{result['max_score']:.2f}")
        st.metric("Avg Score", f"{result['avg_score']:.2f}")

        if st.button("📝 Generate Summary"):
            summary = summary_agent.generate_summary(parsed["text"])
            st.subheader("📄 Candidate Summary")
            st.write(summary)
