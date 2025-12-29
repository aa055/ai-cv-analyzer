from langchain_community.llms import Ollama

class SummaryAgent:
    def __init__(self, model_name: str = "llama3.2"):
        self.model_name = model_name
        self.llm = Ollama(model=model_name)
    
    def update_model(self, model_name: str):
        """Update the LLM model being used"""
        self.model_name = model_name
        self.llm = Ollama(model=model_name)

    def generate_summary(self, cv_text):
        prompt = f"""You are an expert technical recruiter with 10+ years of experience evaluating candidates across various industries.

    Create a comprehensive candidate evaluation report based on the resume below. Be objective, thorough, and provide actionable insights.

    CANDIDATE EVALUATION REPORT
    ================================================================================

    📊 CANDIDATE OVERVIEW
    ━━━━━━━━━━━━━━━━━━━━
    • Name & Title: [Extract from CV]
    • Years of Experience: [Calculate total]
    • Current/Last Role: [Most recent position]
    • Education Level: [Highest degree]
    • Location: [If mentioned]

    🎯 CORE COMPETENCIES
    ━━━━━━━━━━━━━━━━━━━━
    Technical Skills:
    • [Group by proficiency: Expert/Advanced/Intermediate]

    Soft Skills:
    • [Leadership, Communication, Problem-solving, etc.]

    Domain Expertise:
    • [Industry-specific knowledge areas]

    🏆 KEY ACHIEVEMENTS & IMPACT
    ━━━━━━━━━━━━━━━━━━━━━━━━━━
    1. [Most impressive achievement with metrics]
    2. [Second notable accomplishment]
    3. [Third significant contribution]

    💼 PROFESSIONAL EXPERIENCE ANALYSIS
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Career Progression:
    • [Linear/Non-linear, promotions, lateral moves]
    • Average Tenure: [Years per role]
    • Industries Worked: [List sectors]

    Role Complexity Evolution:
    • [How responsibilities have grown over time]

    🎓 EDUCATION & CERTIFICATIONS
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━
    • Degrees: [List with institutions and years]
    • Relevant Certifications: [Current and expired]
    • Continuous Learning: [Recent courses, training]

    💪 STRENGTHS (Top 3)
    ━━━━━━━━━━━━━━━━━━━
    1. [Primary strength with evidence]
    2. [Secondary strength with evidence]  
    3. [Tertiary strength with evidence]

    ⚠️ AREAS OF CONCERN / GAPS
    ━━━━━━━━━━━━━━━━━━━━━━━━━
    • [Red flag 1: e.g., frequent job changes, employment gaps]
    • [Red flag 2: e.g., lack of progression, overqualification]
    • [Skills gaps for senior roles]

    🎲 CULTURAL FIT INDICATORS
    ━━━━━━━━━━━━━━━━━━━━━━━━
    • Team Collaboration: [Evidence of teamwork]
    • Leadership Style: [If applicable]
    • Work Environment Preference: [Startup/Corporate/Remote]

    📈 MARKET POSITIONING
    ━━━━━━━━━━━━━━━━━━━━
    • Seniority Level: [Entry/Mid/Senior/Executive]
    • Salary Expectations: [Based on experience/location if possible]
    • Competitive Advantage: [What makes them unique]

    🎯 BEST FIT ROLES
    ━━━━━━━━━━━━━━━━
    1. [Most suitable role type]
    2. [Alternative role fit]
    3. [Stretch role possibility]

    📋 INTERVIEW FOCUS AREAS
    ━━━━━━━━━━━━━━━━━━━━━━
    • Technical Deep Dive: [Areas to probe]
    • Behavioral Questions: [Specific scenarios to explore]
    • Red Flag Clarification: [Points needing explanation]

    ⭐ OVERALL RATING: [X/10]
    ━━━━━━━━━━━━━━━━━━━━━━━
    Brief Justification: [2-3 sentences on overall assessment]

    🔍 RECRUITER NOTES
    ━━━━━━━━━━━━━━━━━
    • Quick Win: [Is this candidate ready to move?]
    • Negotiation Points: [Potential concerns or leverage]
    • Reference Check Focus: [What to verify]

    ================================================================================

    RESUME CONTENT:
    {cv_text[:3000]}{'...[truncated]' if len(cv_text) > 3000 else ''}

    Generate a thorough evaluation following the structure above. Be specific and provide evidence from the resume for your assessments."""
        
        return self.llm.invoke(prompt)