from langchain_community.llms import Ollama

class FeedbackAgent:
    def __init__(self, model_name: str = "llama3.2"):
        self.model_name = model_name
        self.llm = Ollama(model=model_name)
    
    def update_model(self, model_name: str):
        """Update the LLM model being used"""
        self.model_name = model_name
        self.llm = Ollama(model=model_name)

    def suggest_improvements(self, raw_cv_text, target_role=None):
        """Provide general CV improvement suggestions (excludes ATS and Skills analysis)"""
        role_context = f"for a {target_role} position" if target_role else "for general job applications"

        prompt = f"""You are an expert CV/Resume coach with 15+ years of experience helping candidates optimize their resumes for human recruiters.

Analyze the following CV {role_context} and provide actionable improvement suggestions.

ANALYSIS STRUCTURE:
================================================================================

1. CONTENT QUALITY (Score: X/10)
   • Impact Statements: Transform responsibilities into achievements
   • Quantification: Add metrics, percentages, and numbers where missing
   • Action Verbs: Suggest stronger action verbs for bullet points
   • Relevance: Identify content that should be removed or emphasized

2. PROFESSIONAL SUMMARY
   • Current Assessment: Brief evaluation
   • Suggested Rewrite: Provide a compelling 2-3 line summary

3. EXPERIENCE SECTION
   • Bullet Point Improvements: Rewrite top 3 weak bullets with STAR method
   • Missing Achievements: What accomplishments should be highlighted?
   • Gap Analysis: Any employment gaps that need addressing?

4. TOP 5 PRIORITY ACTIONS
   1. [Most critical improvement]
   2. [Second priority]
   3. [Third priority]
   4. [Fourth priority]
   5. [Fifth priority]

5. INDUSTRY-SPECIFIC RECOMMENDATIONS
   • Certifications to Add: Relevant for {target_role or 'career growth'}
   • Keywords to Include: Industry-specific terms currently missing
   • Sections to Add: (e.g., Publications, Projects, Volunteer Work)

================================================================================

CV CONTENT TO ANALYZE:
{raw_cv_text[:3000]}{'...[truncated]' if len(raw_cv_text) > 3000 else ''}

Please provide specific, actionable feedback following the structure above. Use clear formatting with bullet points and sections."""

        return self.llm.invoke(prompt)

    def check_ats_score(self, raw_cv_text, target_role=None, job_description=None):
        """Analyze CV for ATS (Applicant Tracking System) optimization"""
        role_context = f"for a {target_role} position" if target_role else "for general job applications"

        jd_section = ""
        if job_description:
            jd_section = f"""
JOB DESCRIPTION TO MATCH AGAINST:
{job_description[:2000]}{'...[truncated]' if len(job_description) > 2000 else ''}

"""

        prompt = f"""You are an expert ATS (Applicant Tracking System) specialist with 15+ years of experience helping candidates optimize their resumes to pass automated screening systems.

Analyze the following CV {role_context} and provide a comprehensive ATS analysis.
{jd_section}
ATS ANALYSIS STRUCTURE:
================================================================================

1. OVERALL ATS SCORE: X/100
   Provide an overall ATS compatibility score based on the analysis below.
   {"Compare the CV against the provided job description for keyword matching." if job_description else ""}

2. KEYWORD ANALYSIS
   • Present Keywords: List relevant keywords found in the CV
   • Missing Critical Keywords: {"Keywords from the job description that are missing in the CV" if job_description else "Identify industry-standard keywords that are absent"}
   • Keyword Density: Assess if keywords are appropriately distributed
   • Keyword Match Rate: {"Percentage of job description keywords found in CV" if job_description else f"Specific keywords to add for {target_role or 'better ATS performance'}"}

3. FORMAT COMPATIBILITY (Score: X/10)
   • File Structure: Assess if the CV structure is ATS-friendly
   • Section Headers: Check if standard section names are used (Experience, Education, Skills, etc.)
   • Bullet Points: Verify proper formatting of bullet points
   • Potential Issues: Identify ATS-unfriendly elements (tables, images, text boxes, headers/footers, columns)

4. CONTENT STRUCTURE (Score: X/10)
   • Section Ordering: Is the order optimal for ATS scanning?
   • Contact Information: Is it properly formatted and easily parseable?
   • Date Formats: Are dates consistent and in a standard format?
   • Job Titles: Are they clear and standard (not creative titles)?

5. PARSING ISSUES
   • Potential parsing problems the ATS might encounter
   • Special characters or formatting that could cause issues
   • Missing sections that ATS typically looks for

6. {"JOB DESCRIPTION ALIGNMENT" if job_description else "INDUSTRY ALIGNMENT"}
   • {"Required qualifications from JD that are missing" if job_description else "Common industry requirements that should be added"}
   • {"Experience requirements match" if job_description else "Standard experience formatting"}
   • {"Skills alignment with JD requirements" if job_description else "Recommended skills to highlight"}

7. TOP 5 ATS OPTIMIZATION ACTIONS
   1. [Most critical ATS fix]
   2. [Second priority]
   3. [Third priority]
   4. [Fourth priority]
   5. [Fifth priority]

8. ATS-OPTIMIZED SUGGESTIONS
   • Specific rewrites for better ATS parsing
   • {"Phrases from job description to incorporate" if job_description else "Alternative phrasings that include more keywords"}
   • Section restructuring recommendations

================================================================================

CV CONTENT TO ANALYZE:
{raw_cv_text[:3000]}{'...[truncated]' if len(raw_cv_text) > 3000 else ''}

Please provide specific, actionable ATS optimization feedback. Be precise about what changes will improve ATS compatibility{"and alignment with the job description" if job_description else ""}."""

        return self.llm.invoke(prompt)

    def analyze_skills(self, raw_cv_text, target_role=None, job_description=None):
        """Analyze and provide detailed feedback on the skills section of the CV"""
        role_context = f"for a {target_role} position" if target_role else "for the modern job market"

        jd_section = ""
        if job_description:
            jd_section = f"""
JOB DESCRIPTION TO MATCH AGAINST:
{job_description[:2000]}{'...[truncated]' if len(job_description) > 2000 else ''}

"""

        prompt = f"""You are an expert career coach and skills analyst with 15+ years of experience helping candidates optimize their skills presentation {role_context}.

Analyze the skills in the following CV and provide comprehensive feedback.
{jd_section}
SKILLS ANALYSIS STRUCTURE:
================================================================================

1. OVERALL SKILLS ASSESSMENT (Score: X/10)
   Provide an overall assessment of the candidate's skills presentation.
   {"Include a skills match percentage against the job description requirements." if job_description else ""}

2. CURRENT SKILLS INVENTORY
   • Technical Skills Found: List all technical/hard skills identified
   • Soft Skills Found: List all soft/interpersonal skills identified
   • Tools & Technologies: List all tools, software, and technologies mentioned
   • Certifications & Credentials: List any certifications found

3. {"JOB DESCRIPTION SKILLS MATCH" if job_description else "SKILLS GAP ANALYSIS"}
   • {"Required Skills from JD - Present: Skills mentioned in JD that are in the CV" if job_description else f"Missing Critical Skills: Essential skills for {target_role or 'career advancement'} not present"}
   • {"Required Skills from JD - Missing: Skills mentioned in JD that are NOT in the CV" if job_description else "Emerging Skills to Add: In-demand skills in the current job market"}
   • {"Preferred/Nice-to-have Skills: Additional JD skills that would strengthen the application" if job_description else "Skills to Emphasize: Underrepresented skills that should be highlighted more"}
   • {"Skills Match Rate: X% of required skills present" if job_description else ""}

4. SKILLS TO REMOVE OR DOWNPLAY
   • Outdated Skills: Technologies or skills that are no longer relevant
   • Overused/Generic Skills: Common skills that don't add value (e.g., "Microsoft Office")
   • {"Irrelevant Skills: Skills not aligned with the job description" if job_description else f"Irrelevant Skills: Skills that don't align with {target_role or 'career goals'}"}

5. SKILLS ORGANIZATION RECOMMENDATIONS
   • Suggested Categories: How to group skills effectively
   • Priority Order: {"Order skills to match JD priority" if job_description else "Which skills should appear first"}
   • Proficiency Levels: Should skill levels be indicated? How?

6. SKILLS PRESENTATION IMPROVEMENTS
   • Current Presentation Issues: Problems with how skills are displayed
   • Better Formatting: Suggestions for visual presentation
   • {"JD Keywords to Add: Exact phrases from the job description to include" if job_description else "Context Addition: Where to add context/examples for key skills"}

7. TOP 5 SKILLS ACTIONS
   1. [Most important skills improvement{"to match JD" if job_description else ""}]
   2. [Second priority]
   3. [Third priority]
   4. [Fourth priority]
   5. [Fifth priority]

8. RECOMMENDED SKILLS SECTION REWRITE
   Provide a suggested rewrite of the skills section with:
   • Proper categorization
   • {"Skills aligned with the job description requirements" if job_description else f"Relevant skills for {target_role or 'career growth'}"}
   • Professional formatting
   • {"Keywords from the job description incorporated" if job_description else ""}

================================================================================

CV CONTENT TO ANALYZE:
{raw_cv_text[:3000]}{'...[truncated]' if len(raw_cv_text) > 3000 else ''}

Please provide specific, actionable skills-related feedback. Focus on making the skills section compelling and {"aligned with the job description" if job_description else "relevant"}."""

        return self.llm.invoke(prompt)
