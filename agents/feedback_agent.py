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
        role_context = f"for a {target_role} position" if target_role else "for general job applications"
        
        prompt = f"""You are an expert CV/Resume coach with 15+ years of experience helping candidates optimize their resumes for ATS systems and human recruiters.

Analyze the following CV {role_context} and provide actionable improvement suggestions.

ANALYSIS STRUCTURE:
================================================================================

1. ATS OPTIMIZATION (Score: X/10)
   • Keyword Analysis: Identify missing industry-standard keywords
   • Format Issues: Detect ATS-unfriendly elements (tables, images, headers/footers)
   • File Structure: Assess section ordering and naming conventions

2. CONTENT QUALITY (Score: X/10)
   • Impact Statements: Transform responsibilities into achievements
   • Quantification: Add metrics, percentages, and numbers where missing
   • Action Verbs: Suggest stronger action verbs for bullet points
   • Relevance: Identify content that should be removed or emphasized

3. PROFESSIONAL SUMMARY
   • Current Assessment: Brief evaluation
   • Suggested Rewrite: Provide a compelling 2-3 line summary

4. SKILLS SECTION
   • Missing Critical Skills: Based on {target_role or 'modern job market'}
   • Skills to Remove: Outdated or irrelevant skills
   • Suggested Organization: Technical, Soft Skills, Tools/Technologies

5. EXPERIENCE SECTION
   • Bullet Point Improvements: Rewrite top 3 weak bullets with STAR method
   • Missing Achievements: What accomplishments should be highlighted?
   • Gap Analysis: Any employment gaps that need addressing?

6. TOP 5 PRIORITY ACTIONS
   1. [Most critical improvement]
   2. [Second priority]
   3. [Third priority]
   4. [Fourth priority]
   5. [Fifth priority]

7. INDUSTRY-SPECIFIC RECOMMENDATIONS
   • Certifications to Add: Relevant for {target_role or 'career growth'}
   • Keywords to Include: Industry-specific terms currently missing
   • Sections to Add: (e.g., Publications, Projects, Volunteer Work)

================================================================================

CV CONTENT TO ANALYZE:
{raw_cv_text[:3000]}{'...[truncated]' if len(raw_cv_text) > 3000 else ''}

Please provide specific, actionable feedback following the structure above. Use clear formatting with bullet points and sections."""
        
        return self.llm.invoke(prompt)
