from langchain_community.llms import Ollama

class SummaryAgent:
    def __init__(self):
        self.llm = Ollama(model="llama3.2")

    def generate_summary(self, cv_text):
        prompt = f"""
        Summarize the following candidate resume in terms of:
        - Skills
        - Experience summary
        - Notable highlights
        - Potential red flags

        Resume:
        {cv_text}
        """
        return self.llm.invoke(prompt)