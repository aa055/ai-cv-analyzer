from langchain_community.llms import Ollama
class FeedbackAgent:
    def __init__(self):
        self.llm = Ollama(model="llama3.2")

    def suggest_improvements(self, raw_cv_text, target_role=None):
        prompt = f"""
        You are a resume coach. Analyze the following CV text and suggest specific improvements
        related to clarity, formatting, and tailoring it for a {target_role or 'generic'} role.

        CV:
        {raw_cv_text}
        """
        return self.llm.invoke(prompt)
