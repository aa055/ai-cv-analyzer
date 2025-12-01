from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class CVParserAgent:
    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    def parse_cv(self, pdf_path):
        reader = PdfReader(pdf_path)
        text = " ".join([page.extract_text() for page in reader.pages])
        chunks = self.splitter.split_text(text)
        return {"text": text, "chunks": chunks}
