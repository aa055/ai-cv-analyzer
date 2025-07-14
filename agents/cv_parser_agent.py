from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings

class CVParserAgent:
    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    def parse_cv(self, pdf_path):
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        text = " ".join([p.page_content for p in pages])
        chunks = self.splitter.split_text(text)
        return {"text": text, "chunks": chunks}
