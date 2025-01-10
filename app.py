from flask import Flask, request, jsonify, render_template, session, send_file
from flask_cors import CORS
import os
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
import json
import threading
import time
import fitz  # PyMuPDF
import base64
import logging
from pathlib import Path
import tempfile
import shutil
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup
import re
import html
import backoff
import itertools
from time import sleep
from random import uniform
from requests.exceptions import RequestException
from duckduckgo_search import DDGS


# Document Processing
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.docstore.document import Document

# Initialize Flask app with proper CORS configuration
app = Flask(__name__)
CORS(app,
     resources={r"/*": {
         "origins": "*",
         "methods": ["GET", "POST", "OPTIONS"],
         "allow_headers": ["Content-Type"],
         "supports_credentials": True
     }})
app.secret_key = os.urandom(24)
load_dotenv()


# Configuration
class Config:
    UPLOAD_FOLDER = Path('uploads')
    TEMP_FOLDER = Path('temp')
    ALLOWED_EXTENSIONS = {'pdf', 'docx'}
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    GROQ_API_KEY = os.getenv('GROQ_API_KEY')


app.config.from_object(Config)

# Ensure directories exist
Config.UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
Config.TEMP_FOLDER.mkdir(parents=True, exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

from duckduckgo_search import DDGS
import itertools
import logging


class RateLimitedSearcher:
    def __init__(self, max_retries=3, base_delay=2):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.session = requests.Session()
        self.ddgs = DDGS()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1'
        })

    def search(self, query):
        """Perform web search with rate limiting and error handling"""
        results = []
        try:
            # Get raw search results
            search_generator = self.ddgs.text(
                keywords=query,
                region='wt-wt',
                safesearch='moderate'
            )

            # Convert generator to list and take first 5 results
            raw_results = list(itertools.islice(search_generator, 5))

            for result in raw_results:
                try:
                    # Debug log the raw result
                    logger.debug(f"Raw result: {result}")

                    # Extract data using the correct field names
                    url = result.get('href') or result.get('url')
                    if not url:
                        logger.warning("No URL found in result")
                        continue

                    title = result.get('title', '')
                    description = result.get('snippet', '') or result.get('description', '')

                    # Clean and format the data
                    title = html.unescape(str(title)).strip()
                    title = ' '.join(title.split())[:100]

                    description = html.unescape(str(description)).strip()
                    description = ' '.join(description.split())[:200]

                    results.append({
                        "url": url,
                        "title": title,
                        "summary": description
                    })

                except Exception as e:
                    logger.warning(f"Error processing search result: {str(e)}")
                    continue

        except Exception as e:
            logger.error(f"Search error: {str(e)}")

        # Debug log the final results
        logger.debug(f"Processed results: {results}")
        return results


class DocumentProcessor:
    def __init__(self):
        self.embedding_model = HuggingFaceEmbeddings()
        self.conversation_chain = None
        self.vectorstore = None
        self.current_document = None
        self.chat_history = []
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        self.is_document_loaded = False
        self.last_access = time.time()
        self.current_pdf_path = None
        self.page_mapping = {}
        self.doc_pages = None

    def _initialize_llm(self):
        """Initialize and return the LLM"""
        if not Config.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        return ChatGroq(
            model="llama-3.1-70b-versatile",
            temperature=0.3,
            max_tokens=1024,
            groq_api_key=Config.GROQ_API_KEY
        )

    def _get_text_coordinates(self, page_num, text):
        """Get coordinates for text in PDF page"""
        if not self.doc_pages or page_num >= len(self.doc_pages):
            return []

        page = self.doc_pages[page_num]
        text_instances = page.search_for(text)

        coordinates = []
        for inst in text_instances:
            coordinates.append({
                'x': inst.x0,
                'y': inst.y0,
                'width': inst.x1 - inst.x0,
                'height': inst.y1 - inst.y0
            })

        return coordinates

    def process_document(self, file_path):
        """Process uploaded document and prepare it for Q&A"""
        try:
            file_extension = file_path.suffix.lower()
            if file_extension == '.pdf':
                self.current_pdf_path = file_path
                doc = fitz.open(str(file_path))
                self.doc_pages = [page for page in doc]

                documents = []
                for page_num, page in enumerate(self.doc_pages):
                    text = page.get_text()
                    doc = Document(
                        page_content=text,
                        metadata={"page": page_num + 1}
                    )
                    documents.append(doc)
                    self.page_mapping[text] = page_num + 1

            elif file_extension in ['.docx', '.doc']:
                loader = Docx2txtLoader(str(file_path))
                documents = loader.load()
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")

            self.current_document = documents
            self.chat_history.clear()

            processed_data = self._process_documents(documents)
            self._setup_conversation_chain(documents)

            self.is_document_loaded = True
            self.last_access = time.time()

            if file_extension == '.pdf':
                with open(file_path, "rb") as f:
                    processed_data['pdf_base64'] = base64.b64encode(f.read()).decode()

            return processed_data

        except Exception as e:
            logger.error(f"Document processing error: {str(e)}")
            raise

    def _process_documents(self, documents):
        """Process documents and create summaries"""
        try:
            page_summaries = {}
            full_text = []

            for i, doc in enumerate(documents):
                full_text.append(doc.page_content)
                summary = self._create_summary(doc.page_content)
                if summary:
                    page_summaries[str(i + 1)] = summary

            full_summary = self._create_summary(" ".join(full_text)) if full_text else ""

            return {
                "full_summary": full_summary,
                "page_summaries": page_summaries,
                "status": "success"
            }

        except Exception as e:
            logger.error(f"Document processing error: {str(e)}")
            return {"status": "error", "error": str(e)}

    def _create_summary(self, text):
        """Create a summary of the given text"""
        try:
            sentences = text.split('.')
            summary_sentences = sentences[:3]
            summary = '. '.join(sentence.strip() for sentence in summary_sentences if sentence.strip())
            return summary + '.' if summary else ""
        except Exception as e:
            logger.error(f"Summary creation error: {str(e)}")
            return ""

    def _setup_conversation_chain(self, documents):
        """Setup the conversation chain for Q&A"""
        try:
            texts = self.text_splitter.split_documents(documents)
            self.vectorstore = Chroma.from_documents(documents=texts, embedding=self.embedding_model)

            llm = self._initialize_llm()
            self.conversation_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=self.vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                verbose=True
            )

        except Exception as e:
            logger.error(f"Conversation chain setup error: {str(e)}")
            raise

    def _find_source_page(self, text):
        """Find the page number for a given text"""
        for content, page in self.page_mapping.items():
            if text in content:
                return page
        return None

    def highlight_pdf(self, sources):
        """Highlight multiple sources in PDF and return base64 encoded string"""
        if not self.current_pdf_path or not self.current_pdf_path.exists():
            logger.error("No PDF document loaded")
            return None

        try:
            doc = fitz.open(str(self.current_pdf_path))
            temp_path = Config.TEMP_FOLDER / f"{time.time_ns()}_highlighted.pdf"

            for source in sources:
                page_num = source.get('page', 1) - 1
                text = source.get('text', '')
                if 0 <= page_num < doc.page_count:
                    page = doc[page_num]
                    text_instances = page.search_for(text)
                    for inst in text_instances:
                        highlight = page.add_highlight_annot(inst)
                        highlight.update()

            doc.save(str(temp_path))
            doc.close()

            with open(temp_path, "rb") as f:
                encoded_pdf = base64.b64encode(f.read()).decode()

            temp_path.unlink(missing_ok=True)
            return encoded_pdf

        except Exception as e:
            logger.error(f"Error highlighting PDF: {str(e)}")
            return None

    def ask_question(self, question):
        """Process a question and return answer with source information"""
        try:
            if not self.is_document_loaded or not self.conversation_chain:
                return {"error": "Please upload a document first"}

            self.last_access = time.time()

            response = self.conversation_chain({
                "question": question,
                "chat_history": self.chat_history
            })

            self.chat_history.append((question, response['answer']))

            sources = []
            if 'source_documents' in response and response['source_documents']:
                for doc in response['source_documents']:
                    source_text = doc.page_content
                    page_num = doc.metadata.get('page', self._find_source_page(source_text))

                    if page_num is not None:
                        source = {
                            'page': page_num,
                            'text': source_text,
                            'coordinates': self._get_text_coordinates(page_num - 1, source_text)
                        }
                        sources.append(source)

            highlighted_pdf = self.highlight_pdf(sources) if sources else None

            return {
                "answer": response['answer'],
                "sources": sources,
                "highlighted_pdf": highlighted_pdf,
                "status": "success"
            }

        except Exception as e:
            logger.error(f"Question processing error: {str(e)}")
            return {"status": "error", "error": str(e)}

    def web_search(self, query):
        """Perform web search using rate-limited searcher"""
        try:
            self.last_access = time.time()
            searcher = RateLimitedSearcher()
            return searcher.search(query)
        except Exception as e:
            logger.error(f"Web search error: {str(e)}")
            return []


# Create processors dictionary
processors = {}
processor_lock = threading.Lock()


def get_processor():
    """Get or create a processor for the current session"""
    if 'session_id' not in session:
        session['session_id'] = os.urandom(16).hex()

    session_id = session['session_id']

    with processor_lock:
        if session_id not in processors:
            processors[session_id] = DocumentProcessor()
        return processors[session_id]


def allowed_file(filename):
    """Check if file type is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS


@app.route('/hello-world')
def hello_world():
    return 'Hello, World!'


# Routes
@app.route('/')
def home():
    """Render the home page"""
    return render_template('index.html')





@app.route('/upload', methods=['POST', 'OPTIONS'])
def upload_file():
    if request.method == 'OPTIONS':
        response = app.make_default_options_response()
        return response

    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed'}), 400

        filename = secure_filename(file.filename)
        file_path = Config.UPLOAD_FOLDER / filename
        file.save(file_path)

        processor = get_processor()
        result = processor.process_document(file_path)

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/ask', methods=['POST'])
def ask():
    """Handle user questions"""
    try:
        data = request.json
        question = data.get('question', '')

        if not question:
            return jsonify({'error': 'No question provided'}), 400

        processor = get_processor()
        response = processor.ask_question(question)

        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error during question handling: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/search', methods=['POST'])
def web_search():
    """Handle web search requests"""
    try:
        data = request.json
        query = data.get('query', '').strip()

        if not query:
            return jsonify({'results': [], 'error': 'No query provided'}), 400

        processor = get_processor()
        results = processor.web_search(query)

        return jsonify({'results': results}), 200

    except Exception as e:
        logger.error(f"Error during web search: {str(e)}")
        return jsonify({'results': [], 'error': str(e)}), 500


@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    """Download file from server"""
    file_path = Config.UPLOAD_FOLDER / filename
    if file_path.exists():
        return send_file(file_path, as_attachment=True)
    else:
        return jsonify({'status': 'error', 'message': 'File not found'}), 404


# Cleanup function to remove old processors
def cleanup_old_processors():
    """Remove processors that haven't been accessed in the last hour"""
    while True:
        try:
            current_time = time.time()
            with processor_lock:
                for session_id, processor in list(processors.items()):
                    if current_time - processor.last_access > 3600:  # 1 hour
                        del processors[session_id]
            time.sleep(1800)  # Run cleanup every 30 minutes
        except Exception as e:
            logger.error(f"Error during processor cleanup: {str(e)}")
            time.sleep(60)  # Wait a minute before retrying if there's an error


# Start cleanup thread
cleanup_thread = threading.Thread(target=cleanup_old_processors, daemon=True)
cleanup_thread.start()

if __name__ == '__main__':
    # Ensure the upload and temp directories exist
    Config.UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
    Config.TEMP_FOLDER.mkdir(parents=True, exist_ok=True)

    # Start the Flask application
    app.run(debug=True, host='0.0.0.0', port=8080)
