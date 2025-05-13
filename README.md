# RAG Advanced Knowledge Management System

<div align="center">
  
  <br/>
  <h3>A powerful document processing and question-answering system built with RAG technology</h3>
  <p>Upload documents, ask questions, and get accurate answers powered by LLMs and vector search</p>
</div>

![GitHub stars](https://img.shields.io/github/stars/Arshnoor-Singh-Sohi/RAG-Advanced-Knowledge-Management-System?style=social)
![GitHub forks](https://img.shields.io/github/forks/Arshnoor-Singh-Sohi/RAG-Advanced-Knowledge-Management-System?style=social)
![GitHub issues](https://img.shields.io/github/issues/Arshnoor-Singh-Sohi/RAG-Advanced-Knowledge-Management-System)
![GitHub license](https://img.shields.io/github/license/Arshnoor-Singh-Sohi/RAG-Advanced-Knowledge-Management-System)

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Technology Stack](#-technology-stack)
- [Setup & Installation](#-setup--installation)
- [Usage Guide](#-usage-guide)
- [Core Components](#-core-components)
  - [Document Processing](#document-processing)
  - [Vector Storage](#vector-storage)
  - [Cloud Storage](#cloud-storage)
  - [LLM Service](#llm-service)
- [API Reference](#-api-reference)
- [Future Enhancements](#-future-enhancements)
- [Contributing](#-contributing)
- [License](#-license)

## 🔍 Overview

The RAG Advanced Knowledge Management System is a sophisticated document processing and question-answering platform that leverages Retrieval-Augmented Generation (RAG) technology. This system allows users to upload documents, process them into semantic chunks, and ask questions against their knowledge base to receive accurate, contextually relevant answers.

Built with a modern tech stack including Flask, LangChain, OpenAI, and AWS S3, this application demonstrates how to implement an effective RAG pattern for knowledge management and retrieval. The system processes uploaded documents, stores them in a vector database for semantic search, and uses a large language model to generate accurate responses based on the retrieved document chunks.

## ✨ Key Features

- **Document Upload & Processing**: Support for PDF and TXT documents with automatic chunking
- **Vector-Based Retrieval**: Semantic search using OpenAI embeddings and ChromaDB
- **Cloud Storage Integration**: AWS S3 integration for secure document storage
- **Conversational Memory**: Maintains context across multiple questions
- **Responsive Web Interface**: Clean and intuitive UI for document upload and querying
- **Production-Ready Architecture**: Well-structured application with separation of concerns
- **Error Handling**: Robust error handling for API endpoints and services
- **Scalable Design**: Modular components that can scale with growing document collections

## 🏗️ System Architecture

The system follows a clean, modular architecture built around the RAG pattern:

```
┌─────────────────┐     ┌──────────────────┐     ┌───────────────┐
│                 │     │                  │     │               │
│   Web Interface ├────►│   Flask Backend  ├────►│  AWS S3 Storage│
│                 │     │                  │     │               │
└─────────────────┘     └──────────┬───────┘     └───────────────┘
                                   │
                                   ▼
                         ┌────────────────────┐
                         │                    │
                         │  Document Processor │
                         │                    │
                         └──────────┬─────────┘
                                    │
                                    ▼
┌──────────────────┐     ┌────────────────────┐     ┌───────────────────┐
│                  │     │                    │     │                   │
│  LLM (OpenAI)    │◄────┤   Vector Storage   │◄────┤  Text Chunking     │
│                  │     │    (ChromaDB)      │     │                   │
└──────────────────┘     └────────────────────┘     └───────────────────┘
```

## 🛠️ Technology Stack

| Component | Technologies |
|-----------|--------------|
| **Backend** | Flask, Python 3.11 |
| **Vector Database** | ChromaDB, LangChain |
| **LLM Integration** | OpenAI API, LangChain |
| **Document Processing** | LangChain (PyPDFLoader, TextLoader), Unstructured |
| **Cloud Storage** | AWS S3 (boto3) |
| **Frontend** | HTML, CSS, JavaScript |
| **Environment** | Conda, Python dotenv |
| **Dependencies** | tiktoken, werkzeug |

## 🚀 Setup & Installation

### Prerequisites

- Python 3.11
- Conda (recommended for environment management)
- OpenAI API key
- AWS account with S3 bucket

### Step 1: Clone the repository

```bash
git clone https://github.com/Arshnoor-Singh-Sohi/RAG-Advanced-Knowledge-Management-System.git
cd RAG-Advanced-Knowledge-Management-System
```

### Step 2: Create and activate Conda environment

```bash
conda create -n llmapp python=3.11 -y
conda activate llmapp
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Set up environment variables

Create a `.env` file in the root directory with the following variables:

```env
OPENAI_API_KEY=your_openai_api_key
AWS_ACCESS_KEY=your_aws_access_key
AWS_SECRET_KEY=your_aws_secret_key
AWS_BUCKET_NAME=your_s3_bucket_name
```

### Step 5: Run the application

```bash
python app/main.py
```

Access the application at `http://127.0.0.1:8080` in your web browser.

## 📖 Usage Guide

### Document Upload

1. Navigate to the home page
2. In the "Upload Documents" section, click "Browse" to select PDF or TXT files
3. Click "Upload" to process and store your documents
4. Wait for the confirmation message indicating successful processing

### Asking Questions

1. In the "Ask Questions" section, type your query in the text area
2. Click "Ask" to submit your question
3. View the AI-generated response in the "Response" section

### Tips for Effective Questions

- Be specific in your questions to get more accurate responses
- Refer to document content directly in your questions
- For complex topics, ask follow-up questions to explore further
- If responses lack detail, try reformulating your question

## 📦 Core Components

### Document Processing

The document processing component handles various file types and prepares them for storage and retrieval:

```python
def process_document(file):
    """Process document based on file type and return text chunks"""
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, file.filename)
    
    try:
        # Save file temporarily
        file.save(temp_path)
        
        # Process based on file type
        if file.filename.endswith('.pdf'):
            loader = PyPDFLoader(temp_path)
            documents = loader.load()
        elif file.filename.endswith('.txt'):
            loader = TextLoader(temp_path)
            documents = loader.load()
        else:
            raise ValueError("Unsupported file type")

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        text_chunks = text_splitter.split_documents(documents)
        
        return text_chunks
```

The system uses LangChain's document loaders to handle different file types and implements a recursive character text splitter for optimal chunk creation with 200-token overlaps to maintain context across chunks.

### Vector Storage

The vector storage component creates and manages embeddings for semantic search:

```python
class VectorStore:
    def __init__(self, path):
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = Chroma(
            persist_directory=path,
            embedding_function=self.embeddings
        )

    def add_documents(self, documents):
        self.vector_store.add_documents(documents)
        
    def similarity_search(self, query, k=4):
        return self.vector_store.similarity_search(query, k=k)
```

ChromaDB is utilized as the vector database, with document embeddings generated using OpenAI's embedding models. This enables semantic similarity search to find the most relevant document chunks for each query.

### Cloud Storage

The S3 storage service handles secure document storage in AWS:

```python
class S3Storage:
    def __init__(self):
        self.s3 = boto3.client(
            's3',
            aws_access_key_id=Config.AWS_ACCESS_KEY,
            aws_secret_access_key=Config.AWS_SECRET_KEY
        )
        self.bucket = Config.AWS_BUCKET_NAME

    def upload_file(self, file_obj, filename):
        try:
            self.s3.upload_fileobj(file_obj, self.bucket, filename)
            return True
        except ClientError as e:
            print(f"Error uploading file: {e}")
            return False
```

This component provides a secure, scalable solution for storing the original documents, allowing for potential future retrieval or reprocessing.

### LLM Service

The LLM service integrates with OpenAI's models through LangChain:

```python
class LLMService:
    def __init__(self, vector_store):
        self.llm = ChatOpenAI(
            temperature=0.7,
            model_name="gpt-3.5-turbo",
            openai_api_key=Config.OPENAI_API_KEY
        )
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=vector_store.vector_store.as_retriever(),
            memory=self.memory
        )

    def get_response(self, query):
        try:
            response = self.chain({"question": query})
            return response['answer']
        except Exception as e:
            print(f"Error getting LLM response: {e}")
            return "I encountered an error processing your request."
```

The service utilizes LangChain's ConversationalRetrievalChain to maintain context across multiple questions while retrieving relevant document chunks for each query.

## 📡 API Reference

### Document Upload Endpoint

```
POST /upload

Request:
- multipart/form-data with 'file' field

Response:
{
    "message": "File uploaded and processed successfully",
    "chunks_processed": <number_of_chunks>
}

Errors:
- 400: No file provided/No file selected/Unsupported file type
- 500: Error processing document/Error uploading to S3/Error adding to vector store
```

### Query Endpoint

```
POST /query

Request:
{
    "question": "Your question here"
}

Response:
{
    "response": "AI-generated answer based on your documents"
}

Errors:
- 400: No question provided
- 500: Error generating response
```

## 🔮 Future Enhancements

### Planned Features

1. **Multi-user Support**
   - User authentication and authorization
   - Personal document collections
   - Role-based access control

2. **Advanced Document Processing**
   - Support for more file formats (DOCX, XLSX, HTML, etc.)
   - Image and table extraction
   - Document metadata analysis

3. **Improved Search Capabilities**
   - Hybrid search (semantic + keyword)
   - Filters and faceted search
   - Search result highlighting

4. **UI Enhancements**
   - Document management dashboard
   - Interactive search results
   - Chat history visualization

5. **Deployment & Scaling**
   - Docker containerization
   - Kubernetes orchestration
   - Performance optimization for large document collections

## 👥 Contributing

Contributions to the RAG Advanced Knowledge Management System are welcome! Here's how you can contribute:

1. **Fork the repository**
   ```bash
   git clone https://github.com/Arshnoor-Singh-Sohi/RAG-Advanced-Knowledge-Management-System.git
   ```

2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Add or improve features
   - Fix bugs
   - Improve documentation

4. **Test your changes**
   - Ensure all features work correctly
   - Verify error handling
   - Check code quality

5. **Submit a pull request**
   - Provide a clear description of the changes
   - Reference any related issues

### Coding Standards

- Follow PEP 8 guidelines for Python code
- Include docstrings for new functions and classes
- Add appropriate error handling
- Write meaningful commit messages

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  <h3>Developed with ❤️ by Arshnoor Singh Sohi</h3>
  <p>For questions, feature requests, or bug reports, please open an issue on GitHub</p>
</div>
