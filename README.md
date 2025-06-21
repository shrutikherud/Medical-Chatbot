[![HTML5](https://img.shields.io/badge/html5-E34F26.svg?style=flat&logo=html5&logoColor=white)](https://developer.mozilla.org/en-US/docs/Web/Guide/HTML/HTML5)
[![CSS3](https://img.shields.io/badge/css3-1572B6.svg?style=flat&logo=css3&logoColor=white)](https://developer.mozilla.org/en-US/docs/Web/CSS)
[![Bootstrap](https://img.shields.io/badge/bootstrap-7952B3.svg?style=flat&logo=bootstrap&logoColor=white)](https://getbootstrap.com/)
[![JavaScript](https://img.shields.io/badge/javascript-F7DF1E.svg?style=flat&logo=javascript&logoColor=black)](https://developer.mozilla.org/en-US/docs/Web/JavaScript)
[![Python](https://img.shields.io/badge/python-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/flask-000000.svg?style=flat&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![LangChain](https://img.shields.io/badge/langchain-blue.svg?style=flat&logo=langchain&logoColor=white)](https://www.langchain.com/)
[![Cohere](https://img.shields.io/badge/cohere-1A1A1A.svg?style=flat&logo=cohere&logoColor=white)](https://cohere.com/)
[![Pinecone](https://img.shields.io/badge/pinecone-0093FF.svg?style=flat&logo=pinecone&logoColor=white)](https://www.pinecone.io/)
[![NLP](https://img.shields.io/badge/NLP-4B8BBE.svg?style=flat&logo=spacy&logoColor=white)](https://en.wikipedia.org/wiki/Natural_language_processing)


# 🩺 Medical Chatbot

An intelligent AI-powered chatbot designed to answer medical-related queries using Cohere LLM, Pinecone vector search, and LangChain in a Flask web app.

&nbsp;

## 🛠️ Setup Instructions

### 🔧 Requirements

- Python 3.10+
- Cohere API Key
- Pinecone API Key

&nbsp;

### 📦 Installation & Running

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/Medical-Chatbot.git
cd Medical-Chatbot

# 2. Create and activate a virtual environment
conda create -n mchatbot310 python=3.10 -y
conda activate mchatbot310

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set environment variables (create a .env file)
export COHERE_API_KEY=your_cohere_api_key
export PINECONE_API_KEY=your_pinecone_api_key

# 5. Embed and upload PDF data to Pinecone
python store_index.py

# 6. Start the Flask server
python app.py
```

&nbsp;
## 🧱 Tech Stack

| Layer             | Technology                                                                 |
|------------------|------------------------------------------------------------------------------|
| Frontend         | HTML5, CSS3, Bootstrap 5, JavaScript                                         |
| Web Framework    | Flask (Python)                                                               |
| Document Handling| LangChain (DirectoryLoader, PyPDFLoader)                                     |
| Embedding Models | Cohere, HuggingFace (MiniLM)                                                 |
| Vector Database  | Pinecone                                                                     |
| NLP Engine       | Cohere (via LangChain)                                                       |
| RAG Pipeline     | LangChain Retrieval-Augmented Generation                                     |
| Environment      | Python 3.10+, Conda, dotenv (.env)                                           |



&nbsp;

## 🧠 How It Works

1. Medical PDFs are loaded, split, and embedded using Cohere's embedding model.
2. Embeddings are stored in a Pinecone vector DB.
3. User queries are matched using similarity search.
4. LangChain RAG pipeline uses `command-r` Cohere model to generate answers.
5. Flask backend handles chat logic; frontend built with HTML, CSS, Bootstrap, and JavaScript.

&nbsp;
## ✨ Features

- Ask medical-related questions and get AI-powered responses.
- Uses LangChain and Cohere LLM for semantic understanding.
- Embeds and indexes medical PDF documents into Pinecone.
- Clean, responsive Bootstrap-based UI.

&nbsp;
