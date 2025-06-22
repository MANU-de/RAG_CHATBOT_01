# Project Structure and Documentation

This project implements a RAG (Retrieval Augmented Generation) chatbot.

## Core Technologies 🛠️

* **LangChain:** For the core pipeline.
* **Gemini API:** For the LLM and embeddings (requires an API key).
* **FAISS:** For local vector storage (lightweight and fast).
* **Python's `input()`:** For a basic Command Line Interface (CLI).
* **Python's `logging`:** For basic logging.
* **`ConversationBufferMemory`:** For session-based memory (optional extension).

---

## Project Structure 📂
```txt
rag_chatbot_project/

├── .env                # To store API keys

├── requirements.txt    # Python dependencies

├── documents/          # Folder for your custom documents
│   ├── project_alpha.txt
│   └── company_info.txt

├── main.py             # Main application script

└── utils/
└── ingest.py       # Script/module for document ingestion
```

---

## Setup and Installation 🚀

### 1. `requirements.txt`

Create a `requirements.txt` file in the `rag_chatbot_project/` directory with the following content:

```txt
langchain
langchain-google-genai
faiss-cpu
python-dotenv
langchainhub
```
Install these 
`pip install -r requirements.txt`

**2. .env File**<br/>
Create a .env file in the rag_chatbot_project/ directory to store your API key:

 ```
 GOOGLE_API_KEY="your api_key_here"
 [Replace "your api_key_here" with your actual Google API key.]
 ```
**3. documents/ Folder and Sample Documents**<br/>
Create a folder named documents inside your rag_chatbot_project/ directory. <br/> You can add your custom .txt files here.

For example:

documents/project_alpha.txt<br/>
documents/company_info.txt<br/>
(You need to look at the sample code provided in the GitHub repository.)

## Scripts 📜

### utils/ingest.py<br/>
(You need to look at the sample code provided in the GitHub repository.)

This script handles:

Loading documents from the documents/ folder.<br/>
Splitting documents into manageable chunks.<br/>
Creating embeddings for these chunks using the Gemini API.<br/>
Storing the embeddings in a FAISS vector store.<br/>
(You need to look at the sample code provided in the GitHub repository.)

Run this script once to create your vector store:
```
Bash

python utils/ingest.py
```
This will create a vectorstore_db folder containing the FAISS index.

### main.py
This is the main chatbot application script that loads the vector store and the LLM to answer questions
based on the ingested documents.

(You need to look at the sample code provided in the GitHub repository.)

### How to Run 🏃‍♀️

1. **Set up API Key:** Ensure your .env file in rag_chatbot_project/ has your _GOOGLE_API_KEY_.
2. **Ingest Documents:**<br/>
      - Navigate to the rag_chatbot_project directory in your terminal.
      - Run: python utils/ingest.py
      - This will create a vectorstore_db folder containing the FAISS index.
3. **Run the Chatbot:**
      - In the same terminal, run: python main.py
      - You can now ask questions! 🎉


### Sample Queries to Test ❓

1. **Specific information from** ```project_alpha.txt:```
      - You: What is Project Alpha?
      - You: What programming language is used for Project Alpha?
      - You: Who is the project lead for Alpha?
        
2. **Specific information from** ```company_info.txt:```
      - You: When was Innovatech Solutions founded?
      - You: What products does Innovatech Solutions offer?
      - You: Who is the CEO of Innovatech?
   
3. **Information that might require combining or inference (RAG helps ground this):**
      - You: Tell me about Innovatech's work with AI. (Should primarily pull from company_info.txt)

4. **Questions outside the documents' scope:**
      - You: What is the weather like today? (The bot should ideally say it doesn't know or can't answer based on the provided context).
      - You: Who is the president of the United States?

   
### Explanation and Key Concepts 🧠

1. **Document Ingestion** (utils/ingest.py)
      - DirectoryLoader & TextLoader: Load .txt files from the documents folder.
      - RecursiveCharacterTextSplitter: Breaks down large documents into smaller, manageable chunks.<br/> This is crucial because LLMs have context window limits, and embeddings work best on smaller, semantically coherent text segments.
      - GoogleGenerativeAIEmbeddings (or similar from langchain-google-genai): Converts text chunks into numerical vectors (embeddings) that capture their semantic meaning.
      - FAISS: A library for efficient similarity search on these vectors.<br/>
        FAISS.from_documents() creates the vector store and <br/>
          save_local() persists it.<br/>
          load_local() retrieves it.

2. **RAG Pipeline** (main.py)
      - **Load LLM:** ChatGoogleGenerativeAI (or similar) initializes the language model.
      - **Load Vector Store:** The pre-built FAISS index is loaded.
      - **Retriever:** vectorstore.as_retriever() creates an object that can find documents in the vector store similar to a given query.<br/> search_kwargs={"k": 3} means it will retrieve the top 3 most relevant chunks.
      - **Prompt Template:** This is _critical_. It instructs the LLM on how to behave.
         + {context}: Placeholder where LangChain will insert the relevant document chunks retrieved from FAISS.
         + {question}: Placeholder for the user's question.
         + The instructions ("Use only the following pieces of context...", "If you don't know...") are key to "grounding" the LLM and preventing hallucination.
      - **RetrievalQA Chain:** This is a standard LangChain chain that:
         1. Takes the user's question.
         2. Uses the retriever to find relevant document chunks (context).
         3. Inserts the context and question into the prompt_template.
         4. Sends the formatted prompt to the LLM.
         5. Returns the LLM's generated answer.
         + chain_type="stuff": This method "stuffs" all retrieved documents directly into the context.<br/> Good for a few documents, but can exceed token limits if many/large chunks are retrieved.<br/> Other types like "map_reduce", "refine" exist for handling more context.
         + return_source_documents=True: Allows you to see which document chunks were used to generate the answer, which is great for debugging and transparency.

3. **Basic UX (CLI):**
      - A simple while loop takes user input and calls chatbot.ask() (or a similar function in your main.py).
4. **Logging (logging module)**
      - Provides insights into what the application is doing (loading docs, initializing LLM, retrieved sources, etc.).<br/> This is very helpful for debugging.
5. **Session-Based Memory (ConversationBufferMemory - Optional Extension):**
      - In the Chatbot class (or your main application logic), use_memory=True (or similar implementation) attempts to enable this.
      - ConversationBufferMemory stores the history of the conversation.
      - **[!Important Note] _for RetrievalQA:_** RetrievalQA itself doesn't inherently use the memory object to feed past conversation<br/> _turns into the prompt context_ in a sophisticated way like ```ConversationalRetrievalChain``` does.
      - ConversationalRetrievalChain is specifically designed for this. It first takes the current question and the chat history,<br/> rephrases the question to be standalone (if needed), then performs retrieval,<br/> and finally answers the question based on retrieved docs and current question.
      - To make RetrievalQA truly conversational with a custom prompt, you would typically:
         1. Load chat history from memory.
         2. Format it along with the new question.
         3. Pass this combined string as the "query" to RetrievalQA.<br/>
         The example (if provided in the code) shows where you would save to memory, but true conversational flow with RetrievalQA and a specific prompt needs more careful construction of the input to the chain.<br/>
For a simpler path to conversational RAG,<br/> ```ConversationalRetrievalChain``` is often preferred, though it gives less direct control over the final QA prompt structure.


### Optional Extensions ✨

* Trying different vector stores (ChromaDB, Pinecone, etc.).
* Experimenting with different text splitters and chunking strategies.
* Refining the prompt for better responses.
* Implementing ```ConversationalRetrievalChain``` for more natural conversation flow.
* Building a more sophisticated UI (e.g., with Streamlit or Flask).
* Adding more advanced ReAct (Reasoning and Acting) or CoT (Chain-of-Thought) reasoning if needed for complex queries.



 
