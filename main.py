# main.py
import os
from dotenv import load_dotenv
import logging

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory # For optional session memory
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import faiss

from utils.ingest import load_vectorstore # Re-use the loading function

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables (for API key)
load_dotenv()

# --- Configuration ---
VECTORSTORE_PATH = "vectorstore_db"
GEMINI_MODEL_NAME = "gemini-2.0-flash"

# Set Google API key for gemini
os.environ["GOOGLE_API_KEY"] ="actual_api_key_here" # Replace with your actual API key

# --- Core RAG Components ---
class Chatbot:
    def __init__(self, use_memory=False):
        self.llm = None
        self.vectorstore = None
        self.qa_chain = None
        self.use_memory = use_memory
        self.memory = None
        self.embeddings = None
        self._initialize()

    def _initialize(self):
        logger.info(f"Initializing LLM: {GEMINI_MODEL_NAME}")
        self.llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL_NAME, temperature=1) # Temperature for creativity

        # Initialize embeddings
        logger.info("Initializing embeddings...")
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        # 2. Load Vector Store
        logger.info("Loading vector store...")
        self.vectorstore = load_vectorstore()
        if not self.vectorstore:
            logger.error("Failed to load vector store. Exiting.")
            raise ValueError("Vector store not found. Please run ingest.py first.")
        logger.info("Vector store loaded successfully.")

        # Move index to GPU (e.g., GPU 0)
        #index = self.vectorstore.index  # This is the raw FAISS index object
        #gpu_res = faiss.StandardGpuResources()
        #index_gpu = faiss.index_cpu_to_gpu(gpu_res, 0, index)
        #self.vectorstore.index = index_gpu

        #print(type(vectorstore.index))
        # <class 'faiss.swigfaiss_avx2.GpuIndexFlatIP'>  # Example output if on GPU

        # 3. Setup Retriever
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3}) # Retrieve top 3 relevant chunks

        # 4. Setup Prompt Template
        # A more robust prompt that encourages using only the provided context
        prompt_template = """
        You are an AI assistant for answering questions based on the provided documents.
        Use only the following pieces of context to answer the question at the end.
        If you don't know the answer from the context, just say that you don't know, don't try to make up an answer.
        If the question is not related to the context, politely respond that you are tuned to only answer questions related to the context.
        
        Context:
        {context}
        
        Question: {question}
        
        Helpful Answer:
        """
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )
        
        chain_type_kwargs = {"prompt": PROMPT}

        # 5. Setup QA Chain
        if self.use_memory:
            logger.info("Initializing QA chain with memory...")
            self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key='answer')
            # For ConversationalRetrievalChain, the prompt is handled differently
            # It typically has a `condense_question_prompt` and `combine_docs_chain_kwargs`
            # For simplicity with a custom prompt, we might need to adapt or use a more complex setup.
            # Let's stick to RetrievalQA for now and inject memory manually or use a simpler memory approach.
            # The easiest way with RetrievalQA and custom prompt is to manage history outside or pass it in context.
            # A more integrated approach is ConversationalRetrievalChain, but it has its own prompt structure.
            # For this example, let's use RetrievalQA and show how to *potentially* add memory.
            # Note: True session memory with RetrievalQA and a custom prompt needs more plumbing.
            # ConversationalRetrievalChain is better suited for this but has less control over the final QA prompt.

            # Using RetrievalQA as it's simpler for custom prompts, memory integration is more manual here.
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff", # "stuff" puts all context into the prompt
                retriever=retriever,
                chain_type_kwargs=chain_type_kwargs,
                return_source_documents=True, # To see which documents were retrieved
                # memory=self.memory # RetrievalQA doesn't directly use memory object in this way for history
            )
            logger.info("QA chain with memory placeholder initialized (manual history management needed for RetrievalQA).")
            logger.warning("Note: For full conversational memory with RetrievalQA and custom prompt, "
                           "you'd typically format chat history into the 'question' or 'context'. "
                           "Consider `ConversationalRetrievalChain` for built-in memory handling.")

        else:
            logger.info("Initializing QA chain without memory...")
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs=chain_type_kwargs,
                return_source_documents=True
            )
            logger.info("QA chain initialized.")

    def ask(self, question: str):
        if not self.qa_chain:
            logger.error("QA chain not initialized.")
            return "Error: Chatbot not ready."

        logger.info(f"Received question: {question}")
        
        # If using memory and RetrievalQA, you'd need to format the history + question
        # For example:
        # history_string = self.memory.load_memory_variables({})["chat_history"] (if messages)
        # combined_input = f"Chat history:\n{history_string}\n\nNew question: {question}"
        # For this basic example, we'll just pass the question directly.

        try:
            # LangChain's RetrievalQA expects a dictionary for input if you've used input_variables
            # in the prompt, or a string if it's a simple question.
            # The `query` key is standard for RetrievalQA.
            result = self.qa_chain.invoke({"query": question}) 
            answer = result["result"]
            source_documents = result.get("source_documents", [])

            logger.info(f"LLM Answer: {answer}")
            if source_documents:
                logger.info("Retrieved source documents:")
                for i, doc in enumerate(source_documents):
                    logger.info(f"  Source {i+1}: {doc.metadata.get('source', 'Unknown source')} (Excerpt: {doc.page_content[:100]}...)")
            
            # If using memory, save context
            if self.use_memory and self.memory:
                # RetrievalQA doesn't automatically use the memory for chat history in the prompt itself.
                # It's more for chains like ConversationalRetrievalChain.
                # Here, we're just demonstrating saving to memory.
                # For actual conversational context, the prompt or chain needs to be designed to use it.
                self.memory.save_context({"question": question}, {"answer": answer})
                # logger.info(f"Current memory: {self.memory.load_memory_variables({})}")


            return answer, source_documents
        except Exception as e:
            logger.error(f"Error during QA chain execution: {e}")
            return f"An error occurred: {e}", []

# --- Basic CLI UX ---
def run_cli():
    if not os.environ.get("GOOGLE_API_KEY"):
        print("🚨 GOOGLE_API_KEY not found. Please set it in your .env file or environment.")
        return

    print("🤖 Initializing RAG Chatbot...")
    print("   (This might take a moment to load models and vector store)")
    
    # Set use_memory=True to test memory feature (though its integration with RetrievalQA is basic here)
    # For more robust memory, consider ConversationalRetrievalChain
    try:
        chatbot = Chatbot(use_memory=False) # Set to True to experiment with memory
    except ValueError as e:
        print(f"🚨 Error initializing chatbot: {e}")
        print("   Please ensure you've run `python utils/ingest.py` first to create the vector store.")
        return
        
    print("✅ Chatbot initialized! Type 'exit' or 'quit' to end.")
    print("---")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("🤖 Exiting chatbot. Goodbye!")
            break
        
        if not user_input.strip():
            continue

        answer, sources = chatbot.ask(user_input)
        print(f"Bot: {answer}")
        
        if sources:
            print("\n   🔍 Sources:")
            for i, doc in enumerate(sources):
                source_name = os.path.basename(doc.metadata.get('source', 'Unknown'))
                print(f"     [{i+1}] {source_name}")
        print("---")

if __name__ == "__main__":
    run_cli()