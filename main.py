import streamlit as st
import logging
from pathlib import Path
import os

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self):
        # Optimized settings for better text splitting
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # Smaller chunks for more precise matching
            chunk_overlap=100,  # Good overlap to maintain context
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", " ", ""] # More granular separation
        )

    def process_file(self, file_path: str):
        try:
            file_extension = Path(file_path).suffix.lower()
            
            if file_extension == '.pdf':
                loader = PyPDFLoader(file_path)
                documents = loader.load()
                # Preserve page numbers in metadata
                text_content = [(doc.page_content, {"page": doc.metadata.get('page', 0)}) for doc in documents]
            elif file_extension == '.txt':
                loader = TextLoader(file_path)
                documents = loader.load()
                text_content = [(doc.page_content, {"page": 1}) for doc in documents]
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")

            chunks = []
            metadatas = []
            
            for text, metadata in text_content:
                chunk_texts = self.text_splitter.split_text(text)
                chunks.extend(chunk_texts)
                # Add metadata for each chunk
                metadatas.extend([{
                    "page": metadata["page"],
                    "chunk": i,
                    "source": str(file_path)
                } for i in range(len(chunk_texts))])

            return chunks, metadatas

        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            raise

class EnhancedRAG:
    def __init__(self, api_key: str):
        os.environ["GROQ_API_KEY"] = api_key
        self.llm = ChatGroq(
            model="llama-3.2-11b-text-preview",
            temperature=0.1,  # Lower temperature for more focused answers
            max_tokens=None,
        )

        # Use better embedding model
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        self.vector_store = None
        self.doc_processor = DocumentProcessor()

        self.initial_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant. For all responses:
            1. If you know the answer with high confidence: Provide clear, accurate information
            2. For ANY uncertainty or lack of knowledge: Begin your response with exactly "I don't know" followed by a brief explanation
            3. Never speculate or provide uncertain information
            4. Keep responses natural and direct"""),
            ("human", "{input}")
        ])

        # Enhanced RAG prompt for better document-based answers
        self.doc_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a precise and detailed assistant. Analyze the provided context carefully and answer the question.
            
            Follow these rules strictly:
            1. Base your answer ONLY on the provided context
            2. Quote relevant parts of the context to support your answer
            3. If the context doesn't contain enough information, clearly state what's missing
            4. If you find contradictions in the context, point them out
            5. Maintain the original meaning and intent of the source material
            
            Context: {context}
            
            Remember to focus on accuracy and relevance to the question."""),
            ("human", "{question}")
        ])

    def load_document(self, file_path: str):
        try:
            chunks, metadatas = self.doc_processor.process_file(file_path)
            if self.vector_store is None:
                self.vector_store = FAISS.from_texts(
                    texts=chunks,
                    embedding=self.embeddings,
                    metadatas=metadatas
                )
            else:
                self.vector_store.add_texts(
                    texts=chunks,
                    metadatas=metadatas
                )
            return True
        except Exception as e:
            logger.error(f"Document loading failed: {str(e)}")
            raise

    def get_initial_response(self, question: str):
        try:
            chain = self.initial_prompt | self.llm
            response = chain.invoke({"input": question})
            return response.content
        except Exception as e:
            logger.error(f"Initial response error: {str(e)}")
            return "I don't know. There was an error processing your question."

    def get_doc_based_response(self, question: str):
        try:
            if self.vector_store is None:
                return "Please upload a document first."

            # Enhanced retrieval strategy
            retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={
                    "k": 4,  # Number of relevant chunks to retrieve
                    "score_threshold": 0.3  # Lower threshold for better recall
                }
            )
            
            # Get relevant documents with their scores
            docs_and_scores = self.vector_store.similarity_search_with_score(question, k=4)
            
            if not docs_and_scores:
                return "I couldn't find relevant information in the uploaded documents."
            
            # Sort documents by relevance score
            docs_and_scores.sort(key=lambda x: x[1])
            
            # Format context with relevance information
            formatted_contexts = []
            for doc, score in docs_and_scores:
                page_info = f"[Page {doc.metadata.get('page', 'N/A')}]" if 'page' in doc.metadata else ""
                formatted_contexts.append(f"{page_info} {doc.page_content}")
            
            context = "\n\n---\n\n".join(formatted_contexts)

            # Generate response using the enhanced prompt
            chain = self.doc_prompt | self.llm
            response = chain.invoke({
                "question": question,
                "context": context
            })
            
            # Format the final response with source attribution
            return f"{response.content}\n\nThis information comes from {len(docs_and_scores)} relevant sections of the document."

        except Exception as e:
            logger.error(f"Document-based response error: {str(e)}")
            return "I'm having trouble processing the documents."

# Streamlit interface
st.title("AI Assistant")

# Initialize session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = EnhancedRAG(api_key="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")

# User input
question = st.text_input("Ask me anything!")

if question:
    # Get initial response
    initial_response = st.session_state.rag_system.get_initial_response(question)
    
    # Display the response
    st.write("*Assistant:*", initial_response)
    
    # Show file upload if response starts with "I don't know"
    if initial_response.lower().startswith("i don't know"):
        st.write("📚 Upload a document to help me answer your question.")
        uploaded_file = st.file_uploader("Upload a relevant document (PDF or TXT)", type=["pdf", "txt"])

        if uploaded_file:
            temp_file_path = Path(f"./temp_{uploaded_file.name}")
            temp_file_path.write_bytes(uploaded_file.read())
            
            with st.spinner("Processing your document..."):
                try:
                    if st.session_state.rag_system.load_document(str(temp_file_path)):
                        doc_response = st.session_state.rag_system.get_doc_based_response(question)
                        st.write("*Based on the document:*", doc_response)
                except Exception as e:
                    st.error(f"Error processing document: {e}")
                finally:
                    if temp_file_path.exists():
                        temp_file_path.unlink()