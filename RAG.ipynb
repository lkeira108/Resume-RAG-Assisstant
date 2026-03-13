"""
RAG Resume Chatbot - Converted from Jupyter Notebook
Deployed on Hugging Face Spaces
"""

import os
from os import path, listdir
from typing import List, Optional
import PyPDF2
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from google import genai  # CHANGED: switched from deprecated google.generativeai to new google.genai SDK
from google.genai import types  # CHANGED: import types for content/config objects used by new SDK
import gradio as gr

# Simple Vector Store
class SimpleVectorStore:
    def __init__(self, documents, embed_fn):
        self.documents = documents
        self.doc_embeddings = [embed_fn(doc.page_content) for doc in documents]

    def similarity_search(self, query, embed_fn, k=5):
        query_embedding = embed_fn(query)
        similarities = []
        for i, doc_embedding in enumerate(self.doc_embeddings):
            similarity = sum(a * b for a, b in zip(query_embedding, doc_embedding))
            similarities.append((similarity, i))
        similarities.sort(reverse=True)
        return [self.documents[i] for _, i in similarities[:k]]

# Document Loading
def load_pdfs(resume_directory_path: Optional[str] = None) -> List[Document]:
    if resume_directory_path is None:
        resume_directory_path = "./resumes"

    if not path.exists(resume_directory_path):
        print(f"Directory not found: {resume_directory_path}")
        return []

    documents = []
    for filename in listdir(resume_directory_path):
        if filename.endswith(".pdf") and ("KL" in filename.upper() or "LINKEDIN" in filename.upper()):
            file_path = path.join(resume_directory_path, filename)
            try:
                with open(file_path, 'rb') as pdf_file:
                    pdf_reader = PyPDF2.PdfReader(pdf_file)
                    for page_num, page in enumerate(pdf_reader.pages):
                        page_text = page.extract_text()
                        if page_text.strip():
                            doc_type = "linkedin" if "LINKEDIN" in filename.upper() else "resume"
                            metadata = {
                                "filename": filename,
                                "page_number": page_num + 1,
                                "doc_type": doc_type
                            }
                            documents.append(Document(page_content=page_text, metadata=metadata))
            except Exception as e:
                print(f"Error: {filename}: {e}")
    return documents

# RAG Chat System
class RAGChatSystem:
    def __init__(self):
        # CHANGED: new SDK uses genai.Client() instead of genai.configure()
        self.client = genai.Client(
            api_key=os.environ["GOOGLE_API_KEY"],
        )
        self.vector_store = None

       # CHANGED: switched to local sentence-transformers embeddings to avoid Gemini embedding API issues
        from sentence_transformers import SentenceTransformer
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embed_fn = lambda text: self.embedding_model.encode(text).tolist()

    def initialize_rag(self):
        # Load summary text
        try:
            with open("me/summary.txt", "r", encoding="utf-8") as f:
                self.summary = f.read()
        except FileNotFoundError:
            self.summary = ""
            print("Warning: me/summary.txt not found")

        # Load PDF documents
        documents = load_pdfs()
        if not documents:
            return "No documents found"

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        self.vector_store = SimpleVectorStore(chunks, self.embed_fn)
        return f"RAG system ready! Loaded {len(documents)} document pages."

    def query(self, question):
        if not self.vector_store:
            return "No context available"
        docs = self.vector_store.similarity_search(question, self.embed_fn, k=5)
        return "\n".join([doc.page_content for doc in docs])[:3000]

    def chat(self, message, history):
        """Chat function that combines RAG context with Gemini."""
        try:
            rag_context = self.query(message)
            context = f"Summary: {self.summary}\n\nDetailed context: {rag_context}"
            name = "Keira"

            system_prompt = f"""You ARE {name}. Use "I", "my", "me" - never third person.
You're answering questions about YOUR career and background on YOUR website.
Your background: {context}
Answer questions based on the context provided. If the context contains relevant information, use it fully and elaborately. Only say you don't have details if the topic is truly not mentioned anywhere in the context.
Be professional, optimistic, and engaging, as if talking to a potential client or future employer who came across the website.
"""

            # CHANGED: new SDK uses types.Content and types.Part for history format
            gemini_history = []
            for msg in history[-10:]:
                if isinstance(msg, dict) and 'role' in msg:
                    role = "user" if msg["role"] == "user" else "model"
                    gemini_history.append(
                        types.Content(role=role, parts=[types.Part(text=msg["content"])])
                    )

            # CHANGED: new SDK uses client.chats.create() instead of model.start_chat()
            chat_session = self.client.chats.create(
                model="gemini-3-flash-preview",
                history=gemini_history,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,  # CHANGED: new SDK supports system_instruction natively in config
                    max_output_tokens=1500,
                    temperature=0.5
                )
            )

            # CHANGED: new SDK uses chat_session.send_message() same as before but on the new client object
            gemini_response = chat_session.send_message(message)
            bot_response = gemini_response.text.strip()

            history.append({"role": "user", "content": message})
            history.append({"role": "model", "content": bot_response})

            return bot_response, history

        except Exception as e:
            print(f"Error in chat: {e}")
            error_msg = "I'm having technical difficulties. Please contact me via email for more information."
            history.append({"role": "user", "content": message})
            history.append({"role": "model", "content": error_msg})
            return error_msg, history

# Gradio Interface
def create_gradio_interface():
    """Create and launch Gradio chat interface."""

    rag_chat = RAGChatSystem()
    init_message = rag_chat.initialize_rag()
    print(init_message)

    def chat_fn(message, history):
        # Convert Gradio 5 message dict history to Gemini-compatible dict format
        gemini_history = []
        for msg in history:
            role = "user" if msg["role"] == "user" else "model"
            gemini_history.append({"role": role, "content": msg["content"]})

        response, updated_gemini_history = rag_chat.chat(message, gemini_history)

        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response})
        return history, ""

    with gr.Blocks(title="Resume RAG Chat") as demo:
        gr.Markdown("# 💬 Keira's Resume RAG Chat Assistant")
        gr.Markdown("Ask questions about the candidate's resume!")

        chatbot = gr.Chatbot(
            value=[],
            elem_id="chatbot",
            bubble_full_width=False,
            height=500,
            type="messages"
        )

        with gr.Row():
            msg = gr.Textbox(
                placeholder="Ask me about my experience...",
                container=False,
                scale=7
            )
            submit = gr.Button("Send", scale=1, variant="primary")
            clear = gr.Button("Clear", scale=1)

        msg.submit(chat_fn, [msg, chatbot], [chatbot, msg])
        submit.click(chat_fn, [msg, chatbot], [chatbot, msg])
        clear.click(lambda: ([], ""), None, [chatbot, msg], queue=False)

    return demo

if __name__ == "__main__":
    print("Starting Resume RAG Chat Interface...")
    print("Make sure GOOGLE_API_KEY is set in Hugging Face Secrets!")
    demo = create_gradio_interface()
    demo.launch()
