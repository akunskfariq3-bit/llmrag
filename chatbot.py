from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import gradio as gr
from groq import Groq
import os
from pydantic import Field

from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import LLMResult, Generation
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from dotenv import load_dotenv
load_dotenv()

DATA_PATH = r"data"
CHROMA_PATH = r"chroma_db"

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)
model_name = "llama-3.1-8b-instant"

embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

class GroqLLM(BaseLLM):
    """Custom LangChain LLM wrapper for the Groq API."""
    
    client: Groq = Field(..., exclude=True)
    model_name: str = Field(...)
    
    def __init__(self, client: Groq, model_name: str, **kwargs):
        super().__init__(client=client, model_name=model_name, **kwargs)
    
    @property
    def _llm_type(self) -> str:
        return "groq"
    
    def _generate(self, prompts: list[str], **kwargs) -> LLMResult:
        """Required method by BaseLLM to generate text."""
        generations = []
        
        for prompt in prompts:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0, 
                **kwargs
            )
            text = response.choices[0].message.content
            generations.append([Generation(text=text)])
            
        return LLMResult(generations=generations)

    def stream(self, prompt: str, **kwargs):
        """Implement stream method to support Gradio streaming."""
        response_stream = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            stream=True,
            **kwargs
        )
        
        for chunk in response_stream:
            if chunk.choices and chunk.choices[0].delta.content is not None:
                # Yield the content directly for processing in stream_response
                yield chunk.choices[0].delta

llm = GroqLLM(client=client, model_name=model_name)

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH,
)

num_results = 5
retriever = vector_store.as_retriever(search_kwargs={'k': num_results})

def stream_response(message, history):

    history_str = ""
    for user_msg, bot_msg in history:
        history_str += f"User: {user_msg}\nAssistant: {bot_msg}\n"

    try:
        docs = retriever.invoke(message)
    except Exception as e:
        yield f"Error retrieving documents: {e}. Check if the '{CHROMA_PATH}' directory exists and contains data."
        return

    knowledge = ""
    for doc in docs:
        knowledge += doc.page_content + "\n\n"

    if message is not None:
        partial_message = ""

        rag_prompt = f"""
Anda adalah seorang asisten yang menjawab HANYA menggunakan informasi yang ada di bagian 'Knowledge' di bawah.
JANGAN gunakan pengetahuan internal atau pengetahuan yang tidak relevan.
JANGAN sebutkan bagian 'Knowledge' kepada pengguna.

WALAUPUN dokumen sumber berbahasa Inggris, JAWABAN AKHIR ANDA HARUS SELALU DALAM BAHASA INDONESIA.

Pertanyaan: {message}

Riwayat Percakapan:
{history_str}

Knowledge:
{knowledge}
"""

        try:
            for response in llm.stream(rag_prompt):
                content_chunk = response.content
                if content_chunk:
                    partial_message += content_chunk
                    yield partial_message
        except Exception as e:
            yield partial_message + f"\n\n[Error: Failed to stream response from LLM. Check API key and network connection: {e}]"


chatbot = gr.ChatInterface(
    stream_response,
    title="RAG Chatbot (Groq + HuggingFace Embeddings + ChromaDB)",
    textbox=gr.Textbox(
        placeholder="Ask something...",
        container=False,
        autoscroll=True,
        scale=7
    ),
)

if __name__ == "__main__":
    print("Launching Gradio App...")
    chatbot.launch()