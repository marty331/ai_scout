import os
import json
import logging

import chromadb
import gradio as gr

from huggingface_hub import snapshot_download
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import RetrieverTool, ToolMetadata
from llama_index.agent.openai import OpenAIAgent
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.llms import MessageRole
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.memory import ChatSummaryMemoryBuffer
from llama_index.core import Settings
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Placeholder for the API key
api_key = None

PROMPT_SYSTEM_MESSAGE = """You are an AI expert in the area of Scouts BSA (formerly Boy Scouts of America, answering questions from visitors regarding all things related to Scouts BSA, such as rank requirements, merit badge requirements, high adventure, and camping. 
Topics covered include Scouting ranks and requirements, merit badge requirements, different choices for high adventure, requirements to attend high adventure camps, camping, outdoor activies that can be performed on a campout, leadership opportunities in scounting. Questions should be understood in this context. Your answers are aimed to teach 
visitors, so they should be complete, clear, and easy to understand. Use the available tools to gather insights pertinent to the topic of Scouting.
To find relevant information for answering visitors questions, always use the "Scouts_Information_related_resources" tool.

Only some information returned by the tool might be relevant to the question, so ignore the irrelevant part and answer the question with what you have. Your responses are exclusively based on the output provided 
by the tools. Refrain from incorporating information not directly obtained from the tool's responses.
If a user requests further elaboration on a specific aspect of a previously discussed topic, you should reformulate your input to the tool to capture this new angle or more profound layer of inquiry. Provide 
comprehensive answers, ideally structured in multiple paragraphs, drawing from the tool's variety of relevant details. The depth and breadth of your responses should align with the scope and specificity of the information retrieved. 
Should the tool response lack information on the queried topic, politely inform the user that the question transcends the bounds of your current knowledge base, citing the absence of relevant content in the tool's documentation. 
At the end of your answers, always invite the students to ask deeper questions about the topic if they have any.
Do not refer to the documentation directly, but use the information provided within it to answer questions. If code is provided in the information, share it with the students. It's important to provide complete code blocks so 
they can execute the code when they copy and paste them. Make sure to format your answers in Markdown format, including code blocks and snippets.
"""

def download_knowledge_base_if_not_exists():
    """Download the knowledge base from the Hugging Face Hub if it doesn't exist locally"""
    if not os.path.exists("data/scout_data"):
        os.makedirs("data/scout_data")

        logging.warning(
            f"Vector database does not exist at 'data/', downloading from Hugging Face Hub..."
        )
        snapshot_download(
            repo_id="marty331/scouts_dataset_vector_store",
            local_dir="data/scout_data",
            repo_type="dataset",
        )
        logging.info(f"Downloaded vector database to 'data/scout_data'")


def get_tools(db_collection="scout_data"):
    db = chromadb.PersistentClient(path=f"data/{db_collection}")
    chroma_collection = db.get_or_create_collection(db_collection)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        show_progress=True,
        use_async=True,
        embed_model=Settings.embed_model
    )
    vector_retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=15,
        embed_model=Settings.embed_model,
        use_async=True,
    )
    tools = [
        RetrieverTool(
            retriever=vector_retriever,
            metadata=ToolMetadata(
                name="Scouts_Information_related_resources",
                description="Useful for info related to artificial intelligence, ML, deep learning. It gathers the info from local data.",
            ),
        )
    ]
    return tools

def generate_completion(query, history, memory):
    logging.info(f"User query: {query}")
    global api_key
    if not api_key:
        raise gr.Error("API Key not set", duration=10)

    # Manage memory
    chat_list = memory.get()
    if len(chat_list) != 0:
        user_index = [i for i, msg in enumerate(chat_list) if msg.role == MessageRole.USER]
        if len(user_index) > len(history):
            user_index_to_remove = user_index[len(history)]
            chat_list = chat_list[:user_index_to_remove]
            memory.set(chat_list)
    logging.info(f"chat_history: {len(memory.get())} {memory.get()}")
    logging.info(f"gradio_history: {len(history)} {history}")

    # Create agent
    tools = get_tools(db_collection="scout_data")
    agent = OpenAIAgent.from_tools(
        llm=Settings.llm,
        memory=memory,
        tools=tools,
        system_prompt=PROMPT_SYSTEM_MESSAGE,
    )

    # Generate answer
    completion = agent.stream_chat(query)
    answer_str = ""
    try:
        for token in completion.response_gen:
            answer_str += token
            yield answer_str
    except Exception as e:
        raise gr.Error(f"Error: {e}", duration=10)

# Function to capture the API key
def set_api_key(input_key):
    global api_key
    api_key = input_key  # Store the API key globally
    os.environ["OPENAI_API_KEY"] = api_key
    # Set up llm and embedding model
    Settings.llm = OpenAI( temperature=0, model="gpt-4o-mini")
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large")
    return f"API Key set to: {api_key}"

def launch_ui():
    with gr.Blocks(
        fill_height=True,
        title="AI Scout Helper",
        analytics_enabled=True,
    ) as demo:
        memory_state = gr.State(
            lambda: ChatSummaryMemoryBuffer.from_defaults(
                token_limit=120000,
            )
        )
        # Define the Gradio components
        # Bind the API key input and chat interface together
        api_key_input = gr.Textbox(label="Enter API Key", placeholder="Enter your API key here, do not change after it's entered.")
        with gr.Row():
            #api_key_input.render()
            gr.Button("Set API Key").click(set_api_key, inputs=api_key_input, outputs=gr.Textbox(label="Status"))
        chatbot = gr.Chatbot(
            scale=1,
            placeholder="<strong>AI Scout Helper: A Question-Answering Bot for all things BSA!</strong><br>",
            show_label=False,
            show_copy_button=True,
        )
        gr.ChatInterface(
            fn=generate_completion, 
            chatbot=chatbot,
            additional_inputs=[memory_state]
            )
        
        demo.queue(default_concurrency_limit=64)
        demo.launch(debug=True, share=False) # Set share=True to share the app online


if __name__ == "__main__":
    # Download the knowledge base if it doesn't exist
    download_knowledge_base_if_not_exists()

    # launch the UI
    launch_ui()