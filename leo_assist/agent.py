import sys
from pathlib import Path
import os

from google.genai import types
from google.adk.agents import Agent
from google.adk.planners.built_in_planner import BuiltInPlanner
from google.adk.tools import google_search
from typing import List

ROOT_DIR = Path(os.path.dirname(os.path.abspath(Path(__file__).parent)))
AGENT_DIR = ROOT_DIR / "leo_assist"

sys.path.insert(0, str(ROOT_DIR))

from leo_assist.retrieval.retriever import DocumentRetriever
from leo_assist.core.chroma_db_impl import ChromaDBVectorStore
from leo_assist.core.chunking.manager import ChunkerManager
from leo_assist.utils.settings import Settings
from leo_assist.core.vector_store import DocumentResult
from leo_assist.utils.logger import logger


settings = Settings(_env_file=f"{AGENT_DIR}/.env")

chunker_manager = ChunkerManager()
vector_store = ChromaDBVectorStore.from_settings(
    settings, chunker_manager=chunker_manager
)

retriever = DocumentRetriever.from_settings(
    vector_store=vector_store,
    settings=settings,
)

def retrieve_documents(query: str) -> str:
    """Retrieve relevant code and documents based on a query. 
    The query should be the exact same word as you user typed. 
    For example, "What is the syntax for a function in Leo?", then pass "What is the syntax for a function in Leo?" to the tool.

    Args:
        query: The search query. The query should be the exact same word as you user typed. For example, "What is the syntax for a function in Leo?", then pass "What is the syntax for a function in Leo?" to the tool.
    Returns:
        Formatted string containing relevant document information

    Examples:
        >>> retrieve_documents("What is the syntax for a function in Leo?")
        ""
    """

    def _format_retrieved_documents(documents: List[DocumentResult]) -> dict[str, str]:
        """Format retrieved documents into a dictionary mapping document IDs to their content.

        Args:
            documents: List of document chunks to format

        Returns:
            Dictionary where keys are document IDs and values are document contents
        """
        if not documents:
            return {}


        return {doc.document.id: doc.document.content for doc in documents}

    try:
        # Retrieve relevant documents
        documents = retriever.retrieve(
            query=query,
        )

        logger.info(f"Retrieved {len(documents)} documents")
        for doc in documents:
            logger.info(doc.document.model_dump_json(indent=2))

        # Format the documents
        return _format_retrieved_documents(documents)

    except Exception as e:
        error_msg = f"Error retrieving documents: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return f"Error retrieving documents: {str(e)}"

root_agent = Agent(
    name="leo_rag_assistant",
    model=settings.llm.model,
    # model="gemini-2.5-flash-preview-05-20",
    description=(
        "You are a Leo Code Assist Agent that provides assistance with the Leo programming language "
        "by retrieving and synthesizing information from documentation."
    ),
    instruction=(
        "You are a Leo Code Assist Agent that provides assistance with the Leo programming language. "
        "You provide accurate, concise, and helpful information about Leo programming, "
        "including syntax, standard library, best practices, and code examples.\n\n"
        "When answering questions:\n"
        "1. ALWAYS use 'retrieved_documents' tool to answer the question, and ALWAYS pass the exact query as provided by the user to the query param of retrieve_documents tool.\n"
        "For example, 'What is the syntax for a function in Leo?', then pass 'What is the syntax for a function in Leo?' to the query param of retrieve_documents tool.\n"
        "2. If you don't have enough information in the retrieved documents, use 'google_search' tool to search the web for more information.\n"
        "3. Be clear and concise in your explanations\n"
        "4. Use markdown to format code blocks and technical terms\n"
        "5. If you reference code from the context, mention the source\n"
        "6. If you don't know the answer, say so rather than making up information\n"
        "7. For code examples, ensure they are complete and runnable\n"
        "8. Follow Leo's coding conventions and best practices"
    ),
    # before_agent_callback=_before_agent_callback,
    tools=[retrieve_documents, google_search],
    planner=BuiltInPlanner(
        thinking_config=types.ThinkingConfig(
            include_thoughts=False,
            thinking_budget=100
        )   
    )
)