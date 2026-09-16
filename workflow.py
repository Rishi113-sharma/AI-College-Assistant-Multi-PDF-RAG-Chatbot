import os
from dotenv import load_dotenv

load_dotenv()
from typing import TypedDict ,Annotated
from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from typing import Literal
import torch    

embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

def build_retriver(pdf_path : str):
    loader = PyPDFLoader(pdf_path)
    document = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(chunk_size = 800, chunk_overlap = 100)
    
    chunks = splitter.split_documents(document)
    
    vector_store = FAISS.from_documents(chunks,embeddings)
    return vector_store.as_retriever(search_kwargs = {"k" :3})


academic_retriver = build_retriver("academics_handbook.pdf")
fee_retriver = build_retriver("fee_structure.pdf")
btech_retriver = build_retriver("btech_syllabus.pdf")

from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=0
)

class State(TypedDict):
    programmme : str
    messages : Annotated[list,add_messages]
    query_type : str
    retrieved_context: str


def get_text(response) -> str:

    content = response.content

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        text = ""

        for block in content:
            if isinstance(block, dict):
                text += block.get("text", "")

        return text.strip()

    return str(content).strip()    
    
    
    
def classifier_node(state : State) -> dict:
    """LOOK AT THE LATEST HUMAN MESSAGE AND DECIDE WHICH PATH TO TAKE."""
    
    last_message = state["messages"][-1].content
    prompt=  ( f"""You are a message classifier.

Classify the user's message into exactly ONE of these categories:

1. academic
   - Questions about subjects, courses, classes, exams, assignments,
     syllabus, attendance, results, marks, faculty, timetable, projects,
     or other academic matters.

2. fee
   - Questions about college fees, tuition fees, payment, fee structure,
     scholarships related to fees, due dates, refunds, fines, or payments.

3. general
   - Anything that does not belong to academic or fee-related topics.

User message:
{last_message}

Return ONLY the category name:
academic
fee
general
"""
    )
    response = llm.invoke(prompt)
    category = get_text(response).lower()
    
    if"academic" in category:
        category = "academic"
    elif "fee" in category:
        category ="fee"
    else : 
        category = "general"
        
    return{"query_type" : category}


def academic_rag_node(state: State) -> dict:
    query = state["messages"][-1].content

    docs = academic_retriver.invoke(query)

    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    return {
        "retrieved_context": context
    }

def fee_rag_node(state: State) -> dict:

    query = state["messages"][-1].content

    docs = fee_retriver.invoke(query)

    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    return {
        "retrieved_context": context
    }
def btech_rag_node(state: State) -> dict:

    query = state["messages"][-1].content

    docs = btech_retriver.invoke(query)

    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    return {
        "retrieved_context": context
    }
    
def general_node(state: State) -> dict:
    """ANSWER DIRECTLY USING LLM'S OWN DATA, NO RETRIEVAL NEEDED."""

    query = state["messages"][-1].content

    response = llm.invoke(query)
    answer = get_text(response)

    return {
        "messages": [("ai", answer)]
    }


def response_node(state: State) -> dict:

    query = state["messages"][-1].content
    programme = state.get("programme", "Unknown")
    context = state["retrieved_context"]

    prompt = f"""You are a college RAG assistant.

Student programme: {programme}

User question:
{query}

Information from the college document:
{context}

Answer the user's question using ONLY the information above.

IMPORTANT RULES:
- Give ONE answer only.
- Answer the exact question.
- Do not summarize unrelated information.
- Do not repeat any sentence.
- Do not repeat any fact.
- Do not invent information.
- Do not explain your reasoning.
- If the requested information is not present, say so in ONE sentence.
- Stop after giving the answer.

Answer:
"""

    response = llm.invoke(prompt)

    answer = get_text(response)

    return {
        "messages": [("ai", answer)]
    }

#ROUTER
def router_query(state:State)->Literal["academic_rag","fee_rag","general","btech_rag"]:
    if state['query_type']=="academic":
        return "academic_rag"
    elif state["query_type"]=="fee":
        return"fee_rag"
    elif state["query_type"] == "btech":
        return "btech_rag"
    else:
        return"general"
    
    
graph = StateGraph(State)   
   
graph.add_node("classifier", classifier_node)
graph.add_node("academic_rag", academic_rag_node)
graph.add_node("fee_rag", fee_rag_node)
graph.add_node("general", general_node)
graph.add_node("response", response_node)   
graph.add_node("btech_rag", btech_rag_node) 

graph.add_edge(START,"classifier")
graph.add_conditional_edges("classifier",router_query)
graph.add_edge("academic_rag","response")    
graph.add_edge("fee_rag","response")
graph.add_edge("btech_rag", "response")
graph.add_edge("general",END)
graph.add_edge("response",END)

app=graph.compile()












