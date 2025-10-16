import streamlit as st
import os
import requests
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import sympy as sp
import numpy as np
import torch

# Download PDFs if not present (automated)
@st.cache_resource
def download_pdfs():
    docs_dir = "docs"
    os.makedirs(docs_dir, exist_ok=True)
    
    urls = {
        "uganda_structural.pdf": "https://nbrb.go.ug/wp-content/uploads/2022/01/Buidling-Structural-Design.pdf",
        "eurocode_en1992.pdf": "https://www.phd.eng.br/wp-content/uploads/2015/12/en.1992.1.1.2004.pdf",
        "bs8110_part1.pdf": "https://crcrecruits.files.wordpress.com/2014/04/bs8110-1-1997-structural-use-of-concrete-design-construction.pdf",
        "road_vol1.pdf": "https://www.works.go.ug/sites/default/files/Road%20Design%20Manual%20Volume%201_Geometric%20Design_MoWT_2010.pdf"  # Direct from MoWT
    }
    
    for filename, url in urls.items():
        filepath = os.path.join(docs_dir, filename)
        if not os.path.exists(filepath):
            st.info(f"Downloading {filename}...")
            response = requests.get(url)
            with open(filepath, "wb") as f:
                f.write(response.content)
            st.success(f"Downloaded {filename}")
    
    # Sample AASHTO text (full PDF not free; add manually if obtained)
    aashto_path = os.path.join(docs_dir, "sample_aashto.txt")
    if not os.path.exists(aashto_path):
        with open(aashto_path, "w") as f:
            f.write("""
AASHTO LRFD Bridge Design Specifications (9th Ed., 2020 Excerpts):
Section 3: Loads and Load Factors. Live Load HL-93: Design truck + lane load. Multiple Presence Factor: 1.20 for 2 lanes.
Section 5: Concrete Structures. Reinforcement: Min As = 0.004 Ag for temp/shrinkage.
Bridge Deck: Slab thickness min 7 in., rebar spacing 6 in. c/c.
Partial factors: DC=1.25, DW=1.50, LL=1.75.
""")
        st.success("Added AASHTO sample")

# Build RAG Index (automated)
@st.cache_resource
def build_rag():
    download_pdfs()
    docs_dir = "docs"
    documents = []
    
    # Load PDFs/Texts
    for file in os.listdir(docs_dir):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(docs_dir, file))
        else:
            loader = TextLoader(os.path.join(docs_dir, file))
        documents.extend(loader.load())
    
    # Split and Embed
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(texts, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 3})

# Calc Tools
@tool
def calculate_bridge_reinforcement(span_m: float, width_m: float, live_load_kN_m: float, standard: str = "Eurocode") -> dict:
    """Calculates main reinforcement for simply supported RC bridge deck per Eurocode/AASHTO/BS."""
    L = span_m
    b = width_m
    q_live = live_load_kN_m  # e.g., HA loading ~30 kN/m
    q_dead = 25 * b / 1000  # Self-weight kN/m (concrete 25 kN/m³)
    
    if standard == "AASHTO":
        gamma_d, gamma_l = 1.25, 1.75
    else:  # Eurocode/BS
        gamma_d, gamma_l = 1.35, 1.5
    
    M_ult = (gamma_d * q_dead + gamma_l * q_live) * L**2 / 8 / 1000  # kNm
    z = 0.95 * 200  # Lever arm mm (assume h=250, d=200)
    As = M_ult * 1e6 / (0.87 * 500 * z)  # mm²/m (fyk=500 MPa)
    area_per_bar = np.pi * (12/2)**2  # Assume 12mm dia
    num_bars = np.ceil(As / area_per_bar)
    spacing = 1000 / num_bars * 10  # mm c/c approx
    
    return {"moment_kNm": M_ult, "As_mm2_m": As, "suggested_rebar": f"{int(12)}mm dia, {int(spacing)}mm c/c"}

@tool
def calculate_footing_size(load_kN: float, soil_bearing_kPa: float, standard: str = "Uganda") -> dict:
    """Simple pad footing size per BS 8110/Uganda Code."""
    if standard == "Uganda":
        factor = 1.4  # Partial for dead load
    else:
        factor = 1.4
    service_load = load_kN / factor
    area_m2 = service_load / soil_bearing_kPa
    side_m = np.sqrt(area_m2)
    return {"area_m2": area_m2, "side_length_m": side_m, "soil_capacity_check": f"Bearing stress: {service_load / area_m2:.1f} kPa < {soil_bearing_kPa} kPa"}

# Setup LLM and Agent
@st.cache_resource
def setup_agent(retriever):
    # Small open-source LLM
    model_id = "microsoft/Phi-3-mini-4k-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=200)
    llm = HuggingFacePipeline(pipeline=pipe)
    
    tools = [calculate_bridge_reinforcement, calculate_footing_size]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a Ugandan civil engineering AI expert in transportation/structures per Uganda Code, DRMB, AASHTO, Eurocode, BS. Use tools for calcs; retrieve docs for standards. Cite sources. Disclaimer: Verify professionally."),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True, handle_parsing_errors=True)
    return agent_executor

# Streamlit UI
st.title("🛤️ Uganda Civil Engineering AI: Transportation & Structures")
st.sidebar.info("📚 Docs: Uganda Structural Code, Eurocode, BS 8110, Road Vol1, AASHTO samples loaded.")

retriever = build_rag()
agent = setup_agent(retriever)

# Tabs for Quick Calcs
tab1, tab2, tab3 = st.tabs(["Bridge Reinforcement", "Footing Design", "Research Chat"])

with tab1:
    col1, col2, col3, col4 = st.columns(4)
    with col1: span = st.number_input("Span (m)", value=20.0)
    with col2: width = st.number_input("Width (m)", value=7.5)
    with col3: load = st.number_input("Live Load (kN/m)", value=50.0)
    with col4: std = st.selectbox("Standard", ["Eurocode", "AASHTO"])
    if st.button("Calculate Bridge Rebar", key="bridge"):
        res = calculate_bridge_reinforcement(span, width, load, std)
        st.json(res)
        st.success("Per EN 1992/BS 5400/AASHTO LRFD.")

with tab2:
    col1, col2 = st.columns(2)
    with col1: load_kN = st.number_input("Axial Load (kN)", value=1000.0)
    with col2: bearing = st.number_input("Soil Bearing (kPa)", value=150.0)
    std = st.selectbox("Standard", ["Uganda", "BS8110"])
    if st.button("Calculate Footing", key="footing"):
        res = calculate_footing_size(load_kN, bearing, std)
        st.json(res)
        st.success("Per Uganda Code Schedule 2/BS 8110 Clause 3.11.")

with tab3:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Ask about standards, designs, or calcs (e.g., 'Explain bridge loading per DRMB')"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = agent.invoke({"input": prompt})
            st.markdown(response["output"])
            st.session_state.messages.append({"role": "assistant", "content": response["output"]})
