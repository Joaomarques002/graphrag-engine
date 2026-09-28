import os
import tempfile
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

from src.database.neo4j_client import Neo4jClient
from src.nlp_extraction.extractor import GraphExtractor
from src.ingestion.pipeline import IngestionPipeline
from src.rag_chain.engine import GraphRAGEngine
from src.visualization.graph_visualizer import generate_subgraph_html

load_dotenv()

st.set_page_config(
    page_title="GraphRAG Engine",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def get_system_components():
    db_client = Neo4jClient()
    extractor = GraphExtractor()
    pipeline = IngestionPipeline(db_client=db_client, extractor=extractor)
    rag_engine = GraphRAGEngine(db_client=db_client)
    return db_client, pipeline, rag_engine


try:
    db_client, pipeline, rag_engine = get_system_components()
except Exception as e:
    st.error(f"❌ Erro ao conectar ao Neo4j ou Ollama: {str(e)}")
    st.stop()


# Sidebar
with st.sidebar:
    st.title("🌐 GraphRAG Painel")
    st.caption("Sistema RAG baseado em Grafo de Conhecimento")
    st.divider()

    st.success("🟢 Neo4j Conectado", icon="✅")

    st.subheader("📄 Ingestão de Documentos")
    uploaded_file = st.file_uploader("Carrega ficheiros PDF ou TXT", type=["pdf", "txt"])

    if uploaded_file is not None:
        if st.button("Processar e Ingerir", use_container_width=True, type="primary"):
            with st.spinner("A extrair entidades e a povoar o grafo..."):
                try:
                    suffix = f".{uploaded_file.name.split('.')[-1]}"
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name

                    pipeline.process_file(tmp_path)
                    os.remove(tmp_path)
                    st.toast(f"Ficheiro '{uploaded_file.name}' processado com sucesso!", icon="🎉")
                except Exception as e:
                    st.error(f"Erro na ingestão: {str(e)}")

    st.divider()

    st.subheader("⚙️ Ações do Grafo")
    if st.button("🧹 Limpar Base de Dados", use_container_width=True):
        if st.checkbox("Confirmar eliminação"):
            db_client.clear_database()
            st.warning("Base de dados limpa com sucesso.")
            st.session_state.messages = []


# Área Principal do Chat
st.title("💬 GraphRAG Assistant")
st.markdown("Interroga a tua base de conhecimento fundamentada em **Grafos de Conhecimento**.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Exibir Mensagens Anteriores
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "triples" in message and message["triples"]:
            with st.expander("🔎 Explorar Subgrafo de Contexto (Visual & Texto)"):
                tab_graph, tab_text = st.tabs(["🌐 Subgrafo Interativo (Pyvis)", "📜 Triplos Textuais"])
                
                with tab_graph:
                    html_graph = generate_subgraph_html(message["triples"])
                    components.html(html_graph, height=470, scrolling=False)
                
                with tab_text:
                    st.code(message["context"], language="text")

# Entrada de Pergunta
if prompt := st.chat_input("Faz uma pergunta sobre os documentos ingeridos..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("A navegar pelas conexões do grafo..."):
            result = rag_engine.query(prompt)
            answer = result["answer"]
            context = result["context_used"]
            triples = result["triples"]

            st.markdown(answer)

            if triples:
                with st.expander("🔎 Explorar Subgrafo de Contexto (Visual & Texto)"):
                    tab_graph, tab_text = st.tabs(["🌐 Subgrafo Interativo (Pyvis)", "📜 Triplos Textuais"])
                    
                    with tab_graph:
                        html_graph = generate_subgraph_html(triples)
                        components.html(html_graph, height=470, scrolling=False)
                    
                    with tab_text:
                        st.code(context, language="text")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "context": context,
        "triples": triples
    })