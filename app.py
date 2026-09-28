import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from src.database.neo4j_client import Neo4jClient
from src.nlp_extraction.extractor import GraphExtractor
from src.ingestion.pipeline import IngestionPipeline
from src.rag_chain.engine import GraphRAGEngine

load_dotenv()

# Configuração da página
st.set_page_config(
    page_title="GraphRAG Engine",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 1. Inicialização de Componentes com Cache
# ==========================================

@st.cache_resource
def get_system_components():
    """Inicializa e reaproveita os componentes pesados do sistema."""
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


# ==========================================
# 2. Barra Lateral (Sidebar) - Gestão e Ingestão
# ==========================================

with st.sidebar:
    st.title("🌐 GraphRAG Painel")
    st.caption("Sistema RAG baseado em Grafo de Conhecimento")
    st.divider()

    # Estado da Conexão
    st.success("🟢 Neo4j Conectado", icon="✅")

    st.subheader("📄 Ingestão de Documentos")
    uploaded_file = st.file_uploader(
        "Carrega ficheiros PDF ou TXT", 
        type=["pdf", "txt"],
        help="O documento será processado e convertido em entidades e relações no Neo4j."
    )

    if uploaded_file is not None:
        if st.button("Processar e Ingerir", use_container_width=True, type="primary"):
            with st.spinner("A extrair entidades e a povoar o grafo..."):
                try:
                    # Guardar ficheiro temporariamente
                    suffix = f".{uploaded_file.name.split('.')[-1]}"
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name

                    # Processar no pipeline
                    pipeline.process_file(tmp_path)
                    os.remove(tmp_path)  # Limpar ficheiro temporário
                    
                    st.toast(f"Ficheiro '{uploaded_file.name}' processado com sucesso!", icon="🎉")
                except Exception as e:
                    st.error(f"Erro na ingestão: {str(e)}")

    st.divider()

    # Manutenção da Base de Dados
    st.subheader("⚙️ Ações do Grafo")
    if st.button("🧹 Limpar Base de Dados", use_container_width=True):
        if st.checkbox("Confirmar eliminação de todos os nós"):
            db_client.clear_database()
            st.warning("Base de dados limpa com sucesso.")
            st.session_state.messages = []  # Limpar histórico do chat


# ==========================================
# 3. Interface Principal de Chat (GraphRAG)
# ==========================================

st.title("💬 GraphRAG Assistant")
st.markdown("Interroga a tua base de conhecimento fundamentada em **Grafos de Conhecimento**.")

# Inicializar Histórico de Conversa na Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Exibir Mensagens Anteriores
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "context" in message and message["context"]:
            with st.expander("🔎 Ver Subgrafo Recuperado do Neo4j"):
                st.code(message["context"], language="text")

# Entrada do Utilizador
if prompt := st.chat_input("Faz uma pergunta sobre os documentos ingeridos..."):
    # Guardar e exibir mensagem do utilizador
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Gerar Resposta via GraphRAGEngine
    with st.chat_message("assistant"):
        with st.spinner("A navegar pelas conexões do grafo..."):
            result = rag_engine.query(prompt)
            answer = result["answer"]
            context = result["context_used"]

            st.markdown(answer)
            
            # Exibir contexto relacional
            if context and "Nenhum subgrafo" not in context:
                with st.expander("🔎 Ver Subgrafo Recuperado do Neo4j"):
                    st.code(context, language="text")

    # Guardar resposta no histórico
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "context": context
    })