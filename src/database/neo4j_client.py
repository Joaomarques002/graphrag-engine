import os
import logging
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph

# Carregar variáveis do ficheiro .env
load_dotenv()

# Configuração de logging profissional
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Neo4jClient:
    """Cliente wrapper para gerir a ligação e a execução de queries Cypher no Neo4j."""

    def __init__(
        self,
        url: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ) -> None:
        """Inicializa as credenciais e estabelece a conexão ao Neo4j."""
        self.url = url or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.username = username or os.getenv("NEO4J_USERNAME", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "password123")
        self.graph: Optional[Neo4jGraph] = None
        
        # Conectar imediatamente ao instanciar
        self._connect()

    def _connect(self) -> None:
        """Estabelece a ligação segura com a instância do Neo4j."""
        try:
            self.graph = Neo4jGraph(
                url=self.url,
                username=self.username,
                password=self.password,
                refresh_schema=False,  # Schema será atualizado manualmente quando necessário
            )
            logger.info("✅ Ligação ao Neo4j estabelecida com sucesso.")
        except Exception as e:
            logger.error(f"❌ Falha ao conectar à base de dados Neo4j: {str(e)}")
            raise ConnectionError(f"Erro de conexão ao Neo4j: {str(e)}")

    def execute_query(
        self, query: str, params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Executa uma query Cypher de forma segura utilizando parâmetros.
        
        Args:
            query (str): A instrução Cypher a executar.
            params (dict, optional): Parâmetros para prevenir Cypher Injection.
            
        Returns:
            List[Dict[str, Any]]: Lista com os resultados retornados pelo Neo4j.
        """
        if not self.graph:
            raise RuntimeError("Instância do Neo4j Graph não está inicializada.")

        try:
            return self.graph.query(query, params=params or {})
        except Exception as e:
            logger.error(f"❌ Erro ao executar query Cypher: {str(e)} | Query: {query}")
            raise

    def clear_database(self) -> None:
        """Remove TODOS os nós e relações do grafo. Útil para reset de testes."""
        cypher_clear = "MATCH (n) DETACH DELETE n;"
        try:
            self.execute_query(cypher_clear)
            logger.info("🧹 Base de dados limpa com sucesso.")
        except Exception as e:
            logger.error(f"❌ Erro ao limpar a base de dados: {str(e)}")
            raise

    def refresh_schema(self) -> None:
        """Atualiza a cache do esquema do grafo mantido pelo LangChain."""
        if self.graph:
            self.graph.refresh_schema()
            logger.info("🔄 Esquema do grafo atualizado.")