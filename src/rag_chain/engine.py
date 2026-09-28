import os
import logging
from typing import Dict, List, Optional
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from src.database.neo4j_client import Neo4jClient

load_dotenv()

logger = logging.getLogger(__name__)


# ==========================================
# 1. Prompt de Sistema para o Motor GraphRAG
# ==========================================

RAG_SYSTEM_PROMPT = """Tu és um assistente técnico especialista com acesso a uma Base de Conhecimento em Grafo.
A tua tarefa é responder à pergunta do utilizador EXCLUSIVAMENTE com base nas relações e factos fornecidos no Contexto do Grafo.

Regras Estritas de Resposta:
1. Utiliza apenas a informação presente na estrutura do grafo fornecida abaixo.
2. Se o contexto não contiver informação suficiente para responder com certeza, declara explicitamente: "Com base no Grafo de Conhecimento atual, não existem dados suficientes para responder a essa questão."
3. Não inventes relações, datas ou factos que não estejam explicitamente declarados.
4. Mantém um tom profissional, estruturado e direto.

Contexto do Grafo Recuperado (Subgrafo de até 2 Saltos):
{graph_context}
"""


# ==========================================
# 2. Classe Principal do Motor GraphRAG
# ==========================================

class GraphRAGEngine:
    """Motor de consulta GraphRAG que combina pesquisa de entidades e
    exploração de subgrafos no Neo4j com geração fundamentada via Ollama."""

    def __init__(
        self,
        db_client: Neo4jClient,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.1,
    ) -> None:
        """Inicializa o motor RAG com a conexão Neo4j e o modelo Ollama."""
        self.db_client = db_client
        self.model_name = model_name or os.getenv("OLLAMA_MODEL", "llama3")
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        logger.info(f"🧠 A inicializar GraphRAGEngine com o modelo '{self.model_name}'")

        self.llm = ChatOllama(
            model=self.model_name,
            base_url=self.base_url,
            temperature=temperature,
        )

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", RAG_SYSTEM_PROMPT),
            ("human", "Pergunta: {question}"),
        ])

        self.chain = self.prompt | self.llm

    def _extract_keywords(self, question: str) -> List[str]:
        """Extrai palavras-chave e potenciais entidades da pergunta do utilizador."""
        # Filtra pontuação e palavras muito curtas (stopwords simples)
        words = [
            w.strip("?,.!;:\"'()[]{}")
            for w in question.split()
            if len(w.strip("?,.!;:\"'()[]{}")) > 3
        ]
        return words

    def _get_subgraph_context(self, keywords: List[str], max_results: int = 30) -> str:
        """Procura entidades correspondentes no Neo4j e navega pelas suas relações
        até 2 saltos de distância.
        
        Args:
            keywords (List[str]): Lista de termos a procurar.
            max_results (int): Limite máximo de triplos a retornar para evitar estourar a janela de contexto.
            
        Returns:
            str: Representação textual formatada do subgrafo recuperado.
        """
        if not keywords:
            return "Nenhuma palavra-chave válida foi identificada na pergunta."

        # Query Cypher para expansão de Subgrafo até 2 Saltos
        cypher_subgraph = """
        UNWIND $keywords AS kw
        MATCH (e:Entity)
        WHERE toLower(e.id) CONTAINS toLower(kw) OR toLower(e.type) CONTAINS toLower(kw)
        MATCH path = (e)-[r*1..2]-(neighbor:Entity)
        WITH path LIMIT $max_results
        UNWIND relationships(path) AS rel
        RETURN DISTINCT 
            startNode(rel).id AS source, 
            type(rel) AS relationship, 
            endNode(rel).id AS target
        """

        try:
            results = self.db_client.execute_query(
                cypher_subgraph,
                {"keywords": keywords, "max_results": max_results}
            )

            if not results:
                logger.warning(f"Nenhum subgrafo encontrado no Neo4j para os termos: {keywords}")
                return "Nenhum subgrafo ou relação correspondente foi encontrado na base de dados."

            # Formatar triplos em linguagem legível para o LLM
            formatted_triples = []
            for record in results:
                src = record.get("source")
                rel = record.get("relationship")
                tgt = record.get("target")
                formatted_triples.append(f"• ({src}) -[{rel}]-> ({tgt})")

            context_str = "\n".join(formatted_triples)
            logger.info(f"🔎 Recuperados {len(formatted_triples)} triplos de relações do Neo4j.")
            return context_str

        except Exception as e:
            logger.error(f"❌ Erro ao consultar subgrafo no Neo4j: {str(e)}")
            return "Erro técnico ao aceder ao Grafo de Conhecimento."

    def query(self, question: str) -> Dict[str, str]:
        """Executa a interrogação completa ao sistema GraphRAG.
        
        Args:
            question (str): A pergunta em linguagem natural do utilizador.
            
        Returns:
            Dict[str, str]: Dicionário contendo a resposta gerada e o contexto recuperado.
        """
        logger.info(f"❓ Nova pergunta recebida: '{question}'")

        # 1. Extrair termos e obter o subgrafo
        keywords = self._extract_keywords(question)
        graph_context = self._get_subgraph_context(keywords)

        # 2. Gerar resposta fundamentada
        response = self.chain.invoke({
            "graph_context": graph_context,
            "question": question,
        })

        return {
            "answer": str(response.content),
            "context_used": graph_context,
        }