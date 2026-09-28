import os
import logging
from typing import Dict, List, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

load_dotenv()

logger = logging.getLogger(__name__)


# ==========================================
# 1. Esquemas Pydantic para Saída Estruturada
# ==========================================

class Node(BaseModel):
    """Modelo que representa uma entidade/nó no Grafo de Conhecimento."""
    id: str = Field(
        ...,
        description="Identificador único e canonizado da entidade em Snake_Case ou CamelCase (ex: 'Apple_Inc', 'Sam_Altman')."
    )
    type: str = Field(
        ...,
        description="Tipo ou categoria formal da entidade em maiúsculas (ex: 'ORGANIZATION', 'PERSON', 'LOCATION', 'TECHNOLOGY', 'CONCEPT')."
    )
    properties: Dict[str, str] = Field(
        default_factory=dict,
        description="Propriedades ou atributos adicionais relevantes extraídos do texto."
    )


class Relationship(BaseModel):
    """Modelo que representa uma conexão/aresta direcionada no Grafo."""
    source_id: str = Field(
        ..., 
        description="ID da entidade de origem (deve corresponder exatamente ao 'id' de um Node)."
    )
    target_id: str = Field(
        ..., 
        description="ID da entidade de destino (deve corresponder exatamente ao 'id' de um Node)."
    )
    type: str = Field(
        ...,
        description="Tipo da relação em UPPERCASE_SNAKE_CASE (ex: 'FOUNDED_BY', 'LOCATED_IN', 'DEVELOPED', 'HAS_METRIC')."
    )
    properties: Dict[str, str] = Field(
        default_factory=dict,
        description="Propriedades contextuais da relação (ex: data, valor, estado)."
    )


class KnowledgeGraph(BaseModel):
    """Contentor principal para o Grafo de Conhecimento extraído."""
    nodes: List[Node] = Field(
        default_factory=list, 
        description="Lista de todas as entidades identificadas no texto."
    )
    relationships: List[Relationship] = Field(
        default_factory=list, 
        description="Lista de todas as conexões entre as entidades."
    )


# ==========================================
# 2. Prompt de Sistema de Alta Precisão
# ==========================================

EXTRACTION_SYSTEM_PROMPT = """Tu és um engenheiro de conhecimento especialista em extração de Grafos de Conhecimento de precisão industrial.
A tua missão é analisar o texto fornecido e extrair TODAS as entidades relevantes e as suas relações diretas.

Regras Estritas de Extração:
1. Normalização de IDs: Converte nomes de entidades para identificadores únicos e consistentes (ex: "Empresa Apple" -> "Apple_Inc", "São Francisco" -> "San_Francisco").
2. Tipos de Entidade: Usa categorias formais e em maiúsculas (ex: PERSON, ORGANIZATION, LOCATION, PRODUCT, EVENT, METRIC).
3. Tipos de Relação: Expressa as relações obrigatoriamente em UPPERCASE_SNAKE_CASE (ex: FOUNDED_BY, HEADQUARTERED_IN, PRODUCED_BY, AFFECTS).
4. Integridade Referencial: Todo o 'source_id' e 'target_id' especificado numa relação DEVE estar presente na lista de 'nodes'.
5. Não inventes factos fora do texto. Limita-te estritamente ao conteúdo fornecido.
"""


# ==========================================
# 3. Classe do Extrator NLP
# ==========================================

class GraphExtractor:
    """Módulo responsável por ler blocos de texto e gerar estruturas Pydantic via Ollama."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
    ) -> None:
        """Inicializa o modelo Ollama com saída estruturada obrigatória."""
        self.model_name = model_name or os.getenv("OLLAMA_MODEL", "llama3")
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        
        logger.info(f"🤖 A inicializar GraphExtractor com o modelo '{self.model_name}' em {self.base_url}")

        # Instanciar o LLM Ollama
        self.llm = ChatOllama(
            model=self.model_name,
            base_url=self.base_url,
            temperature=temperature, # 0.0 para garantir previsibilidade
        )

        # Configurar o Prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", EXTRACTION_SYSTEM_PROMPT),
            ("human", "Analisa o seguinte texto e extrai o Grafo de Conhecimento:\n\n{text_chunk}"),
        ])

        # Vincular o modelo com validação Pydantic forçada
        self.chain = self.prompt | self.llm.with_structured_output(KnowledgeGraph)

    def extract(self, text_chunk: str) -> KnowledgeGraph:
        """Processa um bloco de texto e devolve o objeto KnowledgeGraph validado.
        
        Args:
            text_chunk (str): O fragmento de texto a analisar.
            
        Returns:
            KnowledgeGraph: Objeto Pydantic contendo os nós e as arestas extraídos.
        """
        if not text_chunk.strip():
            logger.warning("Bloco de texto vazio recebido para extração.")
            return KnowledgeGraph()

        try:
            result = self.chain.invoke({"text_chunk": text_chunk})
            
            if isinstance(result, KnowledgeGraph):
                logger.info(
                    f"✨ Extração concluída: {len(result.nodes)} nós e "
                    f"{len(result.relationships)} relações encontradas."
                )
                return result
            
            logger.warning("O modelo não devolveu uma instância válida de KnowledgeGraph. A usar fallback.")
            return KnowledgeGraph()

        except Exception as e:
            logger.error(f"❌ Erro durante a extração NLP com o Ollama: {str(e)}")
            # Falha graciosa: devolve grafo vazio para não quebrar o pipeline de ingestão
            return KnowledgeGraph()