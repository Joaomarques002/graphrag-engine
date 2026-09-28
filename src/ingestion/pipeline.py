import os
import logging
from typing import List
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.database.neo4j_client import Neo4jClient
from src.nlp_extraction.extractor import GraphExtractor, KnowledgeGraph

load_dotenv()

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """Pipeline de ingestão responsável por ler ficheiros, dividi-los em chunks,
    extrair triplos RDF/Grafos e persisti-los no Neo4j."""

    def __init__(
        self,
        db_client: Neo4jClient,
        extractor: GraphExtractor,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> None:
        """Inicializa a pipeline com o cliente de BD, extrator NLP e parâmetros de chunking."""
        self.db_client = db_client
        self.extractor = extractor
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

    def load_and_split(self, file_path: str) -> List[str]:
        """Carrega um ficheiro (PDF ou TXT) e divide-o em blocos de texto.
        
        Args:
            file_path (str): Caminho para o ficheiro a processar.
            
        Returns:
            List[str]: Lista de blocos de texto (chunks).
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Ficheiro não encontrado: {file_path}")

        logger.info(f"📄 A carregar ficheiro: {file_path}")
        
        if file_path.lower().endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        else:
            loader = TextLoader(file_path, encoding="utf-8")

        documents = loader.load()
        chunks = self.splitter.split_documents(documents)
        
        chunk_texts = [chunk.page_content for chunk in chunks]
        logger.info(f"✂️ Ficheiro dividido em {len(chunk_texts)} blocos de texto.")
        return chunk_texts

    def _persist_graph(self, kg: KnowledgeGraph) -> None:
        """Persiste os nós e relações extraídos no Neo4j de forma idempotente.
        
        Args:
            kg (KnowledgeGraph): Objeto Pydantic com nós e relações.
        """
        # Cypher para persistência de Nós com MERGE (evita duplicados)
        cypher_node = """
        MERGE (n:Entity {id: $id})
        ON CREATE SET n.type = $type, n.name = $id, n += $properties
        ON MATCH SET n += $properties
        """

        # Cypher com APOC para criação de Relações dinâmicas
        cypher_rel = """
        MATCH (s:Entity {id: $source_id})
        MATCH (t:Entity {id: $target_id})
        CALL apoc.create.relationship(s, $rel_type, $properties, t) YIELD rel
        RETURN count(rel) as created_count
        """

        # 1. Persistir Nós
        for node in kg.nodes:
            try:
                self.db_client.execute_query(
                    cypher_node,
                    {
                        "id": node.id,
                        "type": node.type,
                        "properties": node.properties,
                    },
                )
            except Exception as e:
                logger.warning(f"⚠️ Falha ao guardar nó '{node.id}': {str(e)}")

        # 2. Persistir Relações
        for rel in kg.relationships:
            try:
                # Sanitizar o tipo de relação para garantir sintaxe válida Cypher
                clean_rel_type = "".join(
                    c if c.isalnum() or c == "_" else "_" for c in rel.type.upper()
                )
                
                self.db_client.execute_query(
                    cypher_rel,
                    {
                        "source_id": rel.source_id,
                        "target_id": rel.target_id,
                        "rel_type": clean_rel_type,
                        "properties": rel.properties,
                    },
                )
            except Exception as e:
                logger.warning(
                    f"⚠️ Falha ao criar relação ({rel.source_id})-[{rel.type}]->({rel.target_id}): {str(e)}"
                )

    def process_file(self, file_path: str) -> None:
        """Executa a ingestão completa de um documento.
        
        Args:
            file_path (str): Caminho do ficheiro a ingerir.
        """
        chunks = self.load_and_split(file_path)

        for idx, chunk_text in enumerate(chunks, 1):
            logger.info(f"⚙️ Processamento do bloco {idx}/{len(chunks)}...")
            
            # Extrair o grafo via LLM
            kg = self.extractor.extract(chunk_text)
            
            # Persistir no Neo4j
            self._persist_graph(kg)

        # Atualizar a cache do esquema do grafo
        self.db_client.refresh_schema()
        logger.info(f"✅ Ingestão do ficheiro '{file_path}' concluída com sucesso!")