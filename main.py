import sys
import os
from dotenv import load_dotenv

from src.database.neo4j_client import Neo4jClient
from src.nlp_extraction.extractor import GraphExtractor
from src.ingestion.pipeline import IngestionPipeline
from src.rag_chain.engine import GraphRAGEngine

load_dotenv()


def display_menu():
    print("\n" + "=" * 50)
    print("      🌐 GRAPHRAG ENGINE - INTERFACE CLI")
    print("=" * 50)
    print("1. 📄 Ingerir novo documento (PDF ou TXT)")
    print("2. ❓ Fazer uma pergunta ao Grafo (GraphRAG)")
    print("3. 🧹 Limpar toda a base de dados (Reset)")
    print("4. 🚪 Sair")
    print("=" * 50)


def main():
    print("🚀 A inicializar componentes do sistema...")

    # 1. Inicializar Conexão com Neo4j
    try:
        db_client = Neo4jClient()
    except Exception as e:
        print(f"❌ Erro crítico ao conectar ao Neo4j: {e}")
        sys.exit(1)

    # 2. Inicializar Módulos NLP e Pipeline
    extractor = GraphExtractor()
    pipeline = IngestionPipeline(db_client=db_client, extractor=extractor)
    rag_engine = GraphRAGEngine(db_client=db_client)

    print("✅ Sistema pronto e operacional!\n")

    while True:
        display_menu()
        choice = input("Escolhe uma opção (1-4): ").strip()

        if choice == "1":
            file_path = input("\nDigita o caminho do ficheiro (ex: documento_teste.txt): ").strip()
            if os.path.exists(file_path):
                try:
                    pipeline.process_file(file_path)
                    print(f"✨ Ingestão do ficheiro '{file_path}' concluída!")
                except Exception as e:
                    print(f"❌ Erro durante a ingestão: {e}")
            else:
                print(f"⚠️ Ficheiro não encontrado no caminho: {file_path}")

        elif choice == "2":
            question = input("\nPergunta: ").strip()
            if question:
                print("\n🔎 A consultar o Grafo de Conhecimento e a gerar resposta...")
                result = rag_engine.query(question)
                
                print("\n" + "-" * 50)
                print("📌 CONTEXTO EXTRAÍDO DO GRAFO:")
                print(result["context_used"])
                print("-" * 50)
                print("💡 RESPOSTA DO GRAPHRAG:")
                print(result["answer"])
                print("-" * 50)

        elif choice == "3":
            confirm = input("\nTens a certeza que queres APAGAR TODOS os nós e conexões? (s/n): ").strip().lower()
            if confirm == "s":
                db_client.clear_database()
                print("🧹 Base de dados limpa com sucesso.")

        elif choice == "4":
            print("\nA encerrar o GraphRAG Engine. Até à próxima!")
            break

        else:
            print("⚠️ Opção inválida. Tenta novamente.")


if __name__ == "__main__":
    main()