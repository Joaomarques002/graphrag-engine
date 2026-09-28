# 🌐 GraphRAG Engine

Sistema robusto e modular de **Retrieval-Augmented Generation baseado em Grafos de Conhecimento (GraphRAG)**, construído com Python 3.10+, LangChain, Neo4j (via Docker) e Ollama (Llama 3 local).

---

## 🏛️ Arquitetura do Sistema

```text
                                 +--------------------+
                                 | Documento (PDF/TXT)|
                                 +---------+----------+
                                           |
                                           v
+------------------+             +--------------------+
| Ollama (Llama 3) | <---------- |  IngestionPipeline |
| (Extraction NLP) | --Triplos-> |   (Chunking & Map) |
+------------------+             +---------+----------+
                                           |
                                           v
                                 +--------------------+
                                 |   Neo4j Database   |
                                 |  (Graph + APOC)    |
                                 +---------+----------+
                                           | (Subgrafo 2-hops)
                                           v
+------------------+             +--------------------+
| Ollama (Llama 3) | <---------- |   GraphRAGEngine   |
|(Answer Synthesis)| ----------- |  (Context Enrich)  |
+------------------+             +--------------------+
```
---

## 🚀 Pré-requisitos

* Docker e Docker Compose instalados.
* Python 3.10+
* Ollama instalado localmente com o modelo **llama3**:

```bash
ollama pull llama3
ollama serve
```
---

## 🛠️ Instalação e Configuração

### 1. Clonar o Repositório:

```bash
git clone [https://github.com/](https://github.com/)<O-TEU-UTILIZADOR>/graphrag-engine.git
cd graphrag-engine
```
### 2. Criar e Ativar Ambiente Virtual:

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# ou .\venv\Scripts\Activate.ps1 no Windows
```
### 3. Instalar Dependências:

```bash
pip install -r requirements.txt
```
### 4. Configurar Variáveis de Ambiente:
Criar o ficheiro **.env** baseado no **.env.example**:

```Ini,TOMIL
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password123
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3
```
### 5. Iniciar a Base de Dados Neo4j:

```Bash
docker compose up -d
```
---

## 💻 Como Utilizar

Executa o ponto de entrada principal:

```Bash
python main.py
```
A interface interativa permite:

1. Ingerir documentos e extrair entidades automaticamente para o Neo4j.
2. Fazer perguntas em linguagem natural navegando no subgrafo de até 2 saltos.
3. Gerir e limpar a base de dados.

---