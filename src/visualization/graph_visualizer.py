import tempfile
import os
from typing import List, Dict
from pyvis.network import Network


def generate_subgraph_html(triples: List[Dict[str, str]], height: str = "450px") -> str:
    """Converte uma lista de triplos recuperados num grafo HTML interativo Pyvis.
    
    Args:
        triples: Lista de dicionarios com chaves 'source', 'relationship' e 'target'.
        height: Altura do canvas do grafo em pixels.
        
    Returns:
        str: Conteudo HTML completo do grafo interativo.
    """
    if not triples:
        return "<p style='color: gray;'>Nenhum subgrafo para renderizar.</p>"

    # Inicializar rede dirigida com fundo escuro elegante
    net = Network(
        height=height,
        width="100%",
        bgcolor="#1e1e2e",
        font_color="#ffffff",
        directed=True
    )

    # Configuração da física para estabilidade dos nós
    net.force_atlas_2based(gravity=-50, central_gravity=0.01, spring_length=100)

    nodes = set()

    for triple in triples:
        src = triple.get("source")
        rel = triple.get("relationship")
        tgt = triple.get("target")

        if src and src not in nodes:
            net.add_node(
                src,
                label=src,
                title=f"Entidade: {src}",
                color="#89b4fa",  # Azul pastel
                shape="dot",
                size=20
            )
            nodes.add(src)

        if tgt and tgt not in nodes:
            net.add_node(
                tgt,
                label=tgt,
                title=f"Entidade: {tgt}",
                color="#a6e3a1",  # Verde pastel
                shape="dot",
                size=20
            )
            nodes.add(tgt)

        if src and tgt and rel:
            net.add_edge(
                src,
                tgt,
                title=f"Relação: {rel}",
                label=rel,
                color="#cdd6f4",
                arrows="to"
            )

    # Gerar HTML temporário para leitura
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode="w", encoding="utf-8") as tmp:
        net.save_graph(tmp.name)
        tmp_path = tmp.name

    with open(tmp_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    os.remove(tmp_path)
    return html_content