# ai4agro_server.py — coleta (RSS + fallback), clusterização semântica, ontologia, grafo e hipótese curta (≤20 palavras)
import os
import json
import time
from json import JSONDecodeError
from typing import List, Dict, Any

import requests
import feedparser
from bs4 import BeautifulSoup
from flask import Flask, render_template, jsonify, request

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__, static_folder="static", template_folder="templates")

DATA_PATH = os.path.join("data", "sinais.json")
ONTO_PATH = os.path.join("data", "ontologia.json")

# ========= Fontes públicas =========
RSS_FONTES = [
    "https://www.embrapa.br/busca-de-noticias/-/busca/feed/rss/1/noticias",
    "https://www.gov.br/agricultura/pt-br/assuntos/noticias/@@RSS",
    "https://valor.globo.com/agronegocios/rss.xml",
    "https://revistagloborural.globo.com/rss/ultimas/feed.xml",
    "https://www.canalrural.com.br/feed/",
    "https://www.noticiasagricolas.com.br/rss",
    "https://www.agrolink.com.br/rss/ultimas.xml",
]

HTTP_OPTS = dict(timeout=10, headers={
    "User-Agent": "Mozilla/5.0 (AI4AgroFuture; +https://example.org)",
    "Accept": "application/rss+xml,application/xml,text/xml;q=0.9,*/*;q=0.8",
})

# ========= Util: E/S segura =========
def nrm(txt: str) -> str:
    return " ".join((txt or "").split()).strip()

def carregar_json(path: str, default):
    try:
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            return default
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (JSONDecodeError, OSError, ValueError):
        return default

def salvar_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def carregar_sinais() -> List[Dict[str, Any]]:
    return carregar_json(DATA_PATH, [])

def salvar_sinais(sinais: List[Dict[str, Any]]) -> None:
    salvar_json(DATA_PATH, sinais)

def carregar_ontologia() -> Dict[str, Any]:
    onto = carregar_json(ONTO_PATH, {"conceitos": []})
    # indexa rapidamente
    for c in onto.get("conceitos", []):
        c["keywords_lc"] = [k.lower() for k in c.get("keywords", [])]
    return onto

# ========= Coleta =========
def parse_rss(url: str) -> List[Dict[str, str]]:
    itens = []
    try:
        r = requests.get(url, **HTTP_OPTS)
        r.raise_for_status()
        feed = feedparser.parse(r.content)
    except Exception:
        feed = feedparser.parse(url)
    for e in getattr(feed, "entries", []) or []:
        titulo = nrm(getattr(e, "title", ""))
        link = getattr(e, "link", "")
        if titulo and link:
            itens.append({"titulo": titulo, "fonte": link})
    return itens

def fallback_html_list(url: str, selector: str, attr: str = "href") -> List[Dict[str, str]]:
    try:
        r = requests.get(url, timeout=10, headers={"User-Agent": HTTP_OPTS["headers"]["User-Agent"]})
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        itens = []
        for a in soup.select(selector)[:25]:
            titulo = nrm(a.get_text())
            link = a.get(attr)
            if titulo and link and link.startswith("http"):
                itens.append({"titulo": titulo, "fonte": link})
        return itens
    except Exception:
        return []

def gerar_sinais_automaticos(max_itens: int = 48) -> List[Dict[str, Any]]:
    sinais = []
    vistos = set()
    # 1) RSS
    for rss in RSS_FONTES:
        for item in parse_rss(rss):
            if item["fonte"] in vistos:
                continue
            vistos.add(item["fonte"])
            item["ts"] = int(time.time())
            sinais.append(item)
            if len(sinais) >= max_itens:
                return sinais
    # 2) Fallback HTML se necessário
    if not sinais:
        sinais.extend(fallback_html_list(
            "https://www.gov.br/agricultura/pt-br/assuntos/noticias",
            "a[href*='/assuntos/noticias/']"
        ))
        sinais.extend(fallback_html_list(
            "https://www.embrapa.br/busca-de-noticias",
            "a.nome-noticia, a.card-title, h3 a"
        ))
        arr, vistos = [], set()
        for it in sinais:
            if it["fonte"] not in vistos:
                vistos.add(it["fonte"])
                it["ts"] = int(time.time())
                arr.append(it)
                if len(arr) >= max_itens:
                    break
        sinais = arr
    return sinais

# ========= Ontologia: tagging por keywords =========
def taggear_por_ontologia(sinais: List[Dict[str, Any]], onto: Dict[str, Any]) -> None:
    for s in sinais:
        t = s.get("titulo", "").lower()
        tags = []
        for c in onto.get("conceitos", []):
            if any(k in t for k in c.get("keywords_lc", [])):
                tags.append(c["nome"])
        s["conceitos"] = sorted(list(set(tags)))

# ========= Clusterização semântica (TF-IDF + cosseno) =========
def clusterizar(sinais: List[Dict[str, Any]], threshold: float = 0.24):
    """
    Retorna:
      - clusters: lista de listas de índices
      - sim_edges: arestas (i,j,score) acima do threshold
    """
    titulos = [s["titulo"] for s in sinais]
    if len(titulos) < 2:
        return [[i for i in range(len(titulos))]], []

    vec = TfidfVectorizer(
        strip_accents="unicode",
        lowercase=True,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95
    )
    X = vec.fit_transform(titulos)
    S = cosine_similarity(X)
    # grafo de similaridade
    n = S.shape[0]
    visited = [False]*n
    edges = []
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if S[i, j] >= threshold:
                edges.append((i, j, float(S[i, j])))
                adj[i].append(j)
                adj[j].append(i)
    # componentes conexas
    clusters = []
    for i in range(n):
        if not visited[i]:
            stack = [i]
            comp = []
            visited[i] = True
            while stack:
                u = stack.pop()
                comp.append(u)
                for v in adj[u]:
                    if not visited[v]:
                        visited[v] = True
                        stack.append(v)
            clusters.append(comp)
    # ordena por tamanho
    clusters.sort(key=len, reverse=True)
    return clusters, edges

# ========= Hipótese curta (≤20 palavras, sem inventar texto) =========
def gerar_hipotese_curta(sinais_cluster: List[Dict[str, Any]]) -> str:
    """
    Cria frase curta usando apenas títulos reais do cluster (não inventa).
    Regras:
      - pega até 3 títulos mais representativos e compõe:
        "Tendências poderão convergir entre: 't1', 't2' e 't3'."
      - garante ≤ 20 palavras (corta títulos se necessário).
    """
    if not sinais_cluster:
        return "Sem dados suficientes para hipótese."
    # pega 3 títulos distintos
    ts = [s["titulo"] for s in sinais_cluster[:3]]
    # redução simples de comprimento (palavras demais)
    def encurta(txt, max_pal=6):
        w = txt.split()
        return " ".join(w[:max_pal]) + ("…" if len(w) > max_pal else "")
    ts = [encurta(t) for t in ts]

    if len(ts) == 1:
        frase = f"Tendências poderão convergir a partir de: '{ts[0]}'."
    elif len(ts) == 2:
        frase = f"Tendências poderão convergir entre: '{ts[0]}' e '{ts[1]}'."
    else:
        frase = f"Tendências poderão convergir entre: '{ts[0]}', '{ts[1]}' e '{ts[2]}'."

    # segurança ≤ 20 palavras
    if len(frase.split()) > 20:
        frase = " ".join(frase.split()[:20]) + "…"
    return frase

# ========= Rotas =========
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/dashboard")
def dashboard():
    sinais = carregar_sinais()
    if not sinais:
        sinais = gerar_sinais_automaticos()
        if sinais:
            salvar_sinais(sinais)
    onto = carregar_ontologia()
    taggear_por_ontologia(sinais, onto)
    return render_template("dashboard.html", sinais=sinais)

@app.route("/api/refresh_sinais", methods=["POST"])
def api_refresh_sinais():
    novos = gerar_sinais_automaticos()
    if novos:
        onto = carregar_ontologia()
        taggear_por_ontologia(novos, onto)
        salvar_sinais(novos)
    return jsonify({"ok": True, "qtd": len(novos)})

@app.route("/api/sinais")
def api_sinais():
    return jsonify(carregar_sinais())

@app.route("/api/graph")
def api_graph():
    """
    Retorna o grafo do MAIOR cluster (6–12 nós se possível) e a hipótese curta.
    Formato:
      {
        "titulo": "Cluster de Cenários Antecipativo",
        "hipotese": "...",
        "nodes":[{"id":"hipotese","label":"...", "tipo":"hip"}, {"id":"s0","label":"...", "fonte":"...", "conceitos":[...]}],
        "edges":[{"source":"hipotese","target":"s0"}, {"source":"s0","target":"s1"}]
      }
    """
    sinais = carregar_sinais()
    onto = carregar_ontologia()
    taggear_por_ontologia(sinais, onto)

    if not sinais:
        return jsonify({"titulo": "Cluster de Cenários Antecipativo", "hipotese": "Sem dados.", "nodes": [], "edges": []})

    clusters, edges = clusterizar(sinais)
    top = clusters[0] if clusters else list(range(len(sinais)))
    # reduz o cluster para 6–12 itens
    if len(top) > 12:
        top = top[:12]
    elif len(top) < 6 and len(clusters) > 1:
        # tenta complementar com próximo cluster
        for c in clusters[1:]:
            for idx in c:
                if idx not in top:
                    top.append(idx)
                    if len(top) >= 6:
                        break
            if len(top) >= 6:
                break

    cluster_sinais = [sinais[i] for i in top]
    hipotese = gerar_hipotese_curta(cluster_sinais)

    # nós e arestas: hipotese central (azul) + sinais (laranja); liga hipotese a todos; liga sinais similares (do edges)
    nodes = [{"id": "hipotese", "label": hipotese, "tipo": "hip"}]
    for k, s in enumerate(cluster_sinais):
        nodes.append({
            "id": f"s{k}",
            "label": s["titulo"],
            "fonte": s.get("fonte", ""),
            "conceitos": s.get("conceitos", []),
            "tipo": "sinal"
        })

    # mapeia índice absoluto -> id local
    id_map = {abs_idx: f"s{loc}" for loc, abs_idx in enumerate(top)}
    graph_edges = []
    # liga hipotese a todos os sinais
    for loc_id in range(len(cluster_sinais)):
        graph_edges.append({"source": "hipotese", "target": f"s{loc_id}"})
    # liga sinais com similaridade > threshold dentro do subgrafo
    edge_set = set()
    for (i, j, _) in edges:
        if i in top and j in top:
            a, b = id_map[i], id_map[j]
            key = tuple(sorted([a, b]))
            if key not in edge_set:
                edge_set.add(key)
                graph_edges.append({"source": a, "target": b})

    return jsonify({
        "titulo": "Cluster de Cenários Antecipativo",
        "hipotese": hipotese,
        "nodes": nodes,
        "edges": graph_edges
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001, debug=True)

