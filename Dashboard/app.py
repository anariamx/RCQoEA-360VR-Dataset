import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

PASTA = Path(__file__).resolve().parent / "dados"
RAIZ_BRUTOS = Path(os.environ.get("RCQOE_DIR", Path(__file__).resolve().parent.parent))

VARIAVEIS = {
    "MOS": ("QoE", "Nota pós-vídeo (1–5)"),
    "QoEC_media": ("QoE", "Média da QoE contínua (sem zeros iniciais)"),
    "QoEC_final": ("QoE", "QoEC média nos últimos 3 s"),
    "QoEC_min": ("QoE", "Menor valor da QoEC"),
    "QoEC_max": ("QoE", "Maior valor da QoEC"),
    "QoEC_dp": ("QoE", "Desvio-padrão da QoEC na sessão"),
    "QoEC_tendencia": ("QoE", "Inclinação da QoEC (unidades por 10 s)"),
    "QoEC_pct_anotada": ("QoE", "% do vídeo com QoEC anotada"),
    "FC_media": ("HRV", "Frequência cardíaca média (bpm)"),
    "RR_medio": ("HRV", "Intervalo RR médio (ms)"),
    "SDNN": ("HRV", "Desvio-padrão dos intervalos RR (ms)"),
    "RMSSD": ("HRV", "Raiz da média das diferenças sucessivas de RR (ms)"),
    "pNN50": ("HRV", "% de diferenças sucessivas de RR > 50 ms"),
    "SD1": ("HRV", "Poincaré SD1 (ms) — variabilidade de curto prazo"),
    "SD2": ("HRV", "Poincaré SD2 (ms) — variabilidade de longo prazo"),
    "FC_inicio": ("FC no tempo", "FC média nos primeiros 5 s (bpm)"),
    "FC_fim": ("FC no tempo", "FC média nos últimos 5 s (bpm)"),
    "FC_delta": ("FC no tempo", "FC fim − início (bpm)"),
    "FC_tendencia": ("FC no tempo", "Inclinação da FC (bpm por 10 s)"),
    "SC_media": ("GSR", "Condutância da pele média (µS)"),
    "SC_delta": ("GSR", "Condutância fim − início (µS)"),
    "n_SCR": ("GSR", "Número de respostas fásicas (SCR)"),
    "vel_cabeca_media": ("Cabeça", "Velocidade angular média da cabeça (°/s)"),
    "ordem": ("Sessão", "Posição do vídeo na sequência do participante (1–40)"),
    "qp": ("Sessão", "QP da configuração (hipótese de mapeamento)"),
}
QUESTIONARIOS = {
    "IPQ_total": "IPQ — média das quatro subescalas (−3 a 3)",
    "IPQ_G": "IPQ — presença geral",
    "IPQ_SP": "IPQ — presença espacial",
    "IPQ_INV": "IPQ — envolvimento",
    "IPQ_REAL": "IPQ — realismo",
    "TLX_bruto": "NASA-TLX bruto (média das 6 subescalas, 0–100)",
    "TLX_Mental": "NASA-TLX — demanda mental",
    "TLX_Physical": "NASA-TLX — demanda física",
    "TLX_Temporal": "NASA-TLX — demanda temporal",
    "TLX_Effort": "NASA-TLX — esforço",
    "TLX_Performance": "NASA-TLX — desempenho",
    "TLX_Frustration": "NASA-TLX — frustração",
    "SSQ_total": "SSQ total ao fim do bloco",
    "SSQ_delta": "SSQ do bloco − SSQ pré-teste",
}
SERIES = {"QoEC": "QoE contínua", "FC": "FC instantânea (bpm)", "SC": "Condutância da pele (µS)", "vel_cabeca": "Velocidade da cabeça (°/s)"}
NIVEIS = ["Por sessão", "Por participante × bloco", "Por participante"]
CORES = {"Participante": "participante", "Configuração": "configuracao", "Conteúdo": "conteudo",
         "Bloco": "bloco", "MOS": "MOS", "Nenhuma": None}

st.set_page_config(page_title="RCQoEA-360VR Explorer", page_icon="🥽", layout="wide")


@st.cache_data
def carregar():
    sessoes = pd.read_csv(PASTA / "sessoes.csv")
    series = pd.read_csv(PASTA / "series.csv")
    quest = pd.read_csv(PASTA / "questionarios.csv")
    perfil = pd.read_csv(PASTA / "participantes.csv")
    return sessoes, series, quest, perfil


def descricao(nome):
    if nome in VARIAVEIS:
        return VARIAVEIS[nome][1]
    return QUESTIONARIOS.get(nome, nome)


def observacoes(sessoes, quest, nivel):
    numericas = [v for v in VARIAVEIS if v in sessoes]
    if nivel == "Por sessão":
        return sessoes.merge(quest, on=["participante", "bloco"], how="left")
    if nivel == "Por participante × bloco":
        tab = sessoes.groupby(["participante", "bloco"])[numericas].mean().reset_index()
        return tab.merge(quest, on=["participante", "bloco"], how="left")
    tab = sessoes.groupby("participante")[numericas].mean().reset_index()
    q = quest.drop(columns="bloco").groupby("participante").mean().reset_index()
    return tab.merge(q, on="participante", how="left")


def centralizar(tabela, colunas):
    out = tabela.copy()
    grupos = out.groupby("participante")[colunas]
    out[colunas] = (out[colunas] - grupos.transform("mean")) / grupos.transform("std")
    return out


def corr_intra(tabela, x, y):
    d = tabela[["participante", x, y]].dropna()
    d[[x, y]] = d[[x, y]] - d.groupby("participante")[[x, y]].transform("mean")
    k = d["participante"].nunique()
    gl = len(d) - k - 1
    if gl < 3 or d[x].std() == 0 or d[y].std() == 0:
        return np.nan, np.nan, gl
    r = np.corrcoef(d[x], d[y])[0, 1]
    t = r * np.sqrt(gl / (1 - r ** 2))
    return r, 2 * stats.t.sf(abs(t), gl), gl


def baixar(tabela, nome, chave):
    st.download_button("Baixar CSV", tabela.to_csv(index=False).encode("utf-8"), f"{nome}.csv", "text/csv", key=chave)


def curva_media(dados, coluna, grupo):
    resumo = dados.groupby([grupo, "t"])[coluna].agg(["mean", "std", "count"]).reset_index()
    resumo["ic"] = 1.96 * resumo["std"] / np.sqrt(resumo["count"])
    fig = go.Figure()
    for i, (nome, g) in enumerate(resumo.groupby(grupo)):
        cor = px.colors.qualitative.Plotly[i % 10]
        fig.add_trace(go.Scatter(x=np.r_[g.t, g.t[::-1]], y=np.r_[g["mean"] + g.ic, (g["mean"] - g.ic)[::-1]],
                                 fill="toself", fillcolor=cor, opacity=0.15, line_width=0, showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=g.t, y=g["mean"], name=str(nome), line=dict(color=cor, width=2)))
    return fig, resumo


sessoes_todas, series_todas, quest, perfil = carregar()

st.title("RCQoEA-360VR · explorador científico")
st.caption("Explore QoE contínua (QoEC), nota pós-vídeo (MOS), ECG/HRV, GSR, movimento de cabeça e questionários por bloco. "
           "Valores fisiológicos descrevem sinais; não são diagnósticos.")

with st.sidebar:
    st.header("Seleção global dos dados")
    pessoas = sorted(sessoes_todas.participante.unique())
    configs = list(dict.fromkeys(sessoes_todas.sort_values("VideoID").configuracao))
    conteudos = sorted(sessoes_todas.conteudo.unique())
    uniformes = [c for c in configs if c.count(" ") == 0]
    if "sel_pessoas" not in st.session_state:
        st.session_state.update(sel_pessoas=pessoas, sel_configs=uniformes, sel_conteudos=conteudos, sel_blocos=[1, 2])
    st.multiselect("Participantes", pessoas, key="sel_pessoas", format_func=lambda p: f"P{p}")
    st.button("Todos os participantes", on_click=lambda: st.session_state.update(sel_pessoas=pessoas))
    st.multiselect("Configurações de qualidade", configs, key="sel_configs")
    c1, c2 = st.columns(2)
    c1.button("Só uniformes", on_click=lambda: st.session_state.update(sel_configs=uniformes))
    c2.button("Todas", on_click=lambda: st.session_state.update(sel_configs=configs))
    st.multiselect("Conteúdos", conteudos, key="sel_conteudos",
                   format_func=lambda c: f"{c} · {sessoes_todas.loc[sessoes_todas.conteudo == c, 'nome_conteudo_hipotese'].iloc[0]}?")
    st.multiselect("Blocos", [1, 2], key="sel_blocos")
    sem_anotacao = st.toggle("Excluir sessões sem QoEC anotada", value=True)
    st.caption("Mapeamento VideoID → conteúdo/configuração é uma **hipótese** (arquivo `mapeamento_videos.csv`). "
               "Confirmar com os autores do dataset.")

filtro = (sessoes_todas.participante.isin(st.session_state.sel_pessoas)
          & sessoes_todas.configuracao.isin(st.session_state.sel_configs)
          & sessoes_todas.conteudo.isin(st.session_state.sel_conteudos)
          & sessoes_todas.bloco.isin(st.session_state.sel_blocos))
if sem_anotacao:
    filtro &= sessoes_todas.QoEC_pct_anotada > 0
sessoes = sessoes_todas[filtro]
series = series_todas.merge(sessoes[["participante", "VideoID", "configuracao", "conteudo", "bloco", "MOS", "qp"]],
                            on=["participante", "VideoID"])

st.caption(f"{sessoes.participante.nunique()} participantes · {sessoes.VideoID.nunique()} vídeos · "
           f"{len(sessoes)} sessões na seleção")
if sessoes.empty:
    st.info("Nenhuma sessão corresponde à seleção. Ajuste os filtros na barra lateral.")
    st.stop()

abas = st.tabs(["Sessão", "QoEC × MOS", "Relações", "Correlações", "Questionários", "PCA"])

with abas[0]:
    st.header("Sessão: sinais e QoE")
    st.caption("Cada sessão é um participante assistindo a um vídeo de 30 s. As curvas estão numa grade comum de 0,25 s.")
    opcoes = list(sessoes.sort_values(["participante", "ordem"])[["participante", "VideoID"]].itertuples(index=False, name=None))
    p, v = st.selectbox("Sessão em foco", opcoes, format_func=lambda s: f"P{s[0]} · V{s[1]}")
    linha = sessoes[(sessoes.participante == p) & (sessoes.VideoID == v)].iloc[0]
    outros = st.multiselect("Sobrepor outros participantes (mesmo vídeo)",
                            sorted(set(sessoes.loc[sessoes.VideoID == v, "participante"]) - {p}), format_func=lambda x: f"P{x}")
    mostrar_bruta = st.toggle("Mostrar QoEC bruta (com zeros iniciais)", value=False)

    m = st.columns(6)
    m[0].metric("MOS", f"{linha.MOS:.0f}")
    m[1].metric("QoEC média", f"{linha.QoEC_media:.2f}")
    m[2].metric("FC média", f"{linha.FC_media:.1f} bpm")
    m[3].metric("RMSSD", f"{linha.RMSSD:.1f} ms")
    m[4].metric("Configuração", linha.configuracao)
    m[5].metric("Ordem · bloco", f"{linha.ordem} · B{linha.bloco}")

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.04, subplot_titles=list(SERIES.values()))
    for i, pessoa in enumerate([p, *outros]):
        s = series[(series.participante == pessoa) & (series.VideoID == v)]
        cor = px.colors.qualitative.Plotly[i % 10]
        for linha_fig, col in enumerate(SERIES, start=1):
            fig.add_trace(go.Scatter(x=s.t, y=s[col], name=f"P{pessoa}", legendgroup=str(pessoa), showlegend=linha_fig == 1,
                                     line=dict(color=cor, width=2 if pessoa == p else 1.2)), row=linha_fig, col=1)
        if mostrar_bruta:
            fig.add_trace(go.Scatter(x=s.t, y=s.QoEC_bruta, name=f"P{pessoa} bruta", legendgroup=str(pessoa), showlegend=False,
                                     line=dict(color=cor, dash="dot", width=1)), row=1, col=1)
    fig.add_hline(y=(linha.MOS - 1) / 4, line_dash="dash", line_color="gray", row=1, col=1,
                  annotation_text=f"MOS P{p} = {linha.MOS:.0f} (reescalado (MOS−1)/4)")
    fig.update_layout(height=820, title=f"P{p} · V{v} · {linha.configuracao} · {linha.conteudo}", hovermode="x unified")
    fig.update_xaxes(title_text="tempo (s)", row=4, col=1)
    st.plotly_chart(fig)

    with st.expander("Características da sessão"):
        colunas = [c for c in VARIAVEIS if c in sessoes]
        st.dataframe(pd.DataFrame({"variável": colunas, "valor": linha[colunas].astype(float).round(3).values,
                                   "descrição": [descricao(c) for c in colunas]}), hide_index=True)

    with st.expander("ECG bruto com picos R (lê o arquivo original)"):
        arquivo = RAIZ_BRUTOS / "Physiological_Data/ECG/Raw_ECG_data" / f"ecg_P{p}_V{v}.csv"
        if not arquivo.exists():
            st.info(f"Arquivo não encontrado: {arquivo}. Defina a variável RCQOE_DIR com a pasta do dataset.")
        elif st.toggle("Carregar ECG", key="carregar_ecg"):
            import neurokit2 as nk
            ecg = pd.read_csv(arquivo)
            sinal = -ecg.ECG.to_numpy(float) if linha.ecg_invertido else ecg.ECG.to_numpy(float)
            limpo = nk.ecg_clean(sinal, sampling_rate=130)
            picos = nk.ecg_peaks(limpo, sampling_rate=130, correct_artifacts=True)[1]["ECG_R_Peaks"]
            t = np.arange(len(limpo)) / 130
            fig_ecg = go.Figure([go.Scatter(x=t, y=limpo, name="ECG limpo", line_width=1),
                                 go.Scatter(x=t[picos], y=limpo[picos], mode="markers", name="picos R")])
            fig_ecg.update_layout(height=320, xaxis_title="tempo (s)", yaxis_title="µV",
                                  title=f"Polaridade {'invertida (corrigida)' if linha.ecg_invertido else 'original'} · {len(picos)} picos R")
            st.plotly_chart(fig_ecg)

with abas[1]:
    st.header("QoEC × MOS")
    st.caption("Curvas médias entre sessões com intervalo de confiança de 95%. Use para ver como a QoE contínua se forma "
               "até virar a nota final e se a fisiologia acompanha.")
    c = st.columns(3)
    var_tempo = c[0].selectbox("Variável no tempo", list(SERIES), format_func=lambda x: SERIES[x])
    grupo = c[1].selectbox("Agrupar curvas por", ["configuracao", "MOS", "conteudo", "bloco"],
                           format_func={"configuracao": "Configuração", "MOS": "MOS dado", "conteudo": "Conteúdo", "bloco": "Bloco"}.get)
    ajuste = c[2].selectbox("Ajuste", ["Original", "Subtrair linha de base (primeiros 3 s)", "Centralizar por participante"])
    dados_t = series.dropna(subset=[var_tempo]).copy()
    if ajuste.startswith("Subtrair"):
        base = dados_t[dados_t.t < 3].groupby(["participante", "VideoID"])[var_tempo].mean().rename("base")
        dados_t = dados_t.join(base, on=["participante", "VideoID"])
        dados_t[var_tempo] -= dados_t["base"]
    elif ajuste.startswith("Centralizar"):
        dados_t[var_tempo] -= dados_t.groupby("participante")[var_tempo].transform("mean")
    fig, resumo = curva_media(dados_t, var_tempo, grupo)
    fig.update_layout(height=430, xaxis_title="tempo (s)", yaxis_title=SERIES[var_tempo], title=f"{SERIES[var_tempo]} média por {grupo}")
    st.plotly_chart(fig)
    baixar(resumo, f"curva_{var_tempo}_{grupo}", "baixar_curva")

    e, d = st.columns(2)
    with e:
        fig = px.box(sessoes, x="configuracao", y="MOS", points="all", color="configuracao",
                     category_orders={"configuracao": configs}, title="MOS por configuração")
        fig.update_layout(showlegend=False, height=420)
        st.plotly_chart(fig)
    with d:
        resumo_qoe = st.selectbox("Resumo da QoEC", [v for v in VARIAVEIS if v.startswith("QoEC")])
        fig = px.box(sessoes, x="MOS", y=resumo_qoe, points="all", title=f"{resumo_qoe} por MOS dado")
        rho, pv = stats.spearmanr(sessoes.MOS, sessoes[resumo_qoe], nan_policy="omit")
        r_in, p_in, _ = corr_intra(sessoes, "MOS", resumo_qoe)
        fig.update_layout(height=420)
        st.plotly_chart(fig)
        st.caption(f"Spearman ρ = {rho:.2f} (p = {pv:.3g}) · correlação intraparticipante r = {r_in:.2f} (p = {p_in:.3g})")

    st.subheader("QoEC × FC no tempo, sessão a sessão")
    st.caption("Para cada sessão, correlação de Spearman entre QoEC(t) e a variável fisiológica deslocada. "
               "Defasagem positiva = a fisiologia vem depois da QoEC.")
    c = st.columns(2)
    fisio = c[0].selectbox("Sinal fisiológico", ["FC", "SC", "vel_cabeca"], format_func=lambda x: SERIES[x])
    defasagem = c[1].slider("Defasagem (s)", -6.0, 6.0, 0.0, 0.25)
    passos = int(round(defasagem / 0.25))
    linhas_rho = []
    for (pp, vv), g in series.groupby(["participante", "VideoID"]):
        y = g[fisio].shift(-passos).to_numpy()
        x = g["QoEC"].to_numpy()
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() > 20 and np.std(x[ok]) > 0 and np.std(y[ok]) > 0:
            linhas_rho.append((pp, vv, g.configuracao.iloc[0], stats.spearmanr(x[ok], y[ok])[0]))
    rhos = pd.DataFrame(linhas_rho, columns=["participante", "VideoID", "configuracao", "rho"])
    if len(rhos) > 5:
        fig = px.histogram(rhos, x="rho", nbins=40, title=f"ρ(QoEC, {fisio}) por sessão · defasagem {defasagem:+.2f} s")
        fig.add_vline(x=rhos.rho.median(), line_dash="dash", annotation_text=f"mediana {rhos.rho.median():.3f}")
        fig.update_layout(height=360)
        st.plotly_chart(fig)
        st.caption(f"{len(rhos)} sessões com QoEC variável · Wilcoxon (mediana ≠ 0): p = {stats.wilcoxon(rhos.rho).pvalue:.3g}")
        baixar(rhos, f"rho_qoec_{fisio}", "baixar_rho")

with abas[2]:
    st.header("Relações entre variáveis")
    st.caption("Scatter de duas variáveis. Questionários (IPQ, NASA-TLX, SSQ) são respondidos por bloco: em nível de sessão "
               "o mesmo valor se repete nas 20 sessões do bloco, então prefira o nível participante × bloco para eles.")
    nivel = st.radio("Nível de observação", NIVEIS, horizontal=True, key="nivel_rel")
    tabela = observacoes(sessoes, quest, nivel)
    nomes = [v for v in [*VARIAVEIS, *QUESTIONARIOS] if v in tabela]
    c = st.columns(4)
    x = c[0].selectbox("Variável X", nomes, index=nomes.index("RMSSD"), help="Passe o mouse para ver a descrição")
    y = c[1].selectbox("Variável Y", nomes, index=nomes.index("MOS"))
    opcoes_cor = [k for k, col in CORES.items() if col is None or col in tabela]
    cor = c[2].selectbox("Colorir por", opcoes_cor)
    escala = c[3].selectbox("Escala", ["Valores originais", "z-score por participante"], disabled=nivel == "Por participante")
    st.caption(f"**X:** {descricao(x)} · **Y:** {descricao(y)}")
    if escala.startswith("z") and nivel != "Por participante":
        tabela = centralizar(tabela, [x, y])
    d = tabela.dropna(subset=[x, y]).copy()
    if CORES[cor] in ("participante", "bloco"):
        d[CORES[cor]] = d[CORES[cor]].astype(str)
    fig = px.scatter(d, x=x, y=y, color=CORES[cor], trendline="ols", trendline_scope="overall",
                     hover_data=[c for c in ["participante", "VideoID", "configuracao", "bloco"] if c in d], opacity=0.7)
    fig.update_layout(height=560)
    st.plotly_chart(fig)
    rho, pv = stats.spearmanr(d[x], d[y])
    texto = f"n = {len(d)} · Spearman ρ = {rho:.2f} (p = {pv:.3g})"
    if nivel != "Por participante":
        r_in, p_in, gl = corr_intra(d.assign(participante=d.participante.astype(int)), x, y)
        texto += f" · correlação intraparticipante r = {r_in:.2f} (p = {p_in:.3g}, gl = {gl})"
    st.caption(texto + ". A correlação intraparticipante remove a diferença de nível entre pessoas "
               "(equivale à correlação de medidas repetidas).")
    baixar(d, f"relacao_{x}_{y}", "baixar_rel")

with abas[3]:
    st.header("Matriz de correlações")
    c = st.columns(2)
    nivel_c = c[0].radio("Nível de observação", NIVEIS, horizontal=True, key="nivel_corr", index=1)
    intra = c[1].toggle("Remover diferença entre participantes (z-score por participante)", value=False,
                        disabled=nivel_c == "Por participante")
    tabela = observacoes(sessoes, quest, nivel_c)
    nomes = [v for v in [*VARIAVEIS, *QUESTIONARIOS] if v in tabela and tabela[v].nunique() > 1]
    padrao = ["MOS", "QoEC_media", "QoEC_final", "FC_media", "RMSSD", "SDNN", "FC_delta", "SC_media", "n_SCR", "vel_cabeca_media"]
    if nivel_c != "Por sessão":
        padrao += ["IPQ_total", "TLX_bruto", "SSQ_delta"]
    escolhidas = st.multiselect("Variáveis", nomes, default=[v for v in padrao if v in nomes])
    if len(escolhidas) > 1:
        if intra and nivel_c != "Por participante":
            tabela = centralizar(tabela, escolhidas)
        matriz = tabela[escolhidas].corr(method="spearman")
        fig = px.imshow(matriz, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto",
                        title=f"Spearman · {nivel_c}{' · intraparticipante' if intra else ''} · n = {len(tabela)}")
        fig.update_layout(height=640)
        st.plotly_chart(fig)
        st.caption("Exploratório: muitas comparações ao mesmo tempo. Correlações fracas aqui não são evidência sem correção para múltiplos testes.")
        baixar(matriz.reset_index(), "matriz_correlacao", "baixar_corr")

with abas[4]:
    st.header("Questionários por bloco")
    st.caption("IPQ (presença, −3 a 3), NASA-TLX (carga, 0–100) e SSQ (sintomas) foram respondidos ao fim de cada bloco de 20 vídeos. "
               "Itens invertidos do IPQ: 1, 11 e 13. O NASA-TLX usa a média simples das 6 subescalas porque os pesos faltam para vários participantes.")
    q = quest[quest.participante.isin(st.session_state.sel_pessoas) & quest.bloco.isin(st.session_state.sel_blocos)]
    grupo_q = st.selectbox("Escala", ["IPQ", "TLX", "SSQ"])
    cols_q = [c for c in QUESTIONARIOS if c.startswith(grupo_q)]
    longo = q.melt(id_vars=["participante", "bloco"], value_vars=cols_q, var_name="subescala", value_name="valor")
    fig = px.box(longo, x="subescala", y="valor", color=longo.bloco.astype(str), points="all",
                 labels={"color": "bloco"}, title=f"{grupo_q} por bloco")
    fig.update_layout(height=460)
    st.plotly_chart(fig)
    c = st.columns(2)
    escala_q = c[0].selectbox("Questionário", list(QUESTIONARIOS), format_func=descricao)
    qoe_q = c[1].selectbox("Resumo por bloco", ["MOS", "QoEC_media", "FC_media", "RMSSD", "SC_media"])
    bloco_tab = observacoes(sessoes, quest, "Por participante × bloco")
    fig = px.line(bloco_tab.sort_values("bloco"), x=escala_q, y=qoe_q, color=bloco_tab.sort_values("bloco").participante.astype(str),
                  markers=True, labels={"color": "participante"},
                  title=f"{qoe_q} médio do bloco × {escala_q} (linha liga os 2 blocos do mesmo participante)")
    fig.update_layout(height=480)
    st.plotly_chart(fig)
    with st.expander("Tabela dos questionários"):
        st.dataframe(q.round(2), hide_index=True)
        baixar(q, "questionarios", "baixar_q")

with abas[5]:
    st.header("PCA")
    st.caption("O PCA combina variáveis padronizadas em componentes que maximizam a variância. PC1 e PC2 não são, "
               "por definição, medidas de QoE, estresse ou presença.")
    c = st.columns(3)
    nivel_p = c[0].radio("Observações", NIVEIS[:2], horizontal=True, key="nivel_pca")
    centralizar_pca = c[1].toggle("Centralizar por participante antes do PCA", value=True)
    tabela = observacoes(sessoes, quest, nivel_p)
    nomes = [v for v in VARIAVEIS if v in tabela and tabela[v].nunique() > 1 and v not in ("ordem", "qp")]
    if nivel_p != "Por sessão":
        nomes += [v for v in QUESTIONARIOS if v in tabela]
    grupos = sorted({VARIAVEIS[v][0] for v in nomes if v in VARIAVEIS})
    familias = st.multiselect("Famílias", grupos, default=["HRV", "FC no tempo", "GSR"])
    escolhidas = st.multiselect("Variáveis", nomes, default=[v for v in nomes if v in VARIAVEIS and VARIAVEIS[v][0] in familias])
    cor_pca = c[2].selectbox("Colorir PCA por", [k for k, col in CORES.items() if col is None or col in tabela], key="cor_pca")
    d = tabela.dropna(subset=escolhidas).copy()
    if len(escolhidas) >= 2 and len(d) > len(escolhidas):
        if centralizar_pca:
            d = centralizar(d, escolhidas).dropna(subset=escolhidas)
        pca = PCA(n_components=min(5, len(escolhidas)))
        notas = pca.fit_transform(StandardScaler().fit_transform(d[escolhidas]))
        d["PC1"], d["PC2"] = notas[:, 0], notas[:, 1]
        if CORES[cor_pca] in ("participante", "bloco"):
            d[CORES[cor_pca]] = d[CORES[cor_pca]].astype(str)
        e, dd = st.columns([2, 1])
        fig = px.scatter(d, x="PC1", y="PC2", color=CORES[cor_pca], opacity=0.7,
                         hover_data=[c for c in ["participante", "VideoID", "configuracao", "bloco"] if c in d],
                         labels={"PC1": f"PC1 ({pca.explained_variance_ratio_[0]:.0%})", "PC2": f"PC2 ({pca.explained_variance_ratio_[1]:.0%})"})
        fig.update_layout(height=520)
        e.plotly_chart(fig)
        var = pd.DataFrame({"componente": [f"PC{i + 1}" for i in range(pca.n_components_)], "variância": pca.explained_variance_ratio_})
        dd.plotly_chart(px.bar(var, x="componente", y="variância", title="Variância explicada").update_layout(height=520))
        cargas = pd.DataFrame(pca.components_[:2].T, index=escolhidas, columns=["PC1", "PC2"])
        fig = px.bar(cargas.reset_index().melt(id_vars="index"), x="index", y="value", color="variable", barmode="group",
                     labels={"index": "variável", "value": "carga", "variable": ""}, title="Cargas (loadings)")
        st.plotly_chart(fig.update_layout(height=380))
        baixar(d, "pca_observacoes", "baixar_pca")
    else:
        st.info("Selecione ao menos duas variáveis com dados suficientes.")
