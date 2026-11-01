import re
import sys
import warnings
from pathlib import Path

import neurokit2 as nk
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

RAIZ = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent
SAIDA = Path(__file__).resolve().parent / "dados"
DURACAO = 30.0
PASSO = 0.25
GRADE = np.arange(0, DURACAO + 1e-9, PASSO)
FS = 130

IPQ = {
    "IPQ_G": ["IPQ_8"],
    "IPQ_SP": ["IPQ_3", "IPQ_6", "IPQ_9", "IPQ_10", "-IPQ_13"],
    "IPQ_INV": ["-IPQ_1", "IPQ_7", "-IPQ_11", "IPQ_14"],
    "IPQ_REAL": ["IPQ_2", "IPQ_4", "IPQ_5", "IPQ_12"],
}

SSQ = {
    "N": ["General_Discomfort", "Salivation", "Sweating", "Nausea", "Concentrating", "Stomach", "Burping"],
    "O": ["General_Discomfort", "Fatigue", "Headache", "Eyestrain", "Focusing", "Concentrating", "Vision"],
    "D": ["Focusing", "Nausea", "Fullness", "Vision", "Dizzi_eyeopen", "Dizzi_eyeclosed", "Vertigo"],
}

TLX = ["TLX_Mental", "TLX_Physical", "TLX_Temporal", "TLX_Effort", "TLX_Performance", "TLX_Frustration"]


def na_grade(t, y):
    ok = np.isfinite(t) & np.isfinite(y)
    if ok.sum() < 2:
        return np.full(len(GRADE), np.nan)
    out = np.interp(GRADE, t[ok], y[ok], left=np.nan, right=np.nan)
    return out


def ler_qoec(arquivo):
    q = pd.read_csv(arquivo, sep=";", skipinitialspace=True).iloc[:, 2].to_numpy(float)
    t = np.linspace(0, DURACAO, len(q))
    limpa = q.copy()
    if (q != 0).any():
        limpa[: np.argmax(q != 0)] = np.nan
    else:
        limpa[:] = np.nan
    return na_grade(t, q), na_grade(t, limpa)


def ler_cabeca(arquivo):
    h = pd.read_csv(arquivo, sep=";", skipinitialspace=True).iloc[:, 2:5].to_numpy(float)
    t = np.linspace(0, DURACAO, len(h))
    ang = np.unwrap(np.radians(h[:, :2]), axis=0)
    vel = np.degrees(np.hypot(*np.diff(ang, axis=0).T)) / np.diff(t)
    return na_grade(t[1:], vel)


def ler_mos(arquivo):
    linhas = Path(arquivo).read_text().splitlines()
    return float(linhas[1].split(";")[2])


def ler_ecg(arquivo):
    e = pd.read_csv(arquivo)
    t = (pd.to_datetime(e["time"], format="ISO8601") - pd.to_datetime(e["time"].iloc[0], format="ISO8601")).dt.total_seconds().to_numpy()
    return t, e["ECG"].to_numpy(float)


def processar_ecg(t, ecg, inverter):
    sinal = -ecg if inverter else ecg
    limpo = nk.ecg_clean(sinal, sampling_rate=FS)
    _, info = nk.ecg_peaks(limpo, sampling_rate=FS, correct_artifacts=True)
    picos = info["ECG_R_Peaks"]
    rr = np.diff(t[picos]) * 1000
    t_rr = t[picos][1:]
    ok = (rr > 330) & (rr < 1500)
    rr, t_rr = rr[ok], t_rr[ok]
    if len(rr) < 10:
        return np.full(len(GRADE), np.nan), {}
    dif = np.diff(rr)
    sdnn = np.std(rr, ddof=1)
    rmssd = np.sqrt(np.mean(dif ** 2))
    sd1 = rmssd / np.sqrt(2)
    hrv = {
        "FC_media": 60000 / rr.mean(),
        "RR_medio": rr.mean(),
        "SDNN": sdnn,
        "RMSSD": rmssd,
        "pNN50": np.mean(np.abs(dif) > 50) * 100,
        "SD1": sd1,
        "SD2": np.sqrt(max(2 * sdnn ** 2 - sd1 ** 2, 0)),
        "n_batimentos": len(rr) + 1,
    }
    return na_grade(t_rr, 60000 / rr), hrv


def ler_gsr(arquivo):
    cab = pd.read_csv(arquivo, nrows=0).columns
    col_sc = [c for c in cab if c.endswith("Skin_Conductance_CAL")][0]
    g = pd.read_csv(arquivo, usecols=[cab[0], col_sc], skiprows=[1])
    t = (pd.to_datetime(g[cab[0]], format="ISO8601") - pd.to_datetime(g[cab[0]].iloc[0], format="ISO8601")).dt.total_seconds().to_numpy()
    sc = g[col_sc].to_numpy(float)
    try:
        fasica = nk.eda_phasic(nk.eda_clean(sc, sampling_rate=FS), sampling_rate=FS)["EDA_Phasic"].to_numpy()
        n_scr = len(nk.eda_peaks(fasica, sampling_rate=FS)[1]["SCR_Peaks"])
    except Exception:
        n_scr = np.nan
    return na_grade(t, sc), n_scr


def media_trecho(y, ini, fim):
    trecho = y[(GRADE >= ini) & (GRADE < fim)]
    return np.nanmean(trecho) if np.isfinite(trecho).any() else np.nan


def inclinacao(y):
    ok = np.isfinite(y)
    return np.polyfit(GRADE[ok], y[ok], 1)[0] * 10 if ok.sum() > 4 else np.nan


def montar_sessoes():
    pasta_qoe = RAIZ / "QoE_Annotation_Data"
    linhas = []
    for arq in sorted((pasta_qoe / "Continuous_QoE_Score").glob("RCQoE_*.txt")):
        p, v, data = re.search(r"P(\d+)_V(\d+)_(\w+)\.txt", arq.name).groups()
        linhas.append({"participante": int(p), "VideoID": int(v), "inicio": pd.to_datetime(data), "sufixo": arq.name[6:]})
    s = pd.DataFrame(linhas).sort_values(["participante", "inicio"])
    s["ordem"] = s.groupby("participante").cumcount() + 1
    s["bloco"] = np.where(s["ordem"] <= 20, 1, 2)
    return s.reset_index(drop=True)


def main():
    SAIDA.mkdir(exist_ok=True)
    sessoes = montar_sessoes()
    mapa = pd.read_csv(Path(__file__).resolve().parent / "mapeamento_videos.csv")
    pasta_qoe = RAIZ / "QoE_Annotation_Data"
    pasta_ecg = RAIZ / "Physiological_Data/ECG/Raw_ECG_data"
    pasta_gsr = RAIZ / "Physiological_Data/GSR/Raw_GSR_data"

    print("Lendo ECG bruto e decidindo a polaridade por participante...")
    ecgs, invertido = {}, {}
    for r in sessoes.itertuples():
        arq = pasta_ecg / f"ecg_P{r.participante}_V{r.VideoID}.csv"
        if arq.exists():
            t, e = ler_ecg(arq)
            ecgs[r.Index] = (t, e)
            invertido[r.Index] = nk.ecg_invert(e, sampling_rate=FS)[1]
    sessoes["ecg_invertido_sessao"] = pd.Series(invertido)
    maioria = sessoes.groupby("participante")["ecg_invertido_sessao"].mean() > 0.5
    sessoes["ecg_invertido"] = sessoes["participante"].map(maioria)

    series, resumo = [], []
    for r in sessoes.itertuples():
        if r.Index % 100 == 0:
            print(f"  sessão {r.Index + 1}/{len(sessoes)}")
        qoec_bruta, qoec = ler_qoec(pasta_qoe / "Continuous_QoE_Score" / f"RCQoE_{r.sufixo}")
        cabeca = ler_cabeca(RAIZ / "Behavioral_Data/HM_data" / f"HM_{r.sufixo}")
        fc, hrv = processar_ecg(*ecgs[r.Index], r.ecg_invertido) if r.Index in ecgs else (np.full(len(GRADE), np.nan), {})
        arq_gsr = pasta_gsr / f"gsr_P{r.participante}_V{r.VideoID}.csv"
        sc, n_scr = ler_gsr(arq_gsr) if arq_gsr.exists() else (np.full(len(GRADE), np.nan), np.nan)

        series.append(pd.DataFrame({
            "participante": r.participante, "VideoID": r.VideoID, "t": GRADE,
            "QoEC_bruta": qoec_bruta, "QoEC": qoec, "FC": fc, "SC": sc, "vel_cabeca": cabeca,
        }))
        valida = np.isfinite(qoec)
        resumo.append({
            "MOS": ler_mos(pasta_qoe / "Post_Video_QoE_Score" / f"PostQuestion_{r.sufixo}"),
            "QoEC_media": np.nanmean(qoec) if valida.any() else np.nan,
            "QoEC_dp": np.nanstd(qoec) if valida.any() else np.nan,
            "QoEC_min": np.nanmin(qoec) if valida.any() else np.nan,
            "QoEC_max": np.nanmax(qoec) if valida.any() else np.nan,
            "QoEC_final": media_trecho(qoec, DURACAO - 3, DURACAO + 1),
            "QoEC_tendencia": inclinacao(qoec),
            "QoEC_pct_anotada": valida.mean() * 100,
            "FC_inicio": media_trecho(fc, 0, 5),
            "FC_fim": media_trecho(fc, DURACAO - 5, DURACAO + 1),
            "FC_tendencia": inclinacao(fc),
            "SC_media": np.nanmean(sc) if np.isfinite(sc).any() else np.nan,
            "SC_delta": media_trecho(sc, DURACAO - 3, DURACAO + 1) - media_trecho(sc, 0, 3),
            "n_SCR": n_scr,
            "vel_cabeca_media": np.nanmean(cabeca) if np.isfinite(cabeca).any() else np.nan,
            **hrv,
        })

    sessoes = pd.concat([sessoes, pd.DataFrame(resumo)], axis=1)
    sessoes["FC_delta"] = sessoes["FC_fim"] - sessoes["FC_inicio"]

    sessoes = sessoes.merge(mapa, on="VideoID", how="left").drop(columns=["sufixo", "ecg_invertido_sessao"])

    quest = ler_questionarios()
    participantes = pd.read_csv(pasta_qoe / "participant_information.csv")
    participantes["participante"] = participantes["Participant_id"].str[1:].astype(int)

    sessoes.to_csv(SAIDA / "sessoes.csv", index=False)
    pd.concat(series).round(4).to_csv(SAIDA / "series.csv", index=False)
    quest.to_csv(SAIDA / "questionarios.csv", index=False)
    participantes.to_csv(SAIDA / "participantes.csv", index=False)
    print(f"Pronto: {len(sessoes)} sessões salvas em {SAIDA}")


def ler_questionarios():
    pasta = RAIZ / "Questionnaires"
    pre = pd.read_excel(pasta / "SSQ_PreTest.xlsx", header=1)
    blocos = []
    for b in (1, 2):
        ipq = pd.read_excel(pasta / f"IPQ_Block{b}.xlsx", header=1)
        tlx = pd.read_excel(pasta / f"NASA_TLX_Block{b}.xlsx", header=2)
        ssq = pd.read_excel(pasta / f"SSQ_Block{b}.xlsx", header=1)
        q = pd.DataFrame({"participante": ipq["Participant_ID"].str[1:].astype(int), "bloco": b})
        for nome, itens in IPQ.items():
            q[nome] = sum(-ipq[i[1:]] if i.startswith("-") else ipq[i] for i in itens) / len(itens)
        q["IPQ_total"] = q[list(IPQ)].mean(axis=1)
        tlx = tlx.set_index(tlx["Participant_ID"].str[1:].astype(int)).reindex(q["participante"])
        for c in TLX:
            q[c] = tlx[c].to_numpy()
        q["TLX_bruto"] = q[TLX].mean(axis=1)
        q["SSQ_total"] = ssq_total(ssq).to_numpy()
        q["SSQ_pre"] = ssq_total(pre).to_numpy()
        q["SSQ_delta"] = q["SSQ_total"] - q["SSQ_pre"]
        blocos.append(q)
    return pd.concat(blocos, ignore_index=True)


def ssq_total(tabela):
    soma = sum(tabela[[f"SSQ_{i}" for i in itens]].sum(axis=1) for itens in SSQ.values())
    return soma * 3.74


if __name__ == "__main__":
    main()
