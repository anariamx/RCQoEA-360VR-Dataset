# RCQoEA-360VR · explorador científico

Dashboard em Streamlit para explorar o dataset RCQoEA-360VR: QoE contínua (QoEC), nota pós-vídeo (MOS), ECG/HRV, GSR, movimento de cabeça e questionários por bloco (IPQ, NASA-TLX, SSQ). Inspirado no explorador do MultiPhysio-HRC.

## Como rodar

A pasta `dashboard/` fica na raiz do repositório, ao lado de `Physiological_Data/`, `QoE_Annotation_Data/` etc. Ela tem o próprio `requirements.txt`, separado do `requirements.txt` da raiz (o dos notebooks), no mesmo esquema do explorador do MultiPhysio-HRC (`requirements.txt` + `requirements-explorer.txt`).

Com o ambiente virtual do projeto ativado, a partir da raiz do repositório:

```bash
pip install -r requirements.txt -r dashboard/requirements.txt
cd dashboard
streamlit run app.py
```

Rode o `streamlit run` de dentro de `dashboard/` para o tema em `.streamlit/config.toml` ser aplicado.

A pasta `dados/` já vem pronta. Para recriá-la a partir dos arquivos originais (leva cerca de 1 minuto), de dentro de `dashboard/`:

```bash
python preparar_dados.py               # usa a pasta acima desta como raiz do dataset
python preparar_dados.py /caminho/do/RCQoEA-360VR-Dataset
```

O botão "ECG bruto com picos R" lê `Physiological_Data/ECG/Raw_ECG_data/` da mesma raiz. Se o dataset estiver em outro lugar, defina `RCQOE_DIR=/caminho/do/dataset` antes do `streamlit run`.

## Abas

| Aba | O que mostra |
|---|---|
| Sessão | QoEC, FC instantânea, condutância da pele e velocidade da cabeça de uma sessão, com o MOS marcado; permite sobrepor outros participantes no mesmo vídeo e ver o ECG bruto com os picos R. |
| QoEC × MOS | Curvas médias no tempo (IC 95%) por configuração, MOS, conteúdo ou bloco; MOS por configuração; resumo da QoEC por MOS; correlação QoEC × FC/GSR/cabeça sessão a sessão com defasagem. |
| Relações | Scatter de duas variáveis por sessão, por participante × bloco ou por participante, com Spearman e correlação intraparticipante (medidas repetidas). |
| Correlações | Matriz de Spearman entre as variáveis escolhidas, com opção de remover a diferença entre participantes. |
| Questionários | IPQ, NASA-TLX e SSQ por bloco e sua relação com o MOS/QoEC/HRV médio do bloco. |
| PCA | PCA das variáveis fisiológicas ou de QoE, com variância explicada e cargas. |

## Arquivos

- `preparar_dados.py`: lê os arquivos originais e gera `dados/sessoes.csv` (uma linha por sessão), `dados/series.csv` (curvas em grade de 0,25 s), `dados/questionarios.csv` (participante × bloco) e `dados/participantes.csv`.
- `app.py`: o dashboard.
- `mapeamento_videos.csv`: VideoID → conteúdo e configuração de qualidade.

## Decisões de processamento (conferir com o orientador)

- **Mapeamento dos vídeos é hipótese.** A documentação não diz qual VideoID corresponde a qual conteúdo/configuração. O padrão do MOS sugere grupos de 8 vídeos por conteúdo (V1–V8, V9–V16, …) com o 4º de cada grupo sendo QP42 (MOS médio entre 1,4 e 1,9). A ordem dentro do grupo (QP15, QP22, QP32, QP42, depois as 4 não uniformes) e os nomes dos conteúdos seguem a ordem do artigo. Para corrigir, basta editar `mapeamento_videos.csv` e rodar `preparar_dados.py` de novo.
- **Tempo.** As linhas da QoEC e do movimento de cabeça são distribuídas uniformemente nos 30 s do vídeo (a coluna TimeStamp da QoEC não representa o tempo real). ECG e GSR usam o horário de cada amostra.
- **QoEC.** Escala observada de 0 a 1. Os zeros do início de cada sessão (antes do primeiro toque no touchpad) são tratados como "sem anotação". 47 sessões não têm anotação utilizável e ficam fora por padrão.
- **HRV.** Calculada do ECG bruto (Polar H10, 130 Hz) com NeuroKit2: limpeza, picos R com correção de artefatos, RR entre 330 e 1500 ms. Só medidas no domínio do tempo e Poincaré (FC, SDNN, RMSSD, pNN50, SD1, SD2), porque 30 s é curto demais para LF e LF/HF. A polaridade do ECG é decidida por participante (maioria das 40 sessões); ficam invertidos P8, P11, P13, P17, P19 e P32.
- **Kubios não foi usado.** O arquivo `KubiosHRVresults_equal_length_clean_nonan.csv` identifica os participantes com números de 1 a 49 (IDs do laboratório), não P1–P32, e a correspondência não é evidente. Vale perguntar aos autores.
- **Blocos.** Bloco 1 = primeiros 20 vídeos em ordem cronológica de cada participante (há um intervalo de 6 a 27 min entre o 20º e o 21º vídeo em todos).
- **IPQ.** Subescalas montadas a partir do texto dos itens no apêndice: G = 8; SP = 3, 6, 9, 10, 13 (invertido); INV = 1 (invertido), 7, 11 (invertido), 14; REAL = 2, 4, 5, 12.
- **NASA-TLX.** Média simples das 6 subescalas (TLX bruto), porque os pesos estão vazios para vários participantes. A subescala de desempenho está como na planilha; confirmar a direção (0 = bom ou ruim).
- **SSQ.** Escore total padrão: (N + O + D) × 3,74.
