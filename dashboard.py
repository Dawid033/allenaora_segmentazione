"""Dashboard interattiva

avvio: stramlit run dashboard.py -> dopo genera_dati e analisi_clustering"""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

CARTELLA_OUTPUT = Path(__file__).parent / "output"
FILE_CLIENTI = CARTELLA_OUTPUT / "clienti_con_cluster.csv"
FILE_PROFILI = CARTELLA_OUTPUT / "profili_cluster.csv"
FILE_CONFRONTO = CARTELLA_OUTPUT / "confronto_k.csv"
FILE_MODELLO = CARTELLA_OUTPUT / "modello.joblib"
COLONNA_SILHOUETTE = "silhouette"

ETICHETTE_ASSI = {
    "R - giorni dall'ultima sessione" : "giorni_ultima_sessione",
    "F - numero di sessioni": "num_sessioni",
    "M - importo medio per sessione ": "importo_medio",
}

COLORI_CLUSTER = {0: "tab:green", 1: "tab:purple", 2:"tab:orange", 3: "tab:gray"}

def controlla_file():
    """se manca uno dei file -> ferma la pagina"""
    file_richiesti = [FILE_CLIENTI, FILE_PROFILI, FILE_CONFRONTO, FILE_MODELLO]
    mancanti = [f.name for f in file_richiesti if not f.exists()]
    if mancanti:
        st.error(
            "file mancanti in output/: " + ", ".join(mancanti)
            +". Eseguire prima: python analisi_clustering.py"
        )
        st.stop()

@st.cache_data 
def carica_tabelle():
    """legge i CSV prodotti da analisi_clustering.py"""
    clienti = pd.read_csv(FILE_CLIENTI, parse_dates=["data_iscrizione"])
    profili = pd.read_csv(FILE_PROFILI)
    confronto = pd.read_csv(FILE_CONFRONTO)
    return clienti, profili, confronto

@st.cache_resource
def carica_modello():
    contenuto = joblib.load(FILE_MODELLO)
    return contenuto["scaler"], contenuto["modello"]

def filtri_sidebar(clienti):
    """disegna i filtri e restutuisce le scelte dell'utente"""
    st.sidebar.header("Filtri")

    tutte_zone = sorted(clienti["zona"].unique())
    zone = st.sidebar.multiselect("Zona", options=tutte_zone, default=tutte_zone)

    data_min = clienti["data_iscrizione"].min().date()
    data_max = clienti["data_iscrizione"].max().date()
    periodo = st.sidebar.date_input(
        "Periodo di iscrizione",
        value =(data_min, data_max),
        min_value=data_min,
        max_value=data_max,
    )

    tutte_categorie = sorted(clienti["categoria_preferita"].unique())
    categorie = st.sidebar.multiselect(
        "Categoria preferita", options=tutte_categorie, default=tutte_categorie
    )
    return zone, periodo, categorie

def applica_filtri(clienti, zone, periodo, categorie):
    """Restituisce solo i clienti che rispettano tutti e tre i filtri."""
    inizio, fine = periodo
    maschera = (
        clienti["zona"].isin(zone)
        & clienti["data_iscrizione"].between(pd.Timestamp(inizio), pd.Timestamp(fine))
        & clienti["categoria_preferita"].isin(categorie)
    )
    return clienti[maschera]

def disegna_scatter(filtrati, nomi_cluster):
    """Scatter dei clienti filtrati su due variabili scelte, un colore fisso per cluster."""
    st.subheader("Clienti per cluster")

    etichette = list(ETICHETTE_ASSI.keys())
    sinistra, destra = st.columns(2)
    etichetta_x = sinistra.selectbox("Asse x", etichette, index=0)
    etichetta_y = destra.selectbox("Asse y", etichette, index=1)
    colonna_x = ETICHETTE_ASSI[etichetta_x]
    colonna_y = ETICHETTE_ASSI[etichetta_y]

    fig, ax = plt.subplots(figsize=(8, 5))
    for numero, gruppo in filtrati.groupby("cluster"):
        ax.scatter(
            gruppo[colonna_x],
            gruppo[colonna_y],
            color=COLORI_CLUSTER[numero],
            label=f"{numero} – {nomi_cluster[numero]}",
            alpha=0.6,
        )
    ax.set_xlabel(etichetta_x)
    ax.set_ylabel(etichetta_y)
    ax.legend(title="Cluster")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

def main():
    st.set_page_config(page_title="AllenaOra – Segmentazione", layout="wide")
    st.title("AllenaOra – Segmentazione dei clienti")

    controlla_file()
    clienti, profili, confronto = carica_tabelle()
    scaler, modello = carica_modello()
    nomi_cluster = dict(zip(profili["cluster"], profili["nome"]))

    zone, periodo, categorie = filtri_sidebar(clienti)
    if len(periodo) != 2:
        st.info("Selezionare anche la data di fine del periodo.")
        st.stop()
    filtrati = applica_filtri(clienti, zone, periodo, categorie)

    k = modello.n_clusters
    silhouette = confronto.loc[confronto["k"] == k, COLONNA_SILHOUETTE].iloc[0]

    col1, col2, col3 = st.columns(3)
    col1.metric("Clienti filtrati", f"{len(filtrati)} su {len(clienti)}")
    col2.metric("Numero di gruppi (k)", k)
    col3.metric("Silhouette del modello (500 clienti)", f"{silhouette:.3f}")

    if filtrati.empty:
        st.warning("Nessun cliente corrisponde ai filtri selezionati.")
    else:
        disegna_scatter(filtrati, nomi_cluster)

if __name__ == "__main__":
    main()
