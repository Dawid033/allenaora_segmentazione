"""Analisi dei clienti AllenaOra: preparazione dei dati, K-Means, profili e validazione"""
from pathlib import Path
import pandas as pd

CARTELLA_DATI = Path(__file__).parent / "data"
FILE_CLIENTI = CARTELLA_DATI/ "clienti.csv"

def carica_dati(percorso):
    """Legge il CSV dei clienti, con la data di iscrizione letta come data."""
    return pd.read_csv(percorso, parse_dates=["data_iscrizione"])

def sostituisci_mancanti(clienti):
    """Calcola l'importo medio per sessione e sostituisce i mancanti:
    mediana per l'importo medio e giorni dall'ultima sessione, moda per la categoria.
    Restituisce la tabella completa e il numero di record toccati."""
    clienti = clienti.copy()
    record_toccati = int(clienti.isna().any(axis=1).sum())

    #dove manca l'importo totale, l'importo medio è NaN
    clienti["importo_medio"] = clienti["importo_totale"]/clienti["num_sessioni"]
    m_mancante= clienti["importo_medio"].isna()

    mediana_m = clienti["importo_medio"].median()
    mediana_r = clienti["giorni_ultima_sessione"].median()
    moda_cat = clienti["categoria_preferita"].mode()[0]

    clienti["importo_medio"] = clienti["importo_medio"].fillna(mediana_m)
    clienti.loc[m_mancante, "importo_totale"] = (
        clienti.loc[m_mancante, "importo_medio"] * clienti.loc[m_mancante, "num_sessioni"]
    ).round(2)
    clienti["giorni_ultima_sessione"] = clienti["giorni_ultima_sessione"].fillna(mediana_r)
    clienti["categoria_preferita"] = clienti["categoria_preferita"].fillna(moda_cat)

    print(f"Mediana importo medio: {mediana_m:.2f} - mediana giorni: {mediana_r} - moda categoria: {moda_cat}")
    return clienti, record_toccati

def main():
    clienti = carica_dati(FILE_CLIENTI)

    print("Forma:", clienti.shape)
    print(clienti.head())
    print(clienti.dtypes)

    righe_identiche = clienti.duplicated().sum()
    id_ripetuti = clienti["id_cliente"].duplicated().sum()
    print("Righe identiche:", righe_identiche, "- id ripetuti:", id_ripetuti)

    clienti = clienti.drop_duplicates()
    print("Forma dopo la rimozione dei duplicati:", clienti.shape)

    print("Mancanti per colonna:")
    print(clienti.isna().sum())

    #analisi_clustering, sostituzione dei mancanti
    clienti, record_toccati = sostituisci_mancanti(clienti)
    print("Record toccati:", record_toccati)
    print("Mancanti dopo la sostituzione:", int(clienti.isna().sum().sum()))
    print("Forma dopo la sostituzione:", clienti.shape)

    # analisi_clustering.py (dentro main, temporaneo)
    anzianita = (pd.Timestamp("2026-06-30") - clienti["data_iscrizione"]).dt.days
    print("Giorni oltre l'anzianità:", int((clienti["giorni_ultima_sessione"] > anzianita).sum()))

if __name__ == "__main__":
    main()
