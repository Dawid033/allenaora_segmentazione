"""Analisi dei clienti AllenaOra: preparazione dei dati, K-Means, profili e validazione"""
from pathlib import Path
import pandas as pd

CARTELLA_DATI = Path(__file__).parent / "data"
FILE_CLIENTI = CARTELLA_DATI/ "clienti.csv"

def carica_dati(percorso):
    """Legge il CSV dei clienti, con la data di iscrizione letta come data."""
    return pd.read_csv(percorso, parse_dates=["data_iscrizione"])

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

if __name__ == "__main__":
    main()
