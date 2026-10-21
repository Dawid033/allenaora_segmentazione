"""Analisi dei clienti AllenaOra: preparazione dei dati, K-Means, profili e validazione"""
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score
import joblib

CARTELLA_DATI = Path(__file__).parent / "data"
FILE_CLIENTI = CARTELLA_DATI/ "clienti.csv"
FILE_SEGMENTI = CARTELLA_DATI / "segmenti_veri.csv"
COLONNE_RFM = ["giorni_ultima_sessione", "num_sessioni", "importo_medio"]  # R, F, M in quest'ordine
CARTELLA_OUTPUT = Path(__file__).parent / "output"
FILE_MODELLO = CARTELLA_OUTPUT / "modello.joblib"
SEED = 42
VALORI_K= range(2,9) #i valori sono da 2 a 8 compresi, il 9 è escluso
K_CANDIDATI = [3,4,5]
K_SCELTO = 4
NOMI_CLUSTER= {
    0: "Fedelissimi assidui",
    1: "Intensivi",
    2: "Saltuari",
    3: "Inattivi",
}
AZIONI_CLUSTER={
    0: "Fidelizzare: abbonamento o pacchetto a prezzo bloccato/Programma porta un amico",
    1: "Trattenere: pacchetto per aumentarne la continuità/proposte di ripresa post estate",
    2: "Aumentare la frequenza: promemoria personalizzati e pacchetti da 5 sessioni a prezzo bloccato",
    3: "Riattivare: campagna di rientro a gennaio e/o settembre con una sessione di prova"
}

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


def prepara_feature(clienti):
    """Restituisce la matrice delle feature: solo R, F e M, nell'ordine di COLONNE_RFM."""
    return clienti[COLONNE_RFM]

def standardizza(X):
    """Standardizza R, F e M (z-score: media 0, deviazione standard 1).
    Restituisce la matrice standardizzata (array NumPy) e lo scaler addestrato,
    che serve a standardizzare i nuovi clienti con le stesse medie e deviazioni standard."""
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    return X_std, scaler

def confronta_k(X_std):
    """Addestra K-Means per ogni k di VALORI_K e misura inerzia e silhouette
    Restituisce una tabella con una riga per ogni k."""
    righe = []
    for k in VALORI_K:
        modello = KMeans(n_clusters=k, random_state=SEED, n_init=10)
        modello.fit(X_std)
        silhouette = silhouette_score(X_std, modello.labels_)
        righe.append({"k": k, "inerzia": modello.inertia_, "silhouette": silhouette})
    return pd.DataFrame(righe)

def salva_grafici(confronto):
    """salva in output i 2 grafici"""
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(confronto["k"], confronto["inerzia"], marker="o")
    ax.set_xlabel("Numero di cluster k")
    ax.set_ylabel("Inerzia")
    ax.set_title("Metodo del gomito")
    fig.tight_layout()
    fig.savefig(CARTELLA_OUTPUT / "grafico_gomito.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(confronto["k"], confronto["silhouette"], marker="o")
    ax.set_xlabel("Numero di cluster k")
    ax.set_ylabel("Silhouette media")
    ax.set_title("Silhouette al variare di k")
    fig.tight_layout()
    fig.savefig(CARTELLA_OUTPUT / "grafico_silhouette.png", dpi=150)
    plt.close(fig)

def descrivi_candidati(X_std, scaler):
    """Per ogni k candidato stampa numerosità dei cluster e centroidi
    riportati come: giorni, sessioni, euro"""
    for k in K_CANDIDATI:
        modello = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit(X_std)
        centroidi =pd.DataFrame(scaler.inverse_transform(modello.cluster_centers_),
                                columns=["R_giorni", "F_sessioni", "M_euro"])
        centroidi["numerosita"] = pd.Series(modello.labels_).value_counts().sort_index()
        print(f"\nk = {k}")
        print(centroidi.round(1))

def assegna_cluster(clienti, X_std):
    """Addestra il modello finale con K_SCELTO cluster e aggiunge una colonna in cluster"""
    clienti = clienti.copy()
    modello = KMeans(n_clusters=K_SCELTO, random_state=SEED, n_init=10)
    clienti["cluster"] = modello.fit_predict(X_std)
    return clienti, modello

def valore_prevalente(serie):
    #restituisce il valore più frequente
    #se c'è parità il primo in ordine
    return serie.mode()[0]

def quota_prevalente(serie):
    #restituisce la quota in percentuale del valore più frequente
    return serie.value_counts(normalize=True).iloc[0]*100

def calcola_profili(clienti):
    """Calcola il profilo di ogni cluster: medie RFM, numerosità
    fatturato e valori più frequenti"""
    clienti=clienti.copy()
    clienti["mese_iscrizione"] = clienti["data_iscrizione"].dt.strftime("%Y-%m")

    profili = clienti.groupby("cluster").agg(
         numerosita=("id_cliente", "count"),
        R_media_giorni=("giorni_ultima_sessione", "mean"),
        F_media_sessioni=("num_sessioni", "mean"),
        M_media_euro=("importo_medio", "mean"),
        importo_totale_medio=("importo_totale", "mean"),
        fatturato=("importo_totale", "sum"),
        categoria_prevalente=("categoria_preferita", valore_prevalente),
        quota_categoria=("categoria_preferita", quota_prevalente),
        zona_prevalente=("zona", valore_prevalente),
        quota_zona=("zona", quota_prevalente),
        mese_frequente=("mese_iscrizione", valore_prevalente),
        quota_mese=("mese_iscrizione", quota_prevalente),
    )
    profili["quota_clienti"] = profili["numerosita"] / profili["numerosita"].sum()*100
    profili["quota_fatturato"] = profili["fatturato"] / profili ["fatturato"].sum()*100
    return profili.round(2).reset_index()

def controlla_nomi(profili):
    """verifica che i nomi dei cluster corrispondano ancora ai vari profili"""
    p= profili.set_index("cluster")
    corrispondono=(
        p["F_media_sessioni"].idxmax()==0
        and p["M_media_euro"].idxmax()==1
        and p["M_media_euro"].idxmin()==2
        and p["R_media_giorni"].idxmax()==3
    )
    if not corrispondono:
        raise ValueError("I cluster non corrispondono più ai nomi in questo momento: " \
        "rileggere i profili e aggiornare NOMI_CLUSTER")

def aggiungi_nomi(profili):
    """Aggiunge ai profili il nome e l'azione di marketing di ogni cluster"""
    controlla_nomi(profili)
    profili=profili.copy()
    profili["nome"]=profili["cluster"].map(NOMI_CLUSTER)
    profili["azione"]=profili["cluster"].map(AZIONI_CLUSTER)
    return profili

def valida_cluster(clienti, percorso_segmenti):
    """confronta i cluster con i segmenti veri, cioè tabella incrociata e ARI"""
    segmenti =pd.read_csv(percorso_segmenti)
    unione=clienti.merge(segmenti, on="id_cliente", validate="one_to_one")
    if len(unione) != len(clienti):
        raise ValueError("Alcuni clienti non hanno un segmento vero corrispondente")
    tabella = pd.crosstab(unione["cluster"], unione["segmento"])
    ari = adjusted_rand_score(unione["segmento"], unione["cluster"])
    return tabella, ari

def salva_modello(scaler, modello, percorso):
    """salva scaler e modello addestrati in un unico file joblib"""
    joblib.dump({"scaler": scaler, "modello": modello}, percorso)

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

    # Passo 3: matrice delle feature RFM
    X = prepara_feature(clienti)
    print("Forma della matrice delle feature:", X.shape)
    print(X.describe().round(2))

    # Passo 4: standardizzazione
    X_std, scaler = standardizza(X)
    print("Medie usate:", scaler.mean_.round(2))
    print("Deviazioni standard usate:", scaler.scale_.round(2))
    print(pd.DataFrame(X_std, columns=COLONNE_RFM).describe().round(2))

    CARTELLA_OUTPUT.mkdir(exist_ok=True)
    confronto =confronta_k(X_std)
    confronto.to_csv(CARTELLA_OUTPUT / "confronto_k.csv", index=False)
    print(confronto.round(3))
    salva_grafici(confronto)
    descrivi_candidati(X_std, scaler)

    clienti, modello = assegna_cluster(clienti, X_std)
    print("Clienti per cluster:")
    print(clienti["cluster"].value_counts().sort_index())
    clienti.to_csv(CARTELLA_OUTPUT / "clienti_con_cluster.csv", index=False)

    profili=calcola_profili(clienti)
    profili=aggiungi_nomi(profili)
    print(profili.set_index("cluster").T.to_string())
    profili.to_csv(CARTELLA_OUTPUT / "profili_cluster.csv", index=False)
    tabella, ari = valida_cluster(clienti, FILE_SEGMENTI)
    print("Tabella incrociata cluster x segmento vero:")
    print(tabella)
    print(f"ARI: {ari:.3f}\n") 
    percorso_validazione = CARTELLA_OUTPUT / "validazione.csv"
    tabella.to_csv(percorso_validazione)
    with open(percorso_validazione, "a") as f:
        f.write(f"ARI,{ari:.3f}\n")

    salva_modello(scaler, modello, FILE_MODELLO)
    print("Scaler e modello salvati in", FILE_MODELLO)

if __name__ == "__main__":
    main()
