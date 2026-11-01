# Segmentazione dei clienti di AllenaOra

Il progetto riguarda un prototipo di segmentazione automatica della clientela, con Machine Learning non supervisionato,
I clienti vengono ragruppati con l'algoritmo K-Means in base alle tre variabili del modello scelto.
Il modello che ho scelto è RFM (Recency o "l'essere recente", frequency o frequenza, monetary o monetario).
I gruppi si esplorano in una dashboard come richiesto dalla traccia.

# Scenario

AllenaOra è una startup fittizia di personal training. 
Questo nome non è in uso ne su App store ne su Play store.
I pagamenti vengono effettuati in base al numero di sessioni a cui partecipa un cliente.
I dati sono simulati: ho ipotizzato 500 clienti sotto osservazione per 1 anno. dal 1 Luglio 2025 al 30 Giugno 2026.
Il dataset viene generato su quattro comportamenti tipici della clientela che ho ipotizzato e che credo si adattino bene al contesto:
    - Clienti costanti,
    - Prova costume,
    - Clienti occasionali,
    - Ripartenze mancate (chi si iscrive dopo le festività, inizio anno o fine estate per 'durare' poche settimane/allenamenti).

# Requisiti

- Python 3.14 (progetto realizzato con Python 3.14.8 su macOs)
- Git

# Installazione
git clone https://github.com/Dawid033/allenaora_segmentazione.git
cd allenaora_segmentazione
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

Su Windows usare `python` al posto di `python3`.

# Esecuzione

Per testare correttamente il progetto, vanno eseguiti questi tre comandi, in questo ordine e con l'ambiente virtuale attivo

```
python genera_dati.py
python analisi_clustering.py
streamlit run dashboard.py
```

- Il primo comando serve per generare il dataset simulato in data\
- Il secondo comando prepara i dati, sceglie il numero di cluster, addestra K-Means, calcola i profili, valida il risultato e salva tutto quanto in output/
- Il terzo comando apre la dashboard nel browser all'indirizzo http://localhost:8501  

L'ordine di esecuzione di questi tre comandi è importante, siccome la dashboard legge output/modello.joblib , che non è incluso nel repository. Questo file viene creato da analisi_clustering.py . Se la dashboard viene avviata prima dell'analisi, compare a schermo un messaggio di errore, che invita l'utente ad eseguire il comando python analisi_clustering.py .

I CSV di data/ e di output/ sono già presenti nel repository.
Rieseguire i primi due script permette di verificarne la riproducibilità, dovuta ad un SEED fisso.

# File prodotti

| File | Contenuto |

| 'data/clienti.csv' | Dataset dei 500 clienti |
| 'data/segmenti_veri.csv' | Segmento vero di ogni cliente, usato solo per la validazione |
| 'output/confronto_k.csv' | Inerzia e silhouette per k da 2 a 8 |
| 'output/grafico_gomito.png | Grafico del metodo del gomito |
| 'output/grafico_silhouette.png' | Grafico della silhouette media|
| 'output/clienti_con_cluster.csv' | Clienti con il rispettivo cluster assegnato |
| 'output/profili_cluster.csv' | Profilo, nome e azione di marketing ipotizzata per ogni cluster |
| 'output/validazione.csv' | Tabella incrociataa cluster x segmento vero e ARI |
| 'output/modello.joblib' | Scaler e modello K-Means usati dalla dashboard |

# Parametri modificabili

| Costante | File | Valore assegnato| Significato |

| 'SEED' | 'genera_dati.py' e 'analisi_clustering.py' | 42 | Seme dei numeri casuali |
| 'N_CLIENTI' | 'genera_dati.py' | 500 | Numero di clienti simulati
| 'DATA_RIFERIMENTO' | 'genera_dati.py' | 30/06/2026 | Fine della finestra di osservazione scelta
| 'QUOTA_MANCANTI' | 'genera_dati.py" | 0,025 | Probabilità che una cella sia vuota
| 'K_SCELTO' | 'analisi_clustering.py' | 4 | Numero di cluster del modello finale

# Struttura della carteòòa

allenaora_segmentazione/
|-- genera_dati.py
|-- analisi_clustering.py
|-- dashboard.py
|-- requirements.txt
|-- README.md
|-- data/
|-- output/

# Tempi di esecuzione
-installazione delle librerie e avvio dell'ambiente virtuale: qualche minuto
-Generatore e analisi: pochi secondi
