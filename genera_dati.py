import calendar
from datetime import date
import numpy as np

SEED = 42
N_CLIENTI = 500
DATA_RIFERIMENTO = date(2026,6,30)
#creo il dizionario
SEGMENTO_CLIENTI = {
    "Costanti": {
        "quota": 0.25,
        "mesi_tipici": [(2025,7), (2025,8), (2025,9), (2025,10), (2025,11), (2025,12)],
        "quota_tipici": 0.75,
        "mesi_ammessi": [(2026,1),(2026,2)],
        
    },
    "Prova_costume": {
        "quota": 0.20,
        "mesi_tipici": [(2026,3), (2026,4), (2026,5)],
        "quota_tipici":1,
        "mesi_ammessi":[],
            
        },
    "Occasionali": {
        "quota": 0.30,
        "mesi_tipici": [(2025,7), (2025,8), (2025,9), (2025,10), (2025,11), (2025,12),(2026,1), (2026,2), (2026,3), (2026,4)],
        "quota_tipici":1,
        "mesi_ammessi": [],
        },
    "Ripartenze_mancate": {
        "quota": 0.25,
        "mesi_tipici": [(2026,1), (2025,9)],
        "quota_tipici": 0.75,
        "mesi_ammessi": [(2025,10),(2025,11),(2025,12)],
        },
}

for nome, regole in SEGMENTO_CLIENTI.items():
    for anno, mese in regole["mesi_tipici"] + regole["mesi_ammessi"]:
        if not ((2025, 7) <= (anno, mese) <= (2026,6)):
            print("Mese fuori dalla finestra di osservazione", nome,anno,mese)
print("Controllo mesi completo")

def estrai_data_iscrizione(rng, regole):
    if rng.random() < regole["quota_tipici"]:
        mesi = regole["mesi_tipici"]
    else:
        mesi = regole["mesi_ammessi"]
    indice = rng.integers(0, len(mesi))
    anno, mese = mesi[indice]
    giorni_nel_mese = calendar.monthrange(anno,mese)[1]
    giorno = rng.integers(1, giorni_nel_mese + 1)
    return date(anno,mese,int(giorno))

def controlla_segmenti(segmenti, date_iscrizione, anzianita):
    for nome, regole in SEGMENTO_CLIENTI.items():
        date_seg = [d for d, s in zip(date_iscrizione, segmenti) if s == nome]
        anz_seg = [a for a, s in zip(anzianita, segmenti) if s == nome]
        n_tipici = sum((d.year, d.month) in regole["mesi_tipici"] for d in date_seg)
        quota_tipici = n_tipici / len(date_seg)
        print(nome, min(date_seg), max(date_seg), min(anz_seg), max(anz_seg), round(quota_tipici, 2))

def main():

    rng = np.random.default_rng(SEED)
    #Segmenti per tutti i vari clienti, prima di ogni estrazione
    nomi_segmenti = list(SEGMENTO_CLIENTI.keys())
    quote = [SEGMENTO_CLIENTI[nome]["quota"] for nome in nomi_segmenti]
    segmenti = rng.choice(nomi_segmenti, size=N_CLIENTI, p=quote)

    #dati di ciascun cliente
    id_clienti =[]
    date_iscrizione = []
    anzianita= []
    for i, segmento in enumerate(segmenti, start=1):
        data =estrai_data_iscrizione(rng, SEGMENTO_CLIENTI[segmento])
        id_clienti.append(f"C{i:04d}")
        date_iscrizione.append(data)
        anzianita.append((DATA_RIFERIMENTO - data).days)

    for j in range(5):
        print(id_clienti[j], segmenti[j], date_iscrizione[j], anzianita[j])

    nomi, conteggi= np.unique(segmenti, return_counts=True)
    for nome, conteggio in zip(nomi, conteggi):
        print(nome, conteggio)
    print("Totale:", conteggi.sum())

    controlla_segmenti(segmenti, date_iscrizione, anzianita)

if __name__ == "__main__":
    main()
