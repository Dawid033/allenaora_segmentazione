"""Genera il dataset simulato dei clienti di AllenaOra (Iscritti tra liglio 2025 e giugno 2026)
Produce data/clienti-csv e data/segmenti_veri.csv
Uso: python genera_dati.py ed utilizzando lo stesso SEED il dataset è
sempre identico"""

import calendar
from datetime import date
import numpy as np
from pathlib import Path
import pandas as pd

GIORNI_PER_MESE = 365/12

SEED = 42
N_CLIENTI = 500
MOLTIPLICATORE_CATEGORIA = { #ritocco di prezzo dovuto alla categoria 
    #tramite moltiplicatore percentuale
    "Forza e tonificazione" : 1.0,
    "Cardio/HIIT": 1.0,
    "Yoga e Pilates": 0.9,
    "Postura e riabilitazione": 1.1,
}
CATEGORIE = list(MOLTIPLICATORE_CATEGORIA.keys())
ZONE = ["Nord-Ovest", "Nord-Est", "Centro", "Sud", "Isole"] #Stessa probabilità x zona
QUOTA_MANCANTI= 0.025
CARTELLA_DATI = Path(__file__).parent / "data"
DATA_RIFERIMENTO = date(2026,6,30)
#creo il dizionario dei segmenti/tipi_di_clienti ipotizzati
SEGMENTO_CLIENTI = {
    "Costanti": {
        "quota": 0.25,
        "mesi_tipici": [(2025,7), (2025,8), (2025,9), (2025,10), (2025,11), (2025,12)],
        "quota_tipici": 0.75,
        "mesi_ammessi": [(2026,1),(2026,2)],
        "ritmo_mensile": (7,10),
        "durata_attiva": None, #regola non necessaria a questo segmento
        "sessioni_totali": None, #regola non necessaria a questo segmento
        "giorni_ultima_sessione": (1,15),
        "prezzo_base": (30,38), #prezzo base per sessione del segmento
        "prob_categorie": [0.30,0.20,0.20,0.30], #Forza e tonificazione, Cardio/HIIT, Yoga e pilates, Postura e riabilitazione
    },
    "Prova_costume": {
        "quota": 0.20,
        "mesi_tipici": [(2026,3), (2026,4), (2026,5)],
        "quota_tipici":1,
        "mesi_ammessi":[],
        "ritmo_mensile": (8,12), 
        "durata_attiva": None,#regola non necessaria a questo segmento
        "sessioni_totali": None, #regola non necessaria a questo segmento
        "giorni_ultima_sessione": (1,25),
        "prezzo_base": (42,55), #prezzo base per sessione del segmento
        "prob_categorie": [0.35,0.35,0.20,0.10], #Forza e tonificazione, Cardio/HIIT, Yoga e pilates, Postura e riabilitazione
    },
    "Occasionali": {
        "quota": 0.30,
        "mesi_tipici": [(2025,7), (2025,8), (2025,9), (2025,10), (2025,11), (2025,12),(2026,1), (2026,2), (2026,3), (2026,4)],
        "quota_tipici":1,
        "mesi_ammessi": [],
        "ritmo_mensile": (0.5,1.5),
        "durata_attiva": None, #regola non necessaria a questo segmento
        "sessioni_totali": None, #regola non necessaria a questo segmento
        "giorni_ultima_sessione": (20,90),
        "prezzo_base": (22,30), #prezzo base per sessione del segmento
        "prob_categorie": [0.20,0.25,0.40,0.15], #Forza e tonificazione, Cardio/HIIT, Yoga e pilates, Postura e riabilitazione
    },
    "Ripartenze_mancate": {
        "quota": 0.25,
        "mesi_tipici": [(2026,1), (2025,9)],
        "quota_tipici": 0.75,
        "mesi_ammessi": [(2025,10),(2025,11),(2025,12)],
        "ritmo_mensile": None,
        "durata_attiva": (14,45),
        "sessioni_totali": (2,8),
        "giorni_ultima_sessione": None,
        "prezzo_base": (35,45), #prezzo base per sessione del segmento
        "prob_categorie": [0.35,0.35,0.25,0.05], #Forza e tonificazione, Cardio/HIIT, Yoga e pilates, Postura e riabilitazione
    },
}

for nome, regole in SEGMENTO_CLIENTI.items():
    for anno, mese in regole["mesi_tipici"] + regole["mesi_ammessi"]:
        if not ((2025, 7) <= (anno, mese) <= (2026,6)):
            print("Mese fuori dalla finestra di osservazione", nome,anno,mese)
print("Controllo mesi completo")

for nome, regole in SEGMENTO_CLIENTI.items():
    p = regole["prob_categorie"]
    if len(p) !=len(CATEGORIE) or abs(sum(p)-1)> 1e-9:
        print("Probabilità delle categorie non valide", nome,p)
print("Controllo categorie completo") 

"""Estrae la data di iscrizione di un cliente secondo le regole del suo segmento
    Questa estrazione avviene in due passi: prima il mese, scelto tra i mesi tipici
    della categoria , altrimenti un mese tra quelli ammessi; poi viene estratto il giorno"""
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

#riestrae un valore da una normale troncata
def normale_troncata(rng,minimo,massimo, limite_superiore=None):
    media = (minimo+massimo)/2
    dev_std = (massimo-minimo) /4
    
    if limite_superiore is None:
        limite_superiore = massimo
    if limite_superiore < minimo:
        raise ValueError("limite_superiore minore del minimo") #nessun valore ammesso

    valore = rng.normal(media, dev_std)
    while valore < minimo or valore > limite_superiore:
        valore = rng.normal(media,dev_std)
    return valore

def poisson_troncata(rng,minimo,massimo):
    lam = (minimo + massimo)/2
    valore = rng.poisson(lam)
    while valore < minimo or valore > massimo:
        valore = rng.poisson(lam)
    return int(valore)

def estrai_num_sessioni(rng, regole, anzianita):
    minimo, massimo = regole["ritmo_mensile"]
    ritmo= rng.uniform(minimo,massimo)
    mesi= anzianita / GIORNI_PER_MESE
    lam= ritmo * mesi
    n=rng.poisson(lam)
    return max(1,int(n)) 

#Estrae la categoria preferita con le probabilità del segmento
def estrai_categoria(rng,regole):
    return str(rng.choice(CATEGORIE, p=regole["prob_categorie"])) 

#Prezzo per sessione: base del segmento (normale troncata, D18) con il ritocco della categoria (D04).
def estrai_prezzo_sessione(rng, regole, categoria):
    minimo, massimo = regole["prezzo_base"]
    prezzo_base = normale_troncata(rng, minimo, massimo)
    return round(float(prezzo_base) * MOLTIPLICATORE_CATEGORIA[categoria], 2)

#Rende vuoto ogni valore con probabilità quota, restituendo la lista nuova ed il numero di mancanti
def inserisci_mancanti(rng,valori,quota):
    mancante= rng.random(len(valori)) < quota
    nuovi = [None if m else v for v, m in zip(valori,mancante)]
    return nuovi, int(mancante.sum())

#stampa per ogni segnmento i controlli attesi
def controlla_segmenti(segmenti, date_iscrizione, anzianita, num_sessioni, 
                       giorni_ultima_sessione, categorie, importo_totale):
    for nome, regole in SEGMENTO_CLIENTI.items():
        date_seg = [d for d, s in zip(date_iscrizione, segmenti) if s == nome]
        anz_seg = [a for a, s in zip(anzianita, segmenti) if s == nome]
        n_tipici = sum((d.year, d.month) in regole["mesi_tipici"] for d in date_seg)
        quota_tipici = n_tipici / len(date_seg)
        print(nome, min(date_seg), max(date_seg), min(anz_seg), max(anz_seg), round(quota_tipici, 2))

        giorni_seg = [g for g, s in zip(giorni_ultima_sessione, segmenti) if s == nome]
        violazioni = sum(g > a for g, a in zip(giorni_seg, anz_seg))
        sotto_90 = sum(a<90 for a in anz_seg)
        print("giorni dall ultima sessione:", min(giorni_seg), max(giorni_seg),
               "oltre l'anzianita", violazioni,"anzianita <90", sotto_90)
        sess_seg = [n for n, s in zip(num_sessioni, segmenti) if s ==nome]
        n_uno = sum(n ==1 for n in sess_seg)
        print(" sessioni:", min(sess_seg), max(sess_seg), "con 1 sessione", n_uno)

        cat_seg = [c for c, s in zip(categorie, segmenti) if s == nome]
        print(" categorie:", [cat_seg.count(c) for c in CATEGORIE])
        medi_seg = [imp / n for imp, n, s in zip(importo_totale, num_sessioni, segmenti)
                    if s == nome]
        print(" importo medio:", round(min(medi_seg), 2), round(max(medi_seg), 2),
              round(sum(medi_seg) / len(medi_seg),2))

#Genera i clienti, esegue i controlli ed espora i due csv
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
    num_sessioni= []
    giorni_ultima_sessione= []
    categorie = []
    importo_totale= []

    for i, segmento in enumerate(segmenti, start=1):
        regole = SEGMENTO_CLIENTI[segmento]
        data =estrai_data_iscrizione(rng, regole)
        anz = (DATA_RIFERIMENTO - data).days
        id_clienti.append(f"C{i:04d}")
        date_iscrizione.append(data)
        anzianita.append(anz)

        if regole["ritmo_mensile"] is not None:
            num_sessioni.append(estrai_num_sessioni(rng, regole, anz))
            min_giorni, max_giorni = regole["giorni_ultima_sessione"]
            limite= min(max_giorni,anz)
            giorni= round(normale_troncata(rng, min_giorni, max_giorni, limite))
            giorni_ultima_sessione.append(giorni)
        else:
            minimo, massimo = regole["durata_attiva"]
            durata = round(normale_troncata(rng,minimo,massimo))
            min_sess, max_sess = regole["sessioni_totali"]
            num_sessioni.append(poisson_troncata(rng,min_sess,max_sess))
            giorni_ultima_sessione.append(anz - durata)

        categoria =estrai_categoria(rng,regole)
        categorie.append(categoria)

        prezzo = estrai_prezzo_sessione(rng, regole, categoria)
        importo_totale.append(round(num_sessioni[-1] * prezzo, 2))

    zone = [str(z) for z in rng.choice(ZONE, size=N_CLIENTI)]

    for j in range(5):
        print(id_clienti[j], segmenti[j], date_iscrizione[j], anzianita[j], num_sessioni[j])

    nomi, conteggi= np.unique(segmenti, return_counts=True)
    for nome, conteggio in zip(nomi, conteggi):
        print(nome, conteggio)
    print("Totale:", conteggi.sum())

    controlla_segmenti(segmenti, date_iscrizione, anzianita, num_sessioni, giorni_ultima_sessione,
                       categorie,importo_totale)

    valori_zone, conteggi_zone = np.unique(zone, return_counts=True)
    for z,c in zip(valori_zone, conteggi_zone):
        print("zona", z,c)

   #valori mancanti dopo i controlli che non gestiscono None
    importo_csv, n_imp = inserisci_mancanti(rng,importo_totale, QUOTA_MANCANTI)
    giorni_csv, n_giorni = inserisci_mancanti(rng,giorni_ultima_sessione,QUOTA_MANCANTI)
    categorie_csv, n_cat = inserisci_mancanti(rng,categorie, QUOTA_MANCANTI)
    print("mancanti: importo", n_imp, "giorni", n_giorni, "categoria", n_cat)

    #tutto quello che riguarda l'esportazione
    CARTELLA_DATI.mkdir(exist_ok=True)
    clienti=pd.DataFrame({
        "id_cliente": id_clienti,
        "zona":zone,
        "data_iscrizione": date_iscrizione,
        "num_sessioni":num_sessioni,
        "importo_totale":importo_csv,
        "giorni_ultima_sessione":giorni_csv,
        "categoria_preferita":categorie_csv,
    })
    clienti["giorni_ultima_sessione"] = clienti["giorni_ultima_sessione"].astype("Int64")
    clienti.to_csv(CARTELLA_DATI / "clienti.csv", index=False)

    segmenti_veri = pd.DataFrame({
        "id_cliente": id_clienti,
        "segmento": [str(s) for s in segmenti],
    })     
    segmenti_veri.to_csv(CARTELLA_DATI/"segmenti_veri.csv", index=False)
    print("Salvati", len(clienti), "clienti in", CARTELLA_DATI)

if __name__ == "__main__":
    main()
