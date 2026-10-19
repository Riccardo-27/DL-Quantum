# Decision Log – Progetto DL-Quantum (9 CFU)

Questo registro traccia formalmente ogni decisione progettuale, architetturale e metodologica, come richiesto dalla guida e dai criteri di valutazione del docente.

---

## [2026-10-19] - S0: Inizializzazione Repository e Setup di Collaborazione

- **Decisione**: Strutturazione del progetto in una cartella dedicata `dl-quantum` destinata a diventare repository Git privato.
- **Motivazione**: I file `trajectories.csv` (~464 MB) e le dispense PDF del docente devono rimanere fuori dal versionamento Git per limiti di dimensione di GitHub (100 MB max) e ordine del codice.
- **Workflow in due**: Moduli condivisi posizionati in `src/`, branch separati con PR su `main` e divisione dei notebook `.ipynb` per evitare merge conflict su file JSON binari.
- **Ambiente di esecuzione**:
  - *Locale*: Virtual environment Python (`.venv`) per sviluppo moduli, linting e test unitari veloci (S1 Gate).
  - *Addestramento*: Google Colab con runtime GPU (T4), con montaggio di Google Drive per il dataset `trajectories.csv` e persistenza di checkpoint/risultati.
- **Divisione del lavoro concordata (orientativa per orale e sviluppo)**:
  - *Collega A*: `data.py`, `windowing.py`, `metrics.py`, `stats.py`, verifiche S2-S3, baseline e modello RNN (M1/M2), Sfida 1 (finestra minima), Analisi disaggregata (S12), Table 1 e Report.
  - *Collega B*: `config.py`, `seeding.py`, `results_log.py`, `plots.py`, train loop ibrido TF+MM+SS (`training.py`), modello Transformer (M3/M4), Sfida 2 (Super-Resolution), Sfida 3 (Attention Maps), Demo di inferenza (S13) e Slide.

---

## [2026-10-19] - S1: Architettura dei Moduli `src/`

- **Decisione**: Configurazione centralizzata in dizionario piatto (`src/config.py`).
- **Motivazione**: Permette di serializzare l'intera configurazione di ogni run direttamente in una riga di `results_log.csv` senza dipendere da file esterni complessi durante l'esecuzione su Colab.
- **Decisione**: Seed centralizzato con determinismo opzionale in `src/seeding.py`.
- **Motivazione**: La guida del professore sancisce: *"Senza risultati identici il progetto non viene valutato"*. `set_seeds` imposta `random`, `numpy`, `PYTHONHASHSEED` e `tf.random` abilitando le op deterministiche su GPU.
- **Decisione**: Logging separato per autore `results_log_<autore>.csv` con merge automatico in lettura.
- **Motivazione**: Evita conflitti di scrittura concorrente quando entrambi i colleghi lanciano esperimenti contemporaneamente su Google Drive.
- **Decisione**: Statistica con t di Student anziché normale ($n=3$).
- **Motivazione**: Con soli 3 seed ($n=3$), l'intervallo di confidenza al 95% calcolato con 1.96 sottostima significativamente l'incertezza. Il quantile corretto è $t_{0.975, 2} \approx 4.303$.
- **Decisione**: Finestre vincolate per singola traiettoria (`src/windowing.py`).
- **Motivazione**: Lo scorrimento delle finestre temporali viene eseguito strettamente all'interno di ciascuna traiettoria per impedire che una finestra colleghi la fine di una traiettoria con l'inizio di quella successiva (prevenzione data leakage).
