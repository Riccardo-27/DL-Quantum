# Decision Log – Progetto DL-Quantum

Questo registro traccia le decisioni progettuali, architetturali e metodologiche adottate durante lo sviluppo del progetto.

---

## [2026-10-19] - Setup Iniziale e Struttura di Collaborazione

- **Decisione**: Strutturazione del progetto in una cartella dedicata `dl-quantum` destinata a diventare repository Git privato.
- **Motivazione**: I file pesanti del dataset (`trajectories.csv`, ~464 MB) e le dispense PDF devono rimanere fuori dal versionamento Git per limiti di dimensione di GitHub (100 MB max) e ordine del codice.
- **Workflow in due**: Moduli condivisi posizionati in `src/`, branch separati con PR su `main` e divisione dei notebook `.ipynb` per evitare merge conflict su file JSON binari.
- **Ambiente di esecuzione**:
  - *Locale*: Virtual environment Python (`.venv`) per sviluppo moduli, linting e test unitari veloci.
  - *Addestramento*: Google Colab con runtime GPU (T4), con montaggio di Google Drive per il dataset `trajectories.csv` e persistenza di checkpoint/risultati.
- **Divisione del lavoro concordata**:
  - *Collega A*: `data.py`, `windowing.py`, `metrics.py`, `stats.py`, verifiche preliminari, baseline e modello RNN (M1/M2), Sfida 1 (finestra minima), Analisi disaggregata, Table 1 e Report.
  - *Collega B*: `config.py`, `seeding.py`, `results_log.py`, `plots.py`, train loop ibrido TF+MM+SS (`training.py`), modello Transformer (M3/M4), Sfida 2 (Super-Resolution), Sfida 3 (Attention Maps), Demo di inferenza e Slide.

---

## [2026-10-19] - Architettura dei Moduli Comuni (`src/`)

- **Decisione**: Configurazione centralizzata in dizionario piatto (`src/config.py`).
- **Motivazione**: Permette di serializzare l'intera configurazione di ogni run direttamente in una riga di `results_log.csv` senza dipendere da file esterni complessi durante l'esecuzione su Colab.
- **Decisione**: Seed centralizzato con determinismo opzionale in `src/seeding.py`.
- **Motivazione**: Garantire la perfetta riproducibilità numerica degli esperimenti. `set_seeds` imposta `random`, `numpy`, `PYTHONHASHSEED` e `tf.random` abilitando le op deterministiche su GPU.
- **Decisione**: Logging separato per autore `results_log_<autore>.csv` con merge automatico in lettura.
- **Motivazione**: Evita conflitti di scrittura concorrente quando entrambi i colleghi lanciano esperimenti contemporaneamente su Google Drive.
- **Decisione**: Statistica con t di Student anziché normale ($n=3$).
- **Motivazione**: Con soli 3 seed ($n=3$), l'intervallo di confidenza al 95% calcolato con 1.96 sottostima significativamente l'incertezza. Il quantile corretto è $t_{0.975, 2} \approx 4.303$.
- **Decisione**: Finestre vincolate per singola traiettoria (`src/windowing.py`).
- **Motivazione**: Lo scorrimento delle finestre temporali viene eseguito strettamente all'interno di ciascuna traiettoria per impedire che una finestra colleghi la fine di una traiettoria con l'inizio di quella successiva (prevenzione data leakage).

---

## [2026-10-19] - Validazione Dati, Partizionamento e Scelta della Normalizzazione

- **Esito Verifiche Integrità**:
  - Shape corretta `(400400, 56)`.
  - Assenza totale di valori `NaN` (0 rilevati).
  - 400 traiettorie con campionamento a $dt=0.02$ e $T_{fin}=20.0$.
  - Range dei dati confinato nell'intervallo fisico $[-1, 1]$.
  - Verifica coerenza fisica delle 45 correlazioni con probabilità congiunte $P(s_i, s_j) \ge 0$ (0 violazioni oltre la soglia numerica).
- **Split per Traiettoria (Seed 42)**:
  - Generata la partizione disgiunta e completa: 280 Train, 60 Validation, 60 Test.
  - Salvato in `results/split_indices.npz`. Questo file è fissato e garantisce l'assenza totale di data leakage.
- **Decisione Normalizzazione**: Dati grezzi (`none`) con attivazione finale `tanh`.
- **Motivazione Sperimentale & Fisica**:
  - *Confronto empirico su 3 seed (42, 43, 44)*:
    - `none` (`tanh`): val MSE = $0.00022 \pm 0.00000$ (mag: $0.00011$, corr: $0.00024$).
    - `zscore` (lineare): val MSE = $0.00023 \pm 0.00000$ (mag: $0.00013$, corr: $0.00026$).
    - La differenza è statisticamente significativa (intervalli di confidenza al 95% disgiunti a favore di `none`).
  - *Motivazione fisica*: Le magnetizzazioni e le correlazioni quantistiche sono vincolate per definizione a $[-1, 1]$. L'attivazione `tanh` rispetta per costruzione il dominio fisico, mentre l'uscita lineare dello z-score può estrapolare valori non fisici ($<-1$ o $>1$).
  - Configurazione confermata: `CFG["normalization"] = "none"` e `CFG["output_activation"] = "tanh"` in `src/config.py`.
