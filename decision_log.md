# Decision Log – Progetto DL-Quantum

Questo registro traccia le decisioni progettuali, architetturali e metodologiche adottate durante lo sviluppo del progetto.

---

## [2026-10-19] - Setup Iniziale e Struttura di Collaborazione

- **Decisione**: Strutturazione del progetto in una cartella dedicata `dl-quantum` configurata come repository Git pubblico.
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

- **Esito Verifiche Integrità & Fisica Quantistica**:
  - Shape corretta `(400400, 56)` e tensore strutturato `(400, 1001, 55)`.
  - Assenza totale di valori `NaN` o `Inf` (0 rilevati).
  - 400 traiettorie con campionamento a $\Delta t = 0.02$ e $T_{fin} = 20.0$.
  - Range delle 55 osservabili rigorosamente confinato nell'intervallo fisico $[-1, 1]$.
  - Verifica della matrice di densità a due corpi: calcolo delle probabilità quantistiche congiunte $P(s_i, s_j) = \frac{1}{4}(1 + s_i \langle \sigma_z^i \rangle + s_j \langle \sigma_z^j \rangle + s_i s_j \langle \sigma_z^i \sigma_z^j \rangle) \ge 0$. Minimo riscontrato $0.1553 > 0$, con **0 violazioni fisiche** su oltre 72 milioni di verifiche ($400 \times 1001 \times 45 \times 4$).
  - Disaccoppiamento iniziale a $t=0$: $|c_{ij} - m_i m_j|$ con $\max = 0.0800$ e $\text{media} = 0.0158$, a conferma di stati iniziali quasi di prodotto (product states) fisicamente realistici.
  - Generata cache binaria `data/cache/trajectories.npy` (88.1 MB, float32) con verifica di identità bit-a-bit rispetto al CSV, per caricamenti istantanei nelle fasi successive.
- **Split per Traiettoria (Seed 42)**:
  - Generata la partizione disgiunta e completa: 280 Train (70%), 60 Validation (15%), 60 Test (15%).
  - Salvato in `results/split_indices.npz`. Questo file è fissato e garantisce l'assenza totale di data leakage temporale.
- **Decisione Normalizzazione**: Dati grezzi (`none`) con attivazione finale `tanh`.
- **Motivazione Sperimentale & Fisica**:
  - *Confronto empirico su 3 seed (42, 43, 44)*:
    - `none` (`tanh`): val MSE = $0.00022 \pm 0.00000$ (mag: $0.00011$, corr: $0.00024$).
    - `zscore` (lineare): val MSE = $0.00023 \pm 0.00000$ (mag: $0.00013$, corr: $0.00026$).
    - La differenza è statisticamente significativa (intervalli di confidenza al 95% con $t$ di Student disgiunti a favore di `none`).
  - *Motivazione fisica*: Le magnetizzazioni e le correlazioni quantistiche sono valori di aspettazione di operatori di Pauli, vincolati per definizione a $[-1, 1]$. L'attivazione `tanh` rispetta per costruzione il dominio fisico, prevenendo estrapolazioni non fisiche e drift a lungo termine in inferenza ricorsiva (free-running).
  - *Preservazione della scala per i Transformer (M3/M4)*: Lo z-score per-feature riscalerebbe artificialmente le correlazioni a lungo raggio quasi piatte (es. oscillanti tra $0.01$ e $0.05$), amplificandone il rumore di fondo e alterando i prodotti scalari $\mathbf{Q}\mathbf{K}^T$ nei meccanismi di attenzione. I dati grezzi preservano invece la fedele gerarchia di ampiezza fisica tra le 55 osservabili.
  - Configurazione confermata: `CFG["normalization"] = "none"` e `CFG["output_activation"] = "tanh"` in `src/config.py`.

---

## [2026-10-20] - Finestrature Temporali, Allineamento Target (Sfida 1) e Super-Risoluzione (Sfida 2)

- **Decisione**: Fissaggio delle due finestrature standard `short` ($T_{in}=10 \to T_{out}=40$) e `long` ($T_{in}=20 \to T_{out}=40$) con $\text{stride}=10$.
- **Motivazione**: Mantenere costante l'orizzonte futuro $T_{out} = 40$ ($0.80$ unità fisiche) consente un confronto pulito in cui l'unica variabile indipendente tra `short` e `long` è la lunghezza della memoria passata osservata.
- **Decisione**: Introduzione dell'allineamento target `align_t_in = 40` per la griglia $T_{in} \in \{5, 10, 20, 40\}$ (Sfida 1).
- **Motivazione**: Senza allineamento, le finestre con $T_{in}$ corto valuterebbero la dinamica transiente iniziale ($t \in [0.10, 0.90]$), mentre quelle con $T_{in}=40$ valuterebbero regimi temporali più tardivi ($t \in [0.80, 1.60]$), introducendo un forte bias di confronto. Con `align_t_in=40`, tutti i $T_{in}$ condividono rigorosamente lo stesso identico set di target $Y$, garantendo una valutazione statistica impeccabile.
- **Decisione**: Generazione in-memory vettorizzata con `numpy.lib.stride_tricks.sliding_window_view`.
- **Motivazione**: L'estrazione dell'intero dataset richiede solo ~330 ms in RAM per 26.880 finestre. È stato scartato il salvataggio persistente su disco, che avrebbe sprecato oltre 500 MB di file ridondanti senza alcun beneficio prestazionale tangibile.
- **Decisione**: Subsampling $\times 2$ continuo per la Super-Risoluzione (Sfida 2).
- **Motivazione**: Campionando $T_{in} = 20$ punti radi a $\Delta t = 0.04$, il contesto copre uno span nativo continuo di 39 step. La funzione `make_sr_windows` estrae contestualmente il segnale continuo denso a $\Delta t = 0.02$, abilitando la visualizzazione e la validazione diretta del super-risolutore.

