# DL-Quantum – Quantum Dynamics Forecasting

Previsione della dinamica quantistica di non-equilibrio nel modello PXP (catena unidimensionale a 10 qubit) mediante modelli di Deep Learning ricorrenti e basati su attenzione.

---

## 🔬 Descrizione del Problema e Dataset

Il progetto affronta la previsione temporale dello stato quantistico di una catena di 10 atomi di Rydberg governata dall'Hamiltoniana PXP, caratterizzata dal fenomeno delle *quantum many-body scars*.

* **Dimensioni del dataset**: 400 traiettorie simulate indipendenti, ciascuna composta da 1001 time step con passo temporale $\Delta t = 0.02$ ($t \in [0, 20]$).
* **55 Variabili fisiche osservabili**:
  * $10$ Magnetizzazioni locali $m_i(t) = \langle \sigma_i^z(t) \rangle \in [-1, 1]$ per ciascun qubit $i \in \{1, \dots, 10\}$.
  * $45$ Correlazioni a due punti connesse/disconnesse $c_{ij}(t) = \langle \sigma_i^z(t) \sigma_j^z(t) \rangle \in [-1, 1]$ per ogni coppia $i < j$.
* **Obiettivi e sfide**:
  1. **Previsione multi-orizzonte**: previsione di una finestra futura $T_{out} = 40$ istanti temporali a partire da contesti di lunghezza variabile $T_{in} \in \{5, 10, 20, 40\}$.
  2. **Super-risoluzione temporale**: ricostruzione della traiettoria continua a frequenza nativa ($\Delta t = 0.02$) a partire da input sottocampionato con passo dimezzato ($2 \times \Delta t = 0.04$).
  3. **Confronto architetturale**: modelli ricorrenti (LSTM / GRU) e modelli Transformer con cross-attention ed embedding temporale continuo.

---

## 📁 Struttura del Repository

```text
dl-quantum/
├── src/                      # Moduli Python riutilizzabili e testati
│   ├── config.py             # Configurazione centralizzata (iperparametri e percorsi)
│   ├── seeding.py            # Riproducibilità e fissaggio seed multi-framework
│   ├── data.py               # Lettura CSV, validazione fisica, split per traiettoria
│   ├── windowing.py          # Finestre temporali, super-resolution, controlli anti-leakage
│   ├── metrics.py            # Metriche globali e disaggregate (qubit, coppie, orizzonte)
│   ├── stats.py              # Statistiche e intervalli di confidenza 95% (t di Student)
│   ├── results_log.py        # Log concorrente su CSV separato per autore
│   ├── plots.py              # Esportazione grafici standardizzati JPEG (>= 300 dpi)
│   └── training.py           # Loop di addestramento ibrido (TF -> MM -> SS)
├── tests/                    # Test automatici di integrità della pipeline
│   └── test_pipeline.py      # Test determinismo, shapes, anti-leakage e metriche
├── notebooks/                # Notebook di analisi, esperimenti e visualizzazione
│   ├── 01_data.ipynb         # Esplorazione dati, split (280/60/60) e normalizzazione
│   ├── 02_windowing.ipynb    # Generazione finestre temporali e super-risoluzione
│   ├── 03_rnn.ipynb          # Modelli ricorrenti (LSTM / GRU)
│   ├── 04_transformer.ipynb  # Modelli Transformer (Encoder-Decoder)
│   └── 05_inference.ipynb    # Pipeline di inferenza rapida e benchmark finale
├── results/                  # File di log sperimentali e indici dello split (split_indices.npz)
├── figures/                  # Grafici JPEG ad alta risoluzione (>= 300 dpi)
├── checkpoints/              # Pesi dei modelli addestrati (.keras, esclusi da git)
├── report/                   # Relazione tecnica di progetto
├── slides/                   # Presentazione per l'esposizione orale
├── decision_log.md           # Registro delle decisioni architetturali e metodologiche
├── requirements.txt          # Dipendenze principali del progetto
└── requirements-lock.txt     # Snapshot completo per la riproducibilità esatta
```

---

## 🚀 Setup Locale

1. **Clonazione del repository**:
   ```bash
   git clone https://github.com/Riccardo-27/DL-Quantum.git
   cd DL-Quantum
   ```

2. **Creazione e attivazione del Virtual Environment**:
   ```powershell
   # Windows PowerShell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

3. **Installazione delle dipendenze**:
   ```powershell
   pip install -r requirements.txt
   # Per replicare l'ambiente esatto:
   # pip install -r requirements-lock.txt
   ```

4. **Esecuzione dei test di verifica**:
   ```powershell
   .venv\Scripts\pytest tests/test_pipeline.py -v
   ```

---

## ☁️ Esecuzione su Google Colab

Per configurare l'ambiente cloud, montare la cartella condivisa di Google Drive e gestire le credenziali Git in sicurezza, consulta la guida passo-passo dedicata: **[Guida Rapida Colab (SETUP_COLAB.md)](SETUP_COLAB.md)**.

Gli addestramenti pesanti e le sessioni di tuning vengono eseguiti su Google Colab con acceleratore GPU (T4):

1. **Collocazione dei dati**: Posizionare `trajectories.csv` (~464 MB) nella cartella Google Drive: `MyDrive/DL-Quantum/data/trajectories.csv`.
2. **Cella iniziale di setup in ciascun notebook**:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')

   import sys
   REPO_PATH = '/content/drive/MyDrive/DL-Quantum'
   if REPO_PATH not in sys.path:
       sys.path.insert(0, REPO_PATH)

   from src.seeding import set_seeds
   set_seeds(42)

   import tensorflow as tf
   print("GPU disponibile:", tf.config.list_physical_devices('GPU'))
   ```

---

## 👥 Flusso di Lavoro Collaborativo su GitHub

* **Repository pubblico**: Codice sorgente, notebook documentati e log numerici (`results/results_log_*.csv`) sono liberamente accessibili e versionati.
* **Separazione dati e codice**: Come definito in `.gitignore`, i file di grandi dimensioni (`trajectories.csv`, array binari `.npy` e checkpoint `.keras`) risiedono esclusivamente su Google Drive per non appesantire la cronologia Git.
* **Branching Model**: Sviluppo su branch dedicati alle funzionalità (`dev-ernesto`, `feat/...`) e unione sul branch `main` tramite Pull Request.
* **Metriche condivise**: Ogni autore scrive su un proprio log dedicato (`results/results_log_<autore>.csv`) per consentire run paralleli su Colab senza conflitti di concorrenza.
