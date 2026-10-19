# DL-Quantum – Quantum Dynamics Forecasting (9 CFU)

Progetto di Deep Learning per la previsione della dinamica quantistica nel modello PXP (10 qubit).

## 📁 Struttura del Progetto

```text
dl-quantum/
├── src/                      # Moduli Python riutilizzabili
│   ├── config.py             # Configurazione centralizzata (iperparametri e percorsi)
│   ├── seeding.py            # Riproducibilità e fissaggio seed
│   ├── data.py               # Lettura CSV, verifica integrità, split per traiettoria
│   ├── windowing.py          # Finestre temporali, super-resolution, decoder inputs
│   ├── metrics.py            # Metriche globali e disaggregate (per qubit, coppia, orizzonte)
│   ├── stats.py              # Media, std e intervallo di confidenza 95% (t di Student)
│   ├── results_log.py        # Append su CSV con gestione multi-autore
│   ├── plots.py              # Grafici standardizzati .jpeg a 300 dpi
│   └── training.py           # Loop di training ibrido parametrico (TF -> MM -> SS)
├── tests/                    # Test di integrità e Gate di avanzamento (S1, S2, ...)
│   └── test_s1_gate.py       # Gate S1 (seeding, shapes, leakage check, metriche)
├── notebooks/                # Notebook di analisi ed esperimenti
│   ├── 01_data.ipynb         # S2: Verifica dati, statistiche e split
│   ├── 02_windowing.ipynb    # S3: Finestre e controlli di leakage
│   ├── 03_rnn.ipynb          # S4-S7: Baseline, LSTM diretto e Seq2Seq
│   ├── 04_transformer.ipynb  # S5-S7: Transformer con Cross-Attention
│   └── 05_inference.ipynb    # S13: Cella di inferenza rapida per orale
├── results/                  # Risultati numerici (log, split_indices.npz)
├── figures/                  # Grafici JPEG esportati a 300 dpi
├── checkpoints/              # Checkpoint dei modelli addestrati (.keras)
├── report/                   # Relazione finale (max 10 pagine)
├── slides/                   # Presentazione per l'orale (5 minuti)
├── decision_log.md           # Registro decisionale di progetto
└── requirements.txt          # Dipendenze Python
```

## 🚀 Setup Locale

1. **Attivazione Virtual Environment**:
   ```powershell
   # Windows PowerShell
   .venv\Scripts\Activate.ps1
   ```
2. **Esecuzione Test Gate S1**:
   ```powershell
   .venv\Scripts\pytest tests/test_s1_gate.py -v
   ```

## ☁️ Esecuzione su Google Colab

I modelli pesanti e gli addestramenti vengono eseguiti su Google Colab con accelerazione GPU (T4):

1. **Dataset**: Caricare `trajectories.csv` (~464 MB) nella cartella Google Drive: `MyDrive/DL-Quantum/data/trajectories.csv`.
2. **Cella di Setup iniziale per ogni Notebook**:
   ```python
   # 1. Monta Drive
   from google.colab import drive
   drive.mount('/content/drive')

   # 2. Configura il path di importazione dei moduli src/
   import sys
   REPO_PATH = '/content/drive/MyDrive/DL-Quantum'
   if REPO_PATH not in sys.path:
       sys.path.insert(0, REPO_PATH)

   # 3. Fissa i seed e il determinismo
   from src.seeding import set_seeds
   set_seeds(42)

   # 4. Verifica GPU
   import tensorflow as tf
   print("GPU:", tf.config.list_physical_devices('GPU'))
   ```

## 👥 Flusso di Lavoro Collaborativo su GitHub

- **Repository**: Privato su GitHub.
- **Branching**: Branch separati per feature/esperimenti (es. `feat/rnn`, `feat/transformer`) con Pull Request su `main`.
- **Ownership Notebook**: Assegnare ciascun notebook `.ipynb` a un unico collega per evitare conflitti di merge sui file JSON dei notebook.
- **Dati & Checkpoint**: I file `trajectories.csv`, `.npy` e i modelli `.keras` sono esclusi da Git tramite `.gitignore` e risiedono esclusivamente su Google Drive.
