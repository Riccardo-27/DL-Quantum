# DL-Quantum – Quantum Dynamics Forecasting

Previsione della dinamica quantistica di non-equilibrio nel modello PXP (catena unidimensionale a 10 qubit Rydberg) mediante modelli di Deep Learning ricorrenti e basati su attenzione.

---

## 🔬 Descrizione del Problema e Dataset

Il progetto affronta la previsione temporale dello stato quantistico di una catena a molti corpi di 10 atomi di Rydberg governata dall'Hamiltoniana PXP, caratterizzata da oscillazioni quantistiche persistenti (*quantum many-body scars*).

* **Dimensioni del dataset**: 400 traiettorie simulate indipendenti, ciascuna composta da 1001 time step con passo temporale $\Delta t = 0.02$ ($t \in [0, 20]$).
* **55 Variabili fisiche osservabili**:
  * $10$ Magnetizzazioni locali $m_i(t) = \langle \sigma_i^z(t) \rangle \in [-1, 1]$ per ciascun qubit $i \in \{1, \dots, 10\}$.
  * $45$ Correlazioni a due punti connesse/disconnesse $c_{ij}(t) = \langle \sigma_i^z(t) \sigma_j^z(t) \rangle \in [-1, 1]$ per ogni coppia $i < j$.
* **Integrità fisica garantita**: Positività semidefinita delle matrici di densità a 2 corpi verificata su oltre 72 milioni di controlli ($P(s_i, s_j) \ge 0$, minimo $0.1553 > 0$).
* **Obiettivi e sfide**:
  1. **Previsione multi-orizzonte**: previsione accurata su un orizzonte futuro $T_{out} = 40$ istanti temporali ($0.80$ unità fisiche).
  2. **Sfida 1 (Finestra minima)**: determinazione della memoria passata minima necessaria tra $T_{in} \in \{5, 10, 20, 40\}$ con target futuro rigorosamente allineato.
  3. **Sfida 2 (Super-risoluzione temporale)**: ricostruzione della dinamica continua densa ($\Delta t = 0.02$) da ingressi radi sottocampionati a $\Delta t = 0.04$.
  4. **Sfida 3 (Interpretabilità fisica)**: estrazione delle mappe di attenzione per identificare principi fisici noti (cono di luce di Lieb-Robinson, blocco di Rydberg).

---

## 📁 Struttura del Repository

```text
dl-quantum/
├── src/                      # Moduli Python riutilizzabili, documentati e testati
│   ├── config.py             # Configurazione centralizzata (costanti fisiche e percorsi)
│   ├── seeding.py            # Riproducibilità e fissaggio seed multi-framework (CPU/GPU)
│   ├── data.py               # Lettura CSV, validazione fisica, caching .npy e split per traiettoria
│   ├── windowing.py          # Finestre temporali, super-resolution, target alignment e anti-leakage
│   ├── models_rnn.py         # Modelli ricorrenti: M1 (Diretto) e M2 (Seq2Seq LSTM/GRU con free-running)
│   ├── models_transformer.py # Modelli Transformer (Lab 7): M3 (Diretto) e M4 (Seq2Seq causale con attention scores)
│   ├── training.py           # Training loop ibrido progressivo a 3 fasi (TF -> MM -> SS)
│   ├── metrics.py            # Metriche globali e disaggregate (qubit, coppie, orizzonte)
│   ├── stats.py              # Statistiche e intervalli di confidenza 95% (t di Student con dof=2)
│   ├── results_log.py        # Log concorrente su CSV con merge automatico
│   └── plots.py              # Esportazione grafici standardizzati JPEG (>= 300 DPI)
├── tests/                    # Suite di test automatici integrati
│   └── test_pipeline.py      # 18 test completi (determinismo, shapes, fisica, modelli RNN/Transformer, training, plot)
├── notebooks/                # Notebook di analisi, esperimenti e visualizzazione
│   ├── 01_data.ipynb         # Esplorazione dati, split (280/60/60) e normalizzazione
│   ├── 02_windowing.ipynb    # Generazione finestre temporali e super-risoluzione
│   ├── 03_rnn.ipynb          # Modelli ricorrenti (LSTM / GRU), Sfida 1 e analisi disaggregata
│   ├── 04_transformer.ipynb  # Modelli Transformer (Start Small, Autoencoder, M3/M4 e Sfide 1-2-3)
│   └── 05_inference.ipynb    # Pipeline di inferenza rapida, Table 1, 5-Fold CV e demo per l'orale
├── results/                  # File di log sperimentali, Table 1 CSV, sintesi risultati (.md) e split_indices.npz
├── figures/                  # Grafici JPEG ad alta risoluzione (>= 300 DPI)
│   ├── 01_data/              # Figure esplorazione dataset, distribuzioni e confronto normalizzazione
│   ├── 02_windowing/         # Figure campionamento finestre, allineamento target e super-risoluzione
│   ├── 03_rnn/               # Curve di training RNN, accumulo errore free-running, Sfida 1 e analisi disaggregata
│   ├── 04_transformer/       # Curve Transformer, Start Small/AE, Sfide 1-2-3 (mappe di attenzione) e analisi disaggregata
│   └── 05_inference/         # Figure benchmark globale (Table 1), confronto Sfide 1-2 e predizione live sul Test Set
├── checkpoints/              # Directory per pesi addestrati (.keras, esclusi da git)
├── report/                   # Relazione tecnica di progetto
├── slides/                   # Presentazione per l'esposizione orale
├── decision_log.md           # Registro delle decisioni architetturali e metodologiche
└── requirements.txt          # Dipendenze complete unificate del progetto
```

---

## 📊 Stato di Avanzamento del Progetto

| Componente / Fase | Stato | Dettagli Principali |
| :--- | :---: | :--- |
| **Validazione & Preprocessing** | ✅ Completato | Split per traiettoria disgiunto 280/60/60 congelato in `results/split_indices.npz`. Decisione motivata di utilizzare **dati grezzi (`none`) con layer finale `tanh`** per preservare l'intervallo fisico $[-1, 1]$ e proteggere i prodotti scalari dell'attenzione. Generata cache `data/cache/trajectories.npy` (88 MB) per caricamento istantaneo. Sintesi in `results/01_data_results.md`. |
| **Finestrature Temporali & Sfida 2** | ✅ Completato | Implementate finestre standard `short` ($T_{in}=10 \to 40$) e `long` ($T_{in}=20 \to 40$) con $\text{stride}=10$. Introdotto allineamento target `align_t_in=40` per la Sfida 1. Generazione finestre di Super-Risoluzione con subsampling $2\times$ per la Sfida 2. Esportate 4 figure ad alta risoluzione in `figures/02_windowing/`. Sintesi in `results/02_windowing_results.md`. |
| **Modelli Ricorrenti (RNN)** | ✅ Completato | Implementati e validati **M1 (RNN Diretto Multi-Step)** e **M2 (RNN Seq2Seq Autoregressivo con LSTM/GRU)**. Addestramento completato con loop ibrido progressivo (**TF $\to$ MM $\to$ SS**). Miglior modello RNN: **M1 (HP1, LSTM 64u)** con $\text{Val MSE} = 0.000161 \pm 0.000001$ (`short`) e $0.000147$ (`long`). Studio della finestra minima (Sfida 1) e analisi disaggregata. Esportate 4 figure in `figures/03_rnn/`, log in `results/results_log_ernesto.csv` e sintesi in `results/03_rnn_results.md`. |
| **Modelli Transformer (M3/M4) & Sfide 1–2–3** | ✅ Completato | Implementati in `src/models_transformer.py` e validati in `notebooks/04_transformer.ipynb`: approccio *Start Small* e Autoencoder (Lab 6), **M3 (`TransformerDirect`)** e **M4 (`TransformerSeq2Seq` causale, Lab 7)**. Miglior modello assoluto: **M3 (HP3: $d=64, H=2, L=1$)** con **$\text{Val MSE} = 0.000121 \pm 0.000001$** (-25% rispetto a M1, zero violazioni fisiche $P \ge 0$). Risolte Sfida 1 ($T_{in}=40 \to \text{MSE } 0.000111$), Sfida 2 (Super-Resolution $\times 2 \to \text{MSE } 0.000121$) e Sfida 3 (mappe di `CrossAttention` e verifica `CausalSelfAttention`). Esportate 9 figure in `figures/04_transformer/` e sintesi in `results/04_transformer_results.md`. |
| **Benchmark Finale, 5-Fold CV & Inferenza (`05_inference.ipynb`)** | ✅ Completato | Implementato `notebooks/05_inference.ipynb`: aggregazione multi-autore, generazione di **Table 1** (`results/table1_benchmark_complete.csv`) con 95% CI ($t$ di Student) e valori migliori in grassetto, **5-Fold Cross-Validation per traiettoria** sui modelli vincitori (`M1 HP1` vs `M3 HP3`) con caching in `results/kfold_cv_results.csv`, grafici comparativi in `figures/05_inference/` e cella di **Demo Live sul Test Set Cieco** pronta per l'orale. |
| **Suite di Test Integrata** | ✅ Completato | **18 test su 18 passanti al 100%** in `tests/test_pipeline.py` (copertura completa su seeding, dati, finestre, metriche, fisica quantistica, modelli M1/M2/M3/M4, mappe di attenzione causali, ciclo ibrido e salvataggio `.keras`). |
| **Documentazione `src/` & Decision Log** | ✅ Completato | Tutti gli 11 moduli di `src/` documentati con Google-style docstrings. `decision_log.md` ristrutturato con focus accademico su Deep Learning e roadmap incrementale (*Cosa introduce / Limite teorico / Superamento*). |
| **Relazione Tecnica & Presentazione** | 📅 Pianificato | Stesura finale del report tecnico ($\le 10$ pagine in `report/`) e delle slide per l'esposizione orale (5 minuti in `slides/`). |

---

## 🚀 Setup Locale e Riproducibilità

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
   ```

4. **Esecuzione della suite completa di test (18 test)**:
   ```powershell
   pytest tests/test_pipeline.py -v
   ```

---

## ☁️ Esecuzione su Google Colab e Versionamento Multi-Utente

Gli addestramenti pesanti e le sessioni di validazione vengono eseguiti su Google Colab con acceleratore GPU (T4).

### 1. Differenziazione Automatica Autore e Branch tramite Segreti Colab
Per evitare conflitti di branch e sovrascritture su Google Drive quando entrambi i membri lavorano sullo stesso repository:
* **Ernesto**: aggiunge nel menu laterale sinistro di Colab (icona **Chiave 🔑 Segreti**) il segreto **`GIT_E_TOKEN`** contenente il proprio Personal Access Token GitHub.
  - Il notebook allinea automaticamente il repository sul branch **`dev-ernesto`**.
  - I log sperimentali vengono scritti automaticamente su `results/results_log_ernesto.csv`.
  - La cella finale di sincronizzazione effettua il push automatico su `origin dev-ernesto`.
* **Riccardo**: aggiunge nel menu laterale di Colab il segreto **`GIT_R_TOKEN`**.
  - Il notebook allinea automaticamente il repository sul branch **`dev-riccardo`**.
  - I log sperimentali vengono scritti su `results/results_log_riccardo.csv`.
  - La cella finale di sincronizzazione effettua il push automatico su `origin dev-riccardo`.

### 2. Struttura Dati su Google Drive
* Posizionare `trajectories.csv` (~464 MB) nel percorso standard di Drive: `MyDrive/DL-Quantum/data/trajectories.csv`.
* La cache binaria `data/cache/trajectories.npy` e le partizioni `results/split_indices.npz` vengono caricate e gestite automaticamente senza copie ridondanti.

---

## 👥 Flusso di Lavoro Collaborativo su GitHub

* **Repository pubblico**: Codice sorgente modulare in `src/`, notebook numerati e log numerici (`results/results_log_*.csv`) sono versionati e tracciati.
* **Separazione dati e codice**: Come definito in `.gitignore`, i file di grandi dimensioni (`trajectories.csv`, array binari `.npy` e checkpoint `.keras`) risiedono esclusivamente su Google Drive per non appesantire la cronologia Git.
* **Logging concorrente**: Ciascun autore scrive su un proprio file di log dedicato (`results/results_log_<autore>.csv`), aggregabile automaticamente in fase di confronto tramite `src/results_log.py` senza conflitti di merge.
* **Branching Model**: Sviluppo attivo isolato sui due branch (`dev-ernesto` e `dev-riccardo`) con allineamento e unione su `main` tramite Pull Request.
