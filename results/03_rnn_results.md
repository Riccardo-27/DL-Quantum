# 📊 Sintesi dei Risultati – Notebook 03: Modelli Ricorrenti (RNN), Confronto M1 vs M2 e Sfida 1

Questo documento riassume in modo **semplice, intuitivo e dettagliato** tutti i risultati sperimentali, i confronti architetturali e le metriche numeriche ottenute dall'addestramento dei modelli ricorrenti (`notebooks/03_rnn.ipynb`).

---

## 1. 🎯 Qual era l'obiettivo del Notebook?
In questo notebook abbiamo addestrato e confrontato le prime due famiglie di modelli di Deep Learning del progetto:
1. **M1 – RNN Diretta (One-Shot)**: emette tutti i 40 passi futuri in un colpo solo tramite una testa densa (nessun ciclo ricorsivo durante la previsione).
2. **M2 – RNN Seq2Seq (Encoder-Decoder)**: genera la previsione passo dopo passo, reiniettando la predizione precedente come input del timestep successivo (*free-running autoregressivo*).

Abbiamo valutato entrambe le famiglie su:
* **3 configurazioni di iperparametri** (HP1: LSTM 1L 64u, HP2: LSTM Stacked 2L 64u con dropout 0.1, HP3: GRU 1L 64u);
* **2 finestre temporali** (`short` $T_{in}=10$ e `long` $T_{in}=20$);
* La **Sfida 1** (scaling dell'errore con contesto passato $T_{in} \in \{5, 10, 20, 40\}$ su target allineati);
* Il **train loop ibrido in 3 fasi**: Teacher Forcing (10 ep.) $\to$ Masked Modeling (10 ep.) $\to$ Scheduled Sampling (5 ep.).

---

## 2. 🏆 Tabella Comparativa Completa dei Risultati (12 Modelli)

Tutti i modelli sono stati addestrati su GPU T4 (Colab) con attivazione d'uscita fisica $\tanh$ e valutati sulla scala reale originaria:

| Famiglia | Modello | Configurazione HP | Finestra | $T_{in}$ | Val MSE | Val MAE | Val RMSE | Tempo Training |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **M1** | **RNN Diretta** | **HP1 (LSTM 64u - Baseline)** | **`short`** | 10 | **0.000161** | 0.00912 | 0.01268 | 46.9 s |
| **M1** | RNN Diretta | HP2 (LSTM 2L - Stacked) | `short` | 10 | 0.000174 | 0.01035 | 0.01375 | 64.3 s |
| **M1** | RNN Diretta | HP3 (GRU 64u - Compatta) | `short` | 10 | 0.000163 | 0.00913 | 0.01275 | 44.2 s |
| **M2** | RNN Seq2Seq | HP1 (LSTM 64u - Baseline) | `short` | 10 | 0.000305 | 0.01231 | 0.01746 | 85.0 s |
| **M2** | RNN Seq2Seq | HP2 (LSTM 2L - Stacked) | `short` | 10 | 0.000266 | 0.01207 | 0.01631 | 137.3 s |
| **M2** | RNN Seq2Seq | HP3 (GRU 64u - Compatta) | `short` | 10 | 0.000341 | 0.01321 | 0.01847 | 79.4 s |
| **M1** | **RNN Diretta** | **HP1 (LSTM 64u - Migliore Assoluto)** | **`long`** | 20 | **0.000147** | **0.00874** | **0.01212** | 50.3 s |
| **M1** | RNN Diretta | HP2 (LSTM 2L - Stacked) | `long` | 20 | 0.000157 | 0.00982 | 0.01253 | 69.6 s |
| **M1** | RNN Diretta | HP3 (GRU 64u - Compatta) | `long` | 20 | 0.000159 | 0.00901 | 0.01261 | 48.0 s |
| **M2** | RNN Seq2Seq | HP1 (LSTM 64u - Baseline) | `long` | 20 | 0.000280 | 0.01180 | 0.01673 | 86.8 s |
| **M2** | RNN Seq2Seq | HP2 (LSTM 2L - Stacked) | `long` | 20 | 0.000242 | 0.01140 | 0.01556 | 142.9 s |
| **M2** | RNN Seq2Seq | HP3 (GRU 64u - Compatta) | `long` | 20 | 0.000333 | 0.01280 | 0.01825 | 83.7 s |

---

## 3. 💡 Le 3 Conclusioni Fondamentali per il Report Tecnico

### 1. Perché M1 (RNN Diretta) batte nettamente M2 (Seq2Seq)?
* **M1 Diretta** ottiene un MSE nettamente migliore (**$0.000147$** vs **$0.000242$**, circa il 40% di errore in meno).
* *Motivazione intuitiva*: M1 emette tutte le 40 previsioni contemporaneamente in un unico passaggio in avanti. Al contrario, M2 deve generare un passo alla volta e riutilizzare la propria predizione come input per il passo successivo. Questo genera il fenomeno dell'**Exposure Bias** (l'accumulo progressivo di piccoli errori che crescono lungo i 40 passi futuri).

### 2. Perché la finestra `long` ($T_{in}=20$) batte `short` ($T_{in}=10$)?
* Sia per M1 che per M2, passare da 10 a 20 passi di storia passata migliora le prestazioni (l'MSE di M1 cala da $0.000161$ a $0.000147$).
* *Motivazione fisica*: Le oscillazioni quantistiche del modello PXP (le *quantum scars*) hanno una periodicità ben definita; una finestra passata più lunga permette all'encoder di osservare un ciclo completo di oscillazione, comprendendo meglio la fase del sistema.

### 3. Qual è l'architettura vincente tra HP1, HP2 e HP3?
* **HP1 (LSTM a 1 layer, 64 unità)** si conferma la scelta ottimale: ottiene l'errore minimo assoluto ed è circa il **30% più veloce** rispetto a HP2 (47-50 secondi vs 65-70 secondi).
* Aggiungere un secondo layer ricorrente (HP2) non migliora le prestazioni su M1 (leggero overfitting, compensato dal dropout), mentre in M2 aiuta ad assorbire meglio la complessità dell'interfaccia encoder-decoder.

---

## 4. 🎯 Sfida 1: Ricerca della Finestra Minima Ottima ($T_{in} \in \{5, 10, 20, 40\}$)

Abbiamo addestrato il modello M1 vincente su 4 diverse lunghezze di contesto passato, mantenendo i target rigorosamente allineati (`align_t_in = 40`):

| Contesto $T_{in}$ (Passi / Tempo) | Validation MSE | Validation MAE | Considerazioni sulla Dinamica Quantistica |
| :---: | :---: | :---: | :--- |
| **5** ($0.10$ s) | **0.000171** | 0.00942 | Troppo corto per cogliere un ciclo completo di oscillazione |
| **10** ($0.20$ s) | **0.000161** | 0.00912 | **Ottimo compromesso**: cattura la tendenza locale con metà dei parametri |
| **20** ($0.40$ s) | **0.000147** | 0.00874 | **Finestra ottimale**: copre l'intero periodo dell'oscillazione quantistica principale |
| **40** ($0.80$ s) | **0.000148** | 0.00878 | Prestazioni identiche a $T_{in}=20$: raddoppiare ulteriormente la memoria non porta vantaggi |

👉 **Risoluzione della Sfida 1**: La finestra minima che garantisce la saturazione delle prestazioni è **$T_{in} = 20$ passi** ($0.40$ secondi).

---

## 5. 🔬 Analisi Disaggregata dell'Errore

Esaminando l'errore per ciascuna delle 55 componenti:
* **Magnetizzazioni vs Correlazioni**:
  * MSE medio sulle 10 magnetizzazioni: **$0.000075$** (errore bassissimo, dinamica molto liscia);
  * MSE medio sulle 45 correlazioni: **$0.000180$** (errore leggermente superiore a causa della natura non-locale delle interazioni a due corpi).
* **Distribuzione Spaziale lungo la catena**:
  * I qubit agli estremi della catena ($q_1$ e $q_{10}$) mostrano una dinamica più semplice da prevedere rispetto ai qubit centrali ($q_5, q_6$), che subiscono interazioni da entrambi i lati.
* **Correlazioni a lungo raggio**:
  * Le correlazioni tra atomi lontani (es. $c_{1,10}$) hanno un errore assoluto quasi nullo, confermando l'importanza di non aver gonfiato artificialmente il rumore con lo Z-score.

---

## 6. 📐 Validazione Statistica su 3 Seed Indipendenti

La configurazione migliore (M1 su finestra `short`) è stata replicata su 3 seed casuali indipendenti (42, 43, 44):
* Seed 42: MSE = $0.0001609$ | MAE = $0.009117$
* Seed 43: MSE = $0.0001609$ | MAE = $0.009140$
* Seed 44: MSE = $0.0001609$ | MAE = $0.009106$

**Stima con Intervallo di Confidenza al 95% ($t$ di Student con 2 g.d.l., $t_{0.975, 2} \approx 4.303$):**
$$\mathbf{Validation\ MSE} = \mathbf{0.000161 \pm 0.000001}$$
$$\mathbf{Validation\ MAE} = \mathbf{0.009121 \pm 0.000043}$$

La variabilità tra run è praticamente trascurabile ($\pm 0.000001$), a testimonianza dell'estrema stabilità del train loop ibrido.

---

## 7. 📁 File e Modelli Generati

Tutti gli artefatti sono salvati nelle cartelle del progetto:
* **Pesi del Modello Ottimale**:
  * [`checkpoints/best_rnn_model.keras`](file:///c:/Users/ernes/Desktop/Deep%20Learning/dl-quantum/checkpoints/best_rnn_model.keras) (salvato su Google Drive, pronto per l'inferenza finale e la prova orale).
* **Log Sperimentale Concorrente**:
  * [`results/results_log_ernesto.csv`](file:///c:/Users/ernes/Desktop/Deep%20Learning/dl-quantum/results/results_log_ernesto.csv): tracciamento completo dei 15 run con iperparametri e metriche disaggregate.
* **Figure ad Alta Definizione in [`figures/03_rnn/`](file:///c:/Users/ernes/Desktop/Deep%20Learning/dl-quantum/figures/03_rnn/)**:
  1. `loss_curves_hybrid.jpeg`: andamento della loss attraverso le tre fasi TF $\to$ MM $\to$ SS.
  2. `free_running_error_accumulation.jpeg`: propagazione dell'errore nel tempo (Teacher Forcing vs Free-Running).
  3. `challenge1_tin_scaling.jpeg`: curva di scaling dell'errore in funzione della lunghezza del contesto passato $T_{in}$.
  4. `disaggregated_error_analysis.jpeg`: matrici di errore per qubit e mappe di calore delle correlazioni a due corpi.
