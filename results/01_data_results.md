# 📊 Sintesi dei Risultati – Notebook 01: Esplorazione, Preprocessing e Validazione Dati

Questo documento riassume in modo **semplice, intuitivo e pratico** tutto ciò che abbiamo fatto e scoperto durante l'esecuzione del primo notebook (`notebooks/01_data.ipynb`).

---

## 1. 🎯 Qual era l'obiettivo del Notebook?
Questo notebook rappresenta le **fondamenta condivise di tutto il progetto**. Prima di addestrare qualsiasi rete neurale sequenziale (LSTM, GRU o Transformer), dovevamo:
1. Verificare che i dati grezzi fossero puliti e privi di anomalie;
2. Capire il significato delle 55 variabili e come oscillano nel tempo;
3. Creare una cache veloce per velocizzare i notebook futuri;
4. Suddividere i dati in **Train, Validation e Test** in modo corretto per evitare il *data leakage temporale*;
5. Risolvere un dubbio chiave: *conviene normalizzare i dati (Z-score) o lasciarli grezzi con attivazione finale `tanh`?*

---

## 2. 🗂️ Carta d'Identità del Dataset

I dati provengono da una simulazione fisica a molti corpi su una catena di **10 atomi quantistici (qubit)**:

| Caratteristica | Valore Reale dai Dati | Cosa significa in pratica |
| :--- | :---: | :--- |
| **File originale** | `trajectories.csv` (~464 MB) | Il dataset grezzo scaricato dal corso |
| **Numero di traiettorie** | **400** | 400 simulazioni fisiche indipendenti |
| **Punti temporali per traiettoria** | **1001** passi | Ciascuna traiettoria va da $t = 0.0$ a $t = 20.0$ |
| **Passo temporale ($\Delta t$)** | **0.02** | Il campionamento è perfettamente regolare e costante |
| **Totale righe nel CSV** | **400.400** | $400 \text{ traiettorie} \times 1001 \text{ punti temporali}$ |
| **Dati mancanti (`NaN` o `Inf`)** | **Esattamente 0** | Il dataset è completo al 100% senza buchi |

---

## 3. 🔍 Le 55 Feature: Cosa Rappresentano e Come si Comportano?

A ogni istante di tempo abbiamo un vettore di **55 variabili continue**, tutte comprese naturalmente tra **$[-1, 1]$**:

1. **10 Magnetizzazioni Locali ($m_1, \dots, m_{10}$)**:
   * Misurano l'orientazione media di ciascuno dei 10 atomi della catena.
   * *Comportamento osservato*: oscillano vistosamente tra valori positivi e negativi (con medie tra $-0.02$ e $-0.07$ e deviazioni standard ampie $\approx 0.03$).
2. **45 Correlazioni a Coppie ($c_{ij}$ per ogni coppia $i < j$)**:
   * Misurano il grado di sincronizzazione e interazione spaziale tra due atomi.
   * *Comportamento osservato*:
     * **Atomi vicini** (es. $c_{1,2}$, $c_{2,3}$): sentono una forte interazione reciproca e raggiungono picchi fino a $+0.25$, con medie positive attorno a $+0.095$.
     * **Atomi lontani agli estremi** (es. $c_{1,10}$, $c_{2,10}$): interagiscono debolmente, hanno oscillazioni piccolissime e una media praticamente nulla ($\approx 0.001$).

---

## 4. 🛡️ I Sanity Check di Integrità e Coerenza

Abbiamo sottoposto il dataset a controlli automatici stringenti prima di procedere:

* **Tutti i valori in $[-1, 1]$**: Confermato al 100%. Nessuna variabile esce dai limiti previsti.
* **Coerenza congiunta ($P \ge 0$)**: 
  Per verificare che le colonne del CSV non fossero scambiate o invertite di segno, abbiamo calcolato la probabilità congiunta reale per ogni coppia:
  $$P(s_i, s_j) = \frac{1 + s_i m_i + s_j m_j + s_i s_j c_{ij}}{4} \ge 0$$
  * *Risultato*: su oltre **72 milioni di controlli** ($400 \times 1001 \times 45 \times 4$), il minimo valore calcolato è stato **$0.1553 > 0$**, con **zero violazioni**. L'ordine delle feature nel CSV è formalmente garantito.
* **Stato iniziale disaccoppiato a $t=0$**: a inizio simulazione le correlazioni corrispondono al prodotto dei singoli atomi ($|c_{ij}(0) - m_i(0) m_j(0)| \le 0.08$, media $0.0158$), confermando che ogni traiettoria parte da uno stato fisico puro ben preparato.

---

## 5. ⚡ Ottimizzazione: Perché Abbiamo Creato la Cache `.npy`?

Leggere e convertire riga per riga il file `trajectories.csv` da 464 MB richiedeva tra i 20 e i 30 secondi a ogni esecuzione.

* Abbiamo isolato le 55 feature in una matrice 3D strutturata `(400 traiettorie, 1001 timestep, 55 feature)` e l'abbiamo salvata in formato binario nativo NumPy:
  $$\mathbf{data/cache/trajectories.npy} \quad (\approx 88.1\text{ MB})$$
* **Vantaggio pratico**: Nei notebook successivi (`02_windowing.ipynb`, `03_rnn.ipynb`, `04_transformer.ipynb`), il caricamento dei dati richiede ora **meno di 0.1 secondi** (istantaneo!), eliminando qualsiasi tempo morto.

---

## 6. ✂️ Partizionamento Anti-Leakage (Split per Traiettoria)

Nel machine learning su serie temporali è fondamentale evitare il **Data Leakage**: se dividessimo a caso le righe, la rete vedrebbe istanti adiacenti della stessa simulazione sia nel train che nel test, imparando a "memorizzare" la traiettoria invece di prevedere il futuro.

Abbiamo quindi diviso il dataset per **intere traiettorie indipendenti**:

| Set | Percentuale | N° Traiettorie | N° Punti Temporali | Scopo |
| :--- | :---: | :---: | :---: | :--- |
| **Training** | **70%** | **280** | 280.280 | Addestrare i pesi delle reti neurali |
| **Validation** | **15%** | **60** | 60.060 | Selezionare gli iperparametri e regolare l'early stopping |
| **Test** | **15%** | **60** | 60.060 | Valutazione finale imparziale di tutti i modelli |

* Gli indici sono stati estratti con `SEED = 42` e **congelati per sempre** nel file [`results/split_indices.npz`](file:///c:/Users/ernes/Desktop/Deep%20Learning/dl-quantum/results/split_indices.npz).
* Tutti i modelli della nostra pipeline (M1, M2, M3, M4) useranno esattamente questi stessi indici.

---

## 7. ⚖️ Decisione Chiave: Dati Grezzi (`none`) vs Standardizzazione (`zscore`)

Abbiamo eseguito un esperimento empirico rigoroso: addestrato una rete neurale baseline su **3 seed stocastici indipendenti (42, 43, 44)** sia con dati grezzi sia con standardizzazione per colonna, confrontando l'errore sulla scala originale:

| Strategia | Attivazione Uscita | Val MSE Medio | MSE Magnetizzazioni | MSE Correlazioni | Tempo Run |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Dati Grezzi (`none`)** | **`tanh`** | **0.000216** $\pm$ 0.00000 | **0.00011** | **0.00024** | ~16.6s |
| **Standardizzazione (`zscore`)** | Lineare | **0.000232** $\pm$ 0.00000 | 0.00013 | 0.00026 | ~17.1s |

### 💡 Perché abbiamo scelto di tenere i Dati Grezzi (`none` + `tanh`):
1. **Migliore accuratezza**: L'errore MSE dei dati grezzi è significativamente più basso ($0.000216$ vs $0.000232$), con intervalli di confidenza al 95% ($t$ di Student) completamente disgiunti.
2. **Stabilità a lungo termine con `tanh`**: Usando una funzione di attivazione $\tanh$ come strato finale, le previsioni della rete sono matematicamente costrette a rimanere dentro $[-1, 1]$. Quando la rete farà predizioni ricorsive a molti passi nel futuro (*free-running*), questo impedisce all'errore di esplodere verso valori assurdi ($> 1$ o $< -1$).
3. **Preservazione del rumore per i Transformer**: Se avessimo applicato lo Z-score, una correlazione debolissima tra atomi lontani (che oscilla appena tra $0.001$ e $0.005$) sarebbe stata forzata ad avere varianza 1, ingigantendone il rumore di fondo e falsando i prodotti scalari dei meccanismi di Self-Attention dei Transformer. Con i dati grezzi, le variabili mantengono la loro gerarchia naturale.

---

## 8. 📁 Dove si Trovano i File Generati da Questo Notebook

Tutti gli artefatti prodotti sono stati generati e salvati nelle rispettive cartelle del repository:

* **File Dati e Split**:
  * [`data/cache/trajectories.npy`](file:///c:/Users/ernes/Desktop/Deep%20Learning/dl-quantum/data/cache/trajectories.npy): cache per caricamento istantaneo;
  * [`results/split_indices.npz`](file:///c:/Users/ernes/Desktop/Deep%20Learning/dl-quantum/results/split_indices.npz): partizioni Train/Val/Test congelate;
  * [`results/normalization_comparison.csv`](file:///c:/Users/ernes/Desktop/Deep%20Learning/dl-quantum/results/normalization_comparison.csv): tabella numerica del test di normalizzazione.
* **Figure Esportate in [`figures/01_data/`](file:///c:/Users/ernes/Desktop/Deep%20Learning/dl-quantum/figures/01_data/)** (JPEG alta definizione 300 dpi):
  * `example_trajectories.jpeg`: andamento delle traiettorie campione (magnetizzazioni e correlazioni a confronto);
  * `feature_stats.jpeg`: intervallo min/max e medie delle 55 feature;
  * `value_distributions.jpeg`: istogrammi di densità ed evoluzione della varianza nel tempo;
  * `zoom_first_steps.jpeg`: dettaglio sui primi 100 timestep con indicazione delle finestre $T_{in}$ e $T_{out}$;
  * `normalization_loss_curves.jpeg`: curve di training e validation loss del confronto tra `none` e `zscore`.
