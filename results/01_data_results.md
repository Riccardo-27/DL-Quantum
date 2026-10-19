# 📊 Sintesi dei Risultati – Notebook 01: Analisi Dati e Preprocessing

Questo documento riassume in modo semplice, chiaro e intuitivo tutti i risultati ottenuti dall'esecuzione del primo notebook (`notebooks/01_data.ipynb`).

---

## 1. 🗂️ Carta d'Identità del Dataset

Il dataset descrive l'evoluzione quantistica di una catena di **10 atomi di Rydberg (qubit)** governati dal modello teorico PXP.

* **File originario**: `trajectories.csv` (~464 MB)
* **Traiettorie simulate**: **400** dinamiche indipendenti
* **Punti temporali per traiettoria**: **1001** istanti, da $t = 0.0$ a $t = 20.0$ con passo uniforme $\Delta t = 0.02$
* **Totale righe analizzate**: $400 \times 1001 = \mathbf{400.400}$
* **Integrità numerica**: **0 valori nulli (NaN)** e **0 infiniti (Inf)**. Tutti i valori sono perfettamente continui e ben definiti.

---

## 2. ⚛️ Le 55 Variabili Fisiche e i Vincoli Quantistici

A ogni istante di tempo misuriamo **55 osservabili fisiche quantistiche** (valori di aspettazione degli operatori di Pauli $\sigma_z$):

1. **10 Magnetizzazioni locali** $m_i(t) = \langle \sigma_z^i(t) \rangle \in [-1, 1]$ (la polarizzazione di ciascun qubit).
2. **45 Correlazioni a due corpi** $c_{ij}(t) = \langle \sigma_z^i(t) \sigma_z^j(t) \rangle \in [-1, 1]$ (l'interazione tra tutte le possibili coppie di atomi $i < j$).

### Verifiche di Coerenza Quantistica Effettuate
* **Bound scalare $[-1, 1]$**: Nessun valore esce dai limiti fisici imposti dalla meccanica quantistica.
* **Probabilità congiunte a due spin positive**: Per ogni coppia di qubit e per ogni configurazione di spin $(s_i, s_j \in \{-1, +1\})$, la probabilità congiunta quantistica vale:
  $$P(s_i, s_j) = \frac{1}{4} \left(1 + s_i m_i + s_j m_j + s_i s_j c_{ij}\right)$$
  * **Minimo assoluto riscontrato**: $\mathbf{0.1553 > 0}$
  * **Violazioni fisiche ($P < 0$)**: **Esattamente 0** su oltre **72 milioni di verifiche** ($400 \text{ traiettorie} \times 1001 \text{ timestep} \times 45 \text{ coppie} \times 4 \text{ stati}$).
  * *Significato intuitivo*: la matrice di densità ridotta a due corpi è matematicamente valida e semidefinita positiva lungo tutta la simulazione.
* **Stati iniziali disaccoppiati a $t=0$**: La differenza $|c_{ij}(0) - m_i(0) m_j(0)|$ ha una media di soli **0.0158** (massimo 0.0800), confermando che i 400 stati iniziali sono stati preparati come stati quasi di prodotto (product states).

---

## 3. ✂️ Partizionamento del Dataset (Split per Traiettoria)

Per addestrare e valutare i modelli in modo rigoroso ed evitare **data leakage** (contaminazione temporale tra passato e futuro), abbiamo suddiviso i dati **per traiettoria completa**, non per singola riga:

| Partizione | Percentuale | N° Traiettorie | N° Timestep Totali | Ruolo |
| :--- | :---: | :---: | :---: | :--- |
| **Train** | **70%** | **280** | 280.280 | Addestramento dei parametri dei modelli |
| **Validation** | **15%** | **60** | 60.060 | Scelta iperparametri e early stopping |
| **Test** | **15%** | **60** | 60.060 | Valutazione finale imparziale (benchmark) |

* **Garanzia di isolamento**: Le tre partizioni sono disgiunte al 100% ($\text{Train} \cap \text{Val} \cap \text{Test} = \emptyset$).
* **File congelato**: Gli indici sono stati salvati definitivamente in `results/split_indices.npz` (con seed 42) e non cambieranno mai per tutto il resto del progetto.

---

## 4. ⚖️ Scelta della Normalizzazione: Dati Grezzi vs Z-Score

Ci siamo posti una domanda fondamentale: *conviene standardizzare i dati con media 0 e varianza 1 (z-score) oppure lasciarli grezzi?*

Per rispondere con certezza scientifica, abbiamo addestrato una rete neurale baseline su **3 seed indipendenti (42, 43, 44)** confrontando le due strategie:

| Strategia | Attivazione Output | Val MSE Totale | MSE Magnetizzazioni | MSE Correlazioni | Tempo Run |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Dati Grezzi (`none`)** | **`tanh`** | **0.00022** $\pm$ 0.00000 | **0.00011** | **0.00024** | ~17.5s |
| **Z-Score (`zscore`)** | Lineare | **0.00023** $\pm$ 0.00000 | 0.00013 | 0.00026 | ~16.2s |

### Perché abbiamo scelto Dati Grezzi (`none`) con `tanh`:
1. **Risultato empirico superiore**: L'errore MSE è significativamente più basso, con intervalli di confidenza al 95% disgiunti.
2. **Protezione fisica con `tanh`**: La funzione tangente iperbolica ($\tanh$) vincola matematicamente le predizioni all'intervallo $(-1, 1)$. Nelle predizioni a lungo termine (dove il modello alimenta le proprie uscite come nuovi ingressi), questo impedisce alla rete di esplodere verso valori fisicamente impossibili ($> 1$ o $< -1$).
3. **Preservazione delle proporzioni per i Transformer**: Molte correlazioni tra qubit lontani hanno oscillazioni piccolissime (es. tra 0.01 e 0.05). Uno z-score le avrebbe gonfiate artificialmente, amplificando il rumore di fondo e sballando i prodotti scalari dei meccanismi di Self-Attention. I dati grezzi mantengono intatta la gerarchia fisica tra le variabili.

---

## 5. 📦 Artefatti Prodotti e Disponibili

Tutti i risultati e gli artefatti prodotti da questo primo notebook sono pronti e disponibili per i notebook successivi:

1. **`data/cache/trajectories.npy`** (88.1 MB): Tensore 3D `(400, 1001, 55)` in formato binario float32. Riduce i tempi di caricamento da 15-20 secondi a soli **0.2 secondi**.
2. **`results/split_indices.npz`**: Contiene gli array con gli indici di traiettoria congelati per Train, Val e Test.
3. **Cartella `figures/01_data/`**: 5 figure esportate ad alta definizione (300 dpi) pronte per report e slide:
   * `example_trajectories.jpeg`: andamento nel tempo delle magnetizzazioni e delle correlazioni.
   * `feature_stats.jpeg`: medie e deviazioni standard delle 55 osservabili.
   * `value_distributions.jpeg`: distribuzioni di densità dei valori osservati.
   * `zoom_first_steps.jpeg`: zoom sulla dinamica quantistica iniziale.
   * `normalization_loss_curves.jpeg`: curve di apprendimento che dimostrano la superiorità di `none` vs `zscore`.
