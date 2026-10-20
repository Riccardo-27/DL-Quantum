# 📊 Sintesi dei Risultati – Notebook 02: Finestre Temporali e Super-Risoluzione

Questo documento riassume le specifiche matematiche, le dimensioni dei tensori, i vincoli anti-leakage e le decisioni metodologiche stabilite nel secondo notebook (`notebooks/02_windowing.ipynb`).

---

## 1. 🪟 Finestrature Standard di Progetto (`short` e `long`)

Tutte le architetture (RNN dirette, RNN seq2seq e Transformer) vengono valutate su **due finestrature temporali standard** aventi lo stesso orizzonte futuro $T_{out} = 40$ ($\Delta t \times 40 = 0.80$ unità di tempo fisico) e lo stesso stride $\Delta s = 10$:

| Configurazione | $T_{in}$ (Passi / Tempo) | $T_{out}$ (Passi / Tempo) | Span Totale | Finestre Train (280 traj) | Finestre Val (60 traj) | Finestre Test (60 traj) | RAM Totale |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`short`** | **10** ($0.20$ s) | **40** ($0.80$ s) | 50 step | **26.880** | **5.760** | **5.760** | ~82 MB |
| **`long`** | **20** ($0.40$ s) | **40** ($0.80$ s) | 60 step | **26.600** | **5.700** | **5.700** | ~98 MB |

* **Formula del numero di finestre per traiettoria**:
  $$N_{win} = \left\lfloor \frac{1001 - (T_{in} + T_{out})}{10} \right\rfloor + 1$$
* **Perché $T_{out} = 40$ fisso**: mantenere costante l'orizzonte futuro isola l'effetto della lunghezza del contesto di memoria ($T_{in}$), rendendo la variazione delle prestazioni interamente attribuibile alla quantità di storia passata osservata.

---

## 2. 🎯 Sfida 1: Griglia $T_{in} \in \{5, 10, 20, 40\}$ e Allineamento dei Target

La **Sfida 1** richiede di determinare la minima finestra di osservazione passata necessaria per prevedere con accuratezza l'evoluzione quantistica.

### Il Problema del Disallineamento Temporale
Estraendo le finestre con partenza a $t=0$:
* $T_{in} = 5$ prevede il futuro a partire da $t = 0.10$.
* $T_{in} = 40$ prevede il futuro solo a partire da $t = 0.80$.
I modelli verrebbero quindi confrontati su target diversi lungo la dinamica transiente!

### La Nostra Soluzione: Allineamento con `align_t_in = 40`
Traslando le finestre con $T_{in} < 40$ di una quantità pari a $40 - T_{in}$, **tutte le finestre condividono esattamente gli stessi target futuri $Y$** a partire da $t = 0.80$.
* **Garanzia formale**: verifica di identità bit-a-bit `np.array_equal(Y_aln[tin], Y_aln[40])` superata al 100%.

| $T_{in}$ | Span Totale | Finestre Val Standard | Finestre Val Allineate | Primo $t$ Target Allineato |
| :---: | :---: | :---: | :---: | :---: |
| **5** | 45 step | 5.760 | **5.580** | $t = 0.80$ |
| **10** | 50 step | 5.760 | **5.580** | $t = 0.80$ |
| **20** | 60 step | 5.700 | **5.580** | $t = 0.80$ |
| **40** | 80 step | 5.580 | **5.580** | $t = 0.80$ |

---

## 3. 🔬 Sfida 2: Finestre per Super-Risoluzione Temporale ($\times 2$)

Nella **Sfida 2**, il modello riceve osservazioni rade con passo temporale raddoppiato ($2 \times \Delta t = 0.04$) per ricostruire la dinamica continua a frequenza nativa ($\Delta t = 0.02$).

* **Input ($X_{SR}$)**: $T_{in} = 20$ punti osservati a passo $\Delta t = 0.04$, che coprono uno span continuo di $(20 - 1) \times 2 + 1 = 39$ passi temporali.
* **Target ($Y_{SR}$)**: $T_{out} = 40$ passi consecutivi a risoluzione piena $\Delta t = 0.02$.
* **Shape generate**:
  * Train: $X_{SR} \in (26040, 20, 55)$, $Y_{SR} \in (26040, 40, 55)$.
  * Val: $X_{SR} \in (5580, 20, 55)$, $Y_{SR} \in (5580, 40, 55)$.
  * Test: $X_{SR} \in (5580, 20, 55)$, $Y_{SR} \in (5580, 40, 55)$.
* **Ground Truth Denso**: estratto simultaneamente ($Xd \in (26040, 39, 55)$) con test formale $X_{SR}[:, i] == Xd[:, 2i]$ superato.

---

## 4. 🔄 Input per Decoder e Teacher Forcing

Per i modelli sequenza-a-sequenza (RNN seq2seq e Transformer con train loop ibrido TF $\to$ MM $\to$ SS):
* La funzione `decoder_inputs(X, Y)` genera il tensore di input al decoder shiftato a destra:
  $$Y_{dec}[:, 0] = X[:, -1], \quad Y_{dec}[:, 1:] = Y[:, :-1]$$
* **Shape**: identica a $Y$ ($26880, 40, 55$).
* **Verifica**: identità numerica asserita sia su $t=0$ che su $t \ge 1$.

---

## 5. 🛡️ Garanzie Formali Anti-Leakage

Sono stati implementati e superati tutti i controlli di integrità:
1. **Confinamento intra-traiettoria**: nessuna finestra attraversa il confine tra due traiettorie indipendenti ($t_{start} + span \le 1001$).
2. **Disgiunzione assoluta tra partizioni**: Train (280), Val (60) e Test (60) non condividono alcuna traiettoria ($\text{Train} \cap \text{Val} \cap \text{Test} = \emptyset$).
3. **Controllo automatico**: `check_no_leakage(metas)` eseguito e passato con successo su tutte le finestrature (`short`, `long`, griglia $T_{in}$ e super-risoluzione).

---

## 6. ⚡ Efficienza I/O e Benchmark in Memoria

Grazie alla vettorizzazione con `numpy.lib.stride_tricks.sliding_window_view`:
* **Tempo medio di estrazione**: **~330 ms** per l'intero set di addestramento (280 traiettorie $\times$ 1001 punti $\to$ 26.880 finestre).
* **Throughput**: **~850 traiettorie/secondo** (> 80.000 finestre/secondo).
* **Decisione**: l'estrazione avviene in-memory al momento dell'addestramento, evitando di salvare decine di file ridondanti su disco (risparmio di oltre 500 MB).

---

## 7. 🖼️ Figure Ufficiali Generate (`figures/02_windowing/`)

Tutti i grafici sono salvati in formato standard JPEG a $\ge 300$ dpi:
1. **`window_sample_short.jpeg`**: esempio di finestra `short` ($T_{in}=10, T_{out}=40$) per magnetizzazione $m_1$ e correlazione $c_{1,2}$.
2. **`window_sample_long.jpeg`**: esempio di finestra `long` ($T_{in}=20, T_{out}=40$).
3. **`window_super_resolution.jpeg`**: visualizzazione intuitiva della super-risoluzione temporale (punti radi osservati vs dinamica continua vs target futuro).
4. **`tin_grid_comparison.jpeg`**: evidenza grafica dell'allineamento dei target tra $T_{in} \in \{5, 10, 20, 40\}$.
