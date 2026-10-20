# 📊 Sintesi dei Risultati – Notebook 02: Finestrature Temporali, Allineamento e Super-Risoluzione

Questo documento riassume in modo **semplice, intuitivo e dettagliato** tutto ciò che è stato verificato e generato nel secondo notebook (`notebooks/02_windowing.ipynb`), analizzando i valori reali dell'esecuzione.

---

## 1. 🎯 Qual era l'obiettivo del Notebook?
I modelli sequenziali (come LSTM, GRU e Transformer) non possono essere addestrati direttamente su intere traiettorie lunghe 1001 passi: devono ricevere **coppie (Input $X$, Target $Y$)** estratte facendo scorrere una finestra temporale lungo la traiettoria.

In questo notebook abbiamo:
1. Definito le **due finestrature standard di benchmark (`short` e `long`)** usate da tutti i modelli del progetto;
2. Risolto il problema del **disallineamento dei target per la Sfida 1** (ricerca del $T_{in}$ minimo);
3. Generato i dati sottocampionati per la **Super-Risoluzione Temporale (Sfida 2)**;
4. Creato e validato l'input al decoder ($Y_{dec}$) per il **Teacher Forcing** (usato da Seq2Seq e Transformer);
5. Certificato l'**assoluta assenza di Data Leakage** tra traiettorie e tra partizioni Train/Val/Test.

---

## 2. 🪟 Finestrature Standard di Progetto (`short` e `long`)

Tutti i modelli del progetto condividono le stesse due finestrature con **orizzonte di predizione fisso a $T_{out} = 40$ passi** ($0.80$ unità fisiche) e **passo di scorrimento (stride) $\Delta s = 10$ passi** ($0.20$ unità fisiche):

| Configurazione | $T_{in}$ (Passato) | $T_{out}$ (Futuro) | Finestre Train (280 traj) | Finestre Val (60 traj) | Finestre Test (60 traj) | RAM Totale (X + Y) | Tempo Estrazione |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`short`** | **10** ($0.20$ s) | **40** ($0.80$ s) | **26.880** | **5.760** | **5.760** | **402.8 MB** | **0.319 s** |
| **`long`** | **20** ($0.40$ s) | **40** ($0.80$ s) | **26.600** | **5.700** | **5.700** | **478.4 MB** | **0.324 s** |

### 💡 Perché $T_{out} = 40$ è fisso per entrambe le configurazioni?
Mantenendo invariata la difficoltà della previsione futura ($40$ passi temporali in avanti), qualsiasi differenza di accuratezza tra `short` e `long` è attribuibile **esclusivamente alla quantità di storia passata osservata** ($10$ vs $20$ passi), consentendo un confronto imparziale.

---

## 3. 🎯 Sfida 1: Griglia $T_{in} \in \{5, 10, 20, 40\}$ e Allineamento Target

La **Sfida 1** richiede di scoprire *qual è la finestra di osservazione minima necessaria per fare previsioni affidabili*.
Abbiamo quindi analizzato una griglia di lunghezze di contesto: $T_{in} \in \{5, 10, 20, 40\}$.

### Il Problema del Disallineamento Temporale
Se estraessimo le finestre facendole partire tutte all'inizio della traiettoria ($t=0$):
* Con $T_{in} = 5$, la rete inizierebbe a prevedere già da $t = 0.10$;
* Con $T_{in} = 40$, la rete inizierebbe a prevedere solo da $t = 0.80$.
I modelli verrebbero quindi valutati su punti temporali diversi, falsando il confronto!

### La Soluzione: Allineamento con `align_t_in = 40`
Traslando l'inizio delle finestre più corte di una quantità pari a $40 - T_{in}$:
* **Tutti i modelli valutano esattamente gli stessi 5.580 target futuri $Y$** a partire da $t = 0.80$.
* L'identità numerica bit-a-bit `Y_aln[tin] == Y_aln[40]` è stata **superata al 100% per tutti i $T_{in}$**.

| $T_{in}$ (Passi / Tempo) | Span Totale | Finestre Val Standard | Finestre Val Allineate | Primo $t$ Target Valutato |
| :---: | :---: | :---: | :---: | :---: |
| **5** ($0.10$ s) | 45 step | 5.760 | **5.580** | **$t = 0.80$** |
| **10** ($0.20$ s) | 50 step | 5.760 | **5.580** | **$t = 0.80$** |
| **20** ($0.40$ s) | 60 step | 5.700 | **5.580** | **$t = 0.80$** |
| **40** ($0.80$ s) | 80 step | 5.580 | **5.580** | **$t = 0.80$** |

---

## 4. 🔬 Sfida 2: Super-Risoluzione Temporale (Subsampling $\times 2$)

Nella **Sfida 2**, chiediamo al modello di prevedere il futuro ad alta risoluzione nativa ($\Delta t = 0.02$) su $T_{out} = 40$ ricevendo in ingresso dati **sottocampionati nel tempo** (frequenza dimezzata, $\Delta t = 0.04$):

* **Input Rado ($X_{SR}$)**: 20 punti campionati ogni 2 step ($\Delta t = 0.04$), coprendo uno span temporale di 39 passi.
* **Target Pieno ($Y$)**: 40 passi consecutivi a risoluzione piena originale ($\Delta t = 0.02$).
* **Dimensioni tensoriali verificate**:
  * Train: $X_{SR} \in (26.040, 20, 55)$, $Y \in (26.040, 40, 55)$
  * Validation: $X_{SR} \in (5.580, 20, 55)$, $Y \in (5.580, 40, 55)$
  * Test: $X_{SR} \in (5.580, 20, 55)$, $Y \in (5.580, 40, 55)$
* Abbiamo conservato anche il **contesto denso nativo ($X_{dense} \in (26.040, 39, 55)$)** per confrontare visivamente il segnale continuo rispetto ai punti radi osservati.

---

## 5. 🔄 Input per il Decoder e Teacher Forcing ($Y_{dec}$)

I modelli Sequenza-a-Sequenza (come LSTM encoder-decoder e Transformer) generano la sequenza passo dopo passo.
Durante l'addestramento con **Teacher Forcing**, al decoder viene fornito il target reale traslato a destra di un passo:
* Al passo $t=0$, riceve l'ultimo valore visto dall'encoder: $Y_{dec}[:, 0] = X[:, -1]$;
* Ai passi successivi $t \ge 1$, riceve il target reale al passo precedente: $Y_{dec}[:, 1:] = Y[:, :-1]$.

Abbiamo testato la funzione `decoder_inputs` verificando l'esatta concordanza formale su tensori di prova: **verifica superata al 100%**.

---

## 6. 🛡️ Garanzie Formali di Assenza di Data Leakage

Sono stati eseguiti e superati tre controlli rigorosi:
1. **Confinamento intra-traiettoria**: nessuna finestra attraversa mai il confine tra due traiettorie diverse (ogni finestra si chiude entro il punto 1001 della propria simulazione).
2. **Disgiunzione assoluta tra partizioni**: Train (280), Val (60) e Test (60) sono insiemi disgiunti al 100% ($\text{Train} \cap \text{Val} \cap \text{Test} = \emptyset$).
3. **Controllo con `check_no_leakage`**: funzione di progetto eseguita con successo su tutte le configurazioni (`short`, `long`, griglia $T_{in}$ e super-risoluzione).

---

## 7. ⚡ Benchmark delle Prestazioni: Generazione al Volo in RAM

Abbiamo misurato le prestazioni della funzione vettorizzata `sliding_window_view`:
* **Tempo medio di estrazione**: **320.6 $\pm$ 82.1 ms** per estrarre tutte le 26.880 finestre di addestramento da 280 traiettorie.
* **Throughput**: **873.4 traiettorie/secondo** (oltre **83.850 finestre/secondo**).
* **Decisione strategica**: Poiché l'estrazione richiede solo un terzo di secondo, le finestre vengono generate direttamente in memoria RAM all'avvio dei notebook dei modelli. Questo evita di salvare su disco centinaia di megabyte ridondanti per ciascuna combinazione.

---

## 8. 🖼️ Figure Ufficiali Esportate (`figures/02_windowing/`)

Tutti i grafici sono stati generati ed esportati in formato JPEG ad alta risoluzione ($\ge 300$ dpi):

1. **`window_sample_short.jpeg`**: visualizzazione di una finestra `short` ($T_{in}=10 \to T_{out}=40$) per la magnetizzazione $m_1$ e la correlazione $c_{1,2}$.
2. **`window_sample_long.jpeg`**: visualizzazione della corrispondente finestra `long` ($T_{in}=20 \to T_{out}=40$).
3. **`window_super_resolution.jpeg`**: rappresentazione visiva della Super-Risoluzione (punti radi osservati ogni 2 step vs dinamica continua reale vs target futuro).
4. **`tin_grid_comparison.jpeg`**: evidenza visiva dell'allineamento dei target per la Sfida 1, mostrando come $T_{in} \in \{5, 10, 20, 40\}$ valutino esattamente la stessa porzione futura.
