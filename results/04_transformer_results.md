# 📊 Sintesi dei Risultati – Notebook 04: Modelli Transformer (M3 vs M4), Approccio "Start Small", Autoencoder e Sfide 1–2–3

Questo documento riassume in modo **semplice, intuitivo e dettagliato** tutti i risultati sperimentali, i confronti architetturali e le metriche numeriche ottenute dall'addestramento dei modelli basati su meccanismi di Attenzione (`notebooks/04_transformer.ipynb`), costruiti a partire dai blocchi modulari dei **Lab 6 (Autoencoder)** e **Lab 7 (Attention & Transformers)**.

---

## 1. 🎯 Qual era l'obiettivo del Notebook?
In questo notebook abbiamo sviluppato, addestrato e validato la seconda famiglia principale di architetture del progetto (**Modelli Transformer**), completando tutte le richieste sperimentali del corso (9 CFU):
1. **Approccio graduale ("Start Small") e Autoencoder (Lab 6)**: verifica preliminare su sottoinsieme ridotto (20 traiettorie) ed esplorazione della compressione dello spazio delle osservabili ($55 \to 16 \to 55$) prima dello scaling sull'intero dataset;
2. **M3 – Transformer Diretto (`TransformerDirect`, Encoder-Only One-Shot)**: elabora il contesto passato con blocchi di `GlobalSelfAttention` e proietta l'ultimo istante contestualizzato direttamente sull'intero orizzonte futuro ($T_{out} = 40$ passi) tramite testa densa con attivazione fisica $\tanh$;
3. **M4 – Transformer Seq2Seq Causale (`TransformerSeq2Seq`, Encoder-Decoder del Lab 7)**: combina `GlobalSelfAttention` nell'Encoder con `CausalSelfAttention` (maschera triangolare inferiore anti-leakage) e `CrossAttention` nel Decoder, generando la traiettoria futura passo dopo passo in *free-running*;
4. **Risoluzione completa delle 3 Sfide Obbligatorie**:
   * **Sfida 1**: ricerca della finestra minima ottima $T_{in} \in \{5, 10, 20, 40\}$ su target rigorosamente allineati (`align_t_in = 40`);
   * **Sfida 2**: Super-Risoluzione temporale ($\times 2$, da $\Delta t = 0.04\text{ s}$ rado a $\Delta t = 0.02\text{ s}$ denso);
   * **Sfida 3**: interpretabilità fisica tramite estrazione eager delle mappe di `CrossAttention` e `CausalSelfAttention`.

---

## 2. 🧪 Fase Preliminare: Approccio "Start Small" e Autoencoder (Lab 6)

Seguendo le indicazioni metodologiche del corso (*"Start Small"* e riduzione della dimensionalità con Autoencoder vista nel Lab 6), abbiamo condotto due esperimenti preliminari su un sottoinsieme ridotto di **20 traiettorie di training** (1.920 finestre) e **10 di validazione** (960 finestre) per 5 epoche ($2\text{ TF} + 2\text{ MM} + 1\text{ SS}$):

| Approccio Preliminare | Architettura | Parametri | Tempo Run | MSE Ricostruzione Pura | Val MSE Fisico Finale (55 var.) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Start Small Diretto** | `M3_Direct_Small` ($d=32, H=2, L=1$) | 87.128 | 5.0 s | — | **0.000752** *(da 0.001153)* |
| **Start Small Seq2Seq** | `M4_Seq2Seq_Small` ($d=32, H=2, L=1$) | 39.351 | 23.4 s | — | **0.001153** *(da 0.078785)* |
| **Autoencoder + Transformer Latente** | AE ($55 \to 16 \to 55$) + Transformer ($16\text{ dim}$) | **34.400** | 5.1 s + 5.0 s | **0.000362** | **0.000737** |

### 💡 Cosa ci ha insegnato questo esperimento?
* L'**Autoencoder** riduce di oltre il **60% i parametri** del Transformer (da $87.128$ a $34.400$) e converge velocemente, ma introduce un **pavimento incomprimibile di errore di ricostruzione** ($\text{MSE}_{\text{AE}} = 0.000362$) dovuto alla perdita di informazione sulle correlazioni quantistiche a lungo raggio di piccola ampiezza.
* Poiché sul dataset completo i nostri modelli nello spazio fisico a 55 variabili scendono fino a $\text{MSE} \approx 0.000121$ (quasi **3 volte inferiore** al solo errore di ricostruzione dell'Autoencoder!), lavorare direttamente sulle **55 osservabili fisiche native** con proiezione lineare continua (`PositionalEmbedding`) risulta nettamente superiore.

---

## 3. 🏆 Tabella Comparativa Completa dei Risultati (12 Modelli Transformer)

Abbiamo scalato l'addestramento sull'intero dataset (280 traiettorie di Train, 60 di Val) con il **train loop ibrido completo (10 ep. TF $\to$ 10 ep. MM $\to$ 5 ep. SS)** testando **3 configurazioni di iperparametri** su entrambe le finestre (`short` $T_{in}=10$ e `long` $T_{in}=20$):
* **HP1 (Baseline Equilibrata)**: $d_{\text{model}} = 64$, $\text{heads} = 4$, $L = 2$ layer, $d_{ff} = 128$, $\text{dropout} = 0.1$
* **HP2 (Alta Capacità)**: $d_{\text{model}} = 128$, $\text{heads} = 4$, $L = 2$ layer, $d_{ff} = 256$, $\text{dropout} = 0.1$
* **HP3 (Architettura Compatta e Veloce)**: $d_{\text{model}} = 64$, $\text{heads} = 2$, $L = 1$ layer, $d_{ff} = 128$, $\text{dropout} = 0.0$

| Famiglia | Modello | Configurazione HP | Finestra | $T_{in}$ | Parametri | Val MSE | Val MAE | Val RMSE | MSE Mag | MSE Corr | Tempo Training |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **M3** | `TransformerDirect` | HP1 ($d=64, H=4, L=2$) | `short` | 10 | 312.984 | 0.000181 | 0.01004 | 0.01344 | 0.000086 | 0.000202 | 122.5 s |
| **M3** | `TransformerDirect` | HP2 ($d=128, H=4, L=2$) | `short` | 10 | 951.448 | 0.000151 | 0.00894 | 0.01229 | 0.000057 | 0.000172 | 132.5 s |
| **M3** | **`TransformerDirect`** | **HP3 ($d=64, H=2, L=1$ - Vincitore)** | **`short`** | **10** | **196.632** | **0.000121** | **0.00783** | **0.01099** | **0.000036** | **0.000140** | **73.0 s** |
| **M4** | `TransformerSeq2Seq` | **HP1 ($d=64, H=4, L=2$ - Best M4)** | `short` | 10 | 476.535 | 0.000396 | 0.01558 | 0.01991 | 0.000279 | 0.000423 | 434.3 s |
| **M4** | `TransformerSeq2Seq` | HP2 ($d=128, H=4, L=2$) | `short` | 10 | 1.870.519 | 0.000437 | 0.01613 | 0.02091 | 0.000266 | 0.000475 | 696.6 s |
| **M4** | `TransformerSeq2Seq` | HP3 ($d=64, H=2, L=1$) | `short` | 10 | 144.183 | 0.000753 | 0.02154 | 0.02744 | 0.000738 | 0.000756 | 203.5 s |
| **M3** | `TransformerDirect` | HP1 ($d=64, H=4, L=2$) | `long` | 20 | 312.984 | 0.000169 | 0.00967 | 0.01301 | 0.000063 | 0.000193 | 131.2 s |
| **M3** | `TransformerDirect` | HP2 ($d=128, H=4, L=2$) | `long` | 20 | 951.448 | 0.000146 | 0.00872 | 0.01207 | 0.000042 | 0.000169 | 144.7 s |
| **M3** | **`TransformerDirect`** | **HP3 ($d=64, H=2, L=1$ - Vincitore)** | **`long`** | **20** | **196.632** | **0.000121** | **0.00786** | **0.01101** | **0.000038** | **0.000140** | **75.1 s** |
| **M4** | `TransformerSeq2Seq` | **HP1 ($d=64, H=4, L=2$ - Best M4)** | `long` | 20 | 476.535 | **0.000338** | 0.01439 | 0.01838 | 0.000225 | 0.000363 | 457.3 s |
| **M4** | `TransformerSeq2Seq` | HP2 ($d=128, H=4, L=2$) | `long` | 20 | 1.870.519 | 0.000406 | 0.01556 | 0.02016 | 0.000248 | 0.000442 | 750.3 s |
| **M4** | `TransformerSeq2Seq` | HP3 ($d=64, H=2, L=1$) | `long` | 20 | 144.183 | 0.000755 | 0.02160 | 0.02747 | 0.000736 | 0.000759 | 209.8 s |

---

## 4. 💡 Le 4 Conclusioni Fondamentali per il Report Tecnico

### 1. M3 (`TransformerDirect`) è il miglior modello assoluto dell'intero progetto
* Con **Val MSE = $0.000121$** (su `short` e `long`), **M3 (HP3)** riduce l'errore quadratico medio del **25%** rispetto alla migliore RNN **M1** ($0.000161$ su `short`, $0.000147$ su `long`) e di oltre l'**80%** rispetto alla baseline di persistenza ($0.000621$).
* Sulle sole **10 magnetizzazioni locali**, l'errore di M3 scende a un impressionante **$\text{MSE}_{\text{mag}} = 0.000036$** (contro $0.000075$ di M1, oltre il **50% di errore in meno**).

### 2. Perché l'architettura compatta HP3 vince su M3 ma perde su M4?
* In **M3 (One-Shot)**, un singolo layer di Self-Attention a 2 teste senza dropout (**HP3**, 196k parametri) evita la sovra-parametrizzazione e preserva la precisione numerica continua delle traiettorie hamiltoniane deterministiche, battendo sia HP1 che HP2 in metà tempo (~73 s).
* In **M4 (Seq2Seq Autoregressivo)**, invece, il decoding passo-passo su 40 istanti futuri richiede maggiore profondità e regolarizzazione per coordinare `CausalSelfAttention` e `CrossAttention`: qui la configurazione a 2 layer con dropout 0.1 (**HP1**) riduce l'errore di oltre la metà rispetto a HP3 ($0.000338$ vs $0.000755$).

### 3. Perché i modelli Diretti (M1, M3) superano sistematicamente i Seq2Seq (M2, M4) in Free-Running?
* Come già osservato per le RNN nel Notebook 03, durante la valutazione realistica a circuito chiuso (*free-running*), i modelli autoregressivi (M2 e M4) devono re-iniettare per 40 volte consecutive le proprie predizioni nel decoder.
* Nonostante lo *Scheduled Sampling* contenga la deriva, l'accumulo quadratico dei micro-errori di fase penalizza l'inferenza ricorsiva rispetto alla proiezione diretta one-shot (M1 e M3), che gode di visione simultanea sull'intero blocco futuro.

### 4. Rispetto rigoroso della causalità nel Decoder M4
* Il test di invarianza tra la predizione `free_running` e il passaggio parallelo auto-alimentato (`SelfFed_TF`) ha registrato uno scarto massimo di **$9.24 \times 10^{-7} < 10^{-5}$**, dimostrando matematicamente che la maschera triangolare inferiore di `CausalSelfAttention` blocca al 100% ogni fuga di informazione dal futuro.

---

## 5. 🎯 Risoluzione delle 3 Sfide Obbligatorie con i Transformer

### Sfida 1: Finestra Minima di Osservazione ($T_{in} \in \{5, 10, 20, 40\}$, Target Allineati)
Abbiamo valutato il modello vincitore **M3 (HP3)** sulla griglia allineata (`align_t_in = 40`, 5.580 finestre che predicono tutte l'intervallo a partire da $t = 0.80\text{ s}$):

| Contesto $T_{in}$ (Passi / Tempo) | Validation MSE | Validation MAE | Test MSE | Test MAE | MSE Magnetizzazioni | MSE Correlazioni | Tempo Run |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **5** ($0.10$ s) | **0.000122** | 0.00792 | **0.000123** | 0.00797 | 0.000044 | 0.000139 | 73.4 s |
| **10** ($0.20$ s) | **0.000122** | 0.00786 | **0.000122** | 0.00789 | 0.000037 | 0.000140 | 73.1 s |
| **20** ($0.40$ s) | **0.000120** | 0.00784 | **0.000121** | 0.00786 | 0.000038 | 0.000139 | 74.2 s |
| **40** ($0.80$ s) | **0.000111** | **0.00758** | **0.000111** | **0.00757** | **0.000036** | **0.000127** | 79.0 s |

* **Confronto fisico con l'RNN (M1)**: mentre l'LSTM saturava a $T_{in}=20$ ($0.000147 \to 0.000148$ a $T_{in}=40$) per via della compressione nello stato nascosto, il **Transformer M3** non solo ottiene già a **$T_{in}=5$** ($0.000122$) un risultato migliore del miglior LSTM a $T_{in}=20$ ($0.000147$), ma a **$T_{in}=40$** scende ulteriormente a **$0.000111$** grazie all'accesso diretto e non attenuato della `GlobalSelfAttention` a tutti i 40 istanti passati.

---

### Sfida 2: Super-Risoluzione Temporale (Subsampling $\times 2$, $\Delta t = 0.04 \to 0.02$)
Abbiamo confrontato il modello **M3 (HP3)** alimentato con **20 punti radi sottocampionati** ($\Delta t = 0.04\text{ s}$) rispetto allo stesso modello alimentato con tutti i **39 punti densi nativi** ($\Delta t = 0.02\text{ s}$) sullo stesso intervallo temporale:

| Configurazione Ingresso | Passo Input ($\Delta t_{in}$) | Passo Output ($\Delta t_{out}$) | Val MSE | Val MAE | Test MSE | Test MAE | Tempo Run |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Super-Risoluzione ($\times 2$, 20 pt radi)** | **$0.04\text{ s}$** | **$0.02\text{ s}$** | **0.000121** | **0.00784** | **0.000121** | **0.00788** | 73.9 s |
| **Risoluzione Nativa Densa (39 pt densi)** | $0.02\text{ s}$ | $0.02\text{ s}$ | **0.000109** | **0.00753** | **0.000110** | **0.00754** | 79.5 s |

* **Conclusione Sfida 2**: Dimezzando la frequenza di campionamento dei sensori quantistici in ingresso, la perdita di accuratezza è minima ($\Delta\text{MSE} \approx 1.1 \times 10^{-5}$): il Transformer ricostruisce fedelmente le oscillazioni ad alta frequenza ($\Delta t = 0.02\text{ s}$) agendo da interpolatore dinamico non lineare.

---

### Sfida 3: Interpretabilità Fisica tramite Mappe di Attenzione (Lab 7)
Eseguendo il passaggio forward in modalità eager fuori da `@tf.function` (metodo `get_attention_scores`, analogo al `Translator` del Lab 7) su 32 finestre del Test Set con **M4 (`TransformerSeq2Seq`, HP1, 4 teste)**, abbiamo estratto e verificato:
1. **`CausalSelfAttention` $(32, 4, 40, 40)$**: somma sulle righe pari a $1.0$ (scarto max $1.79 \times 10^{-7}$) e parte strettamente triangolare superiore **identicamente nulla** ($\max |\text{triu}(k=1)| = 0.00$), confermando l'assenza totale di *future leakage*.
2. **`CrossAttention` $(32, 4, 40, 10)$**: le 4 teste mostrano una chiara **specializzazione fisica**:
   * **Ancoraggio di fase locale**: forte concentrazione dei pesi sugli ultimi istanti osservati ($t \in [-0.04, 0.00]\text{ s}$), indispensabile per agganciare con continuità posizione e derivata temporale delle osservabili al confine passato/futuro;
   * **Memoria periodica (*Quantum Scars*)**: bande secondarie di attenzione distribuite sugli step iniziali e intermedi della finestra passata, che catturano la frequenza caratteristica delle oscillazioni di Rabi a molti corpi e il vincolo del blocco di Rydberg.

---

## 6. 🔬 Analisi Disaggregata e Integrità Fisica delle Predizioni

Sulle predizioni del modello vincitore (`M3_Direct_HP3_short`):
* **Positività della matrice di densità a 2 corpi ($P(s_i, s_j) \ge 0$)**:
  * Minima probabilità congiunta predetta: **$0.1548 > 0$** (in perfetto accordo con il minimo reale del dataset $0.1553$);
  * Tasso di violazioni fisiche: **$0.000000\%$ (zero violazioni)**.
* **Magnetizzazioni vs Correlazioni**:
  * $\text{MSE}_{\text{mag}} = 0.000036$ | $\text{MAE}_{\text{mag}} = 0.00452$;
  * $\text{MSE}_{\text{corr}} = 0.000140$ | $\text{MAE}_{\text{corr}} = 0.00856$.

---

## 7. 📐 Validazione Statistica su 3 Seed Indipendenti ($t$ di Student)

La configurazione migliore (`M3_Direct`, HP3, finestra `short`) è stata replicata sui 3 seed indipendenti `[42, 43, 44]`:
* **Seed 42**: $\text{Val MSE} = 0.0001207$ | $\text{Val MAE} = 0.007827$
* **Seed 43**: $\text{Val MSE} = 0.0001211$ | $\text{Val MAE} = 0.007872$
* **Seed 44**: $\text{Val MSE} = 0.0001214$ | $\text{Val MAE} = 0.007909$

**Stima finale con Intervallo di Confidenza al 95% ($t$ di Student a 2 g.d.l., $t_{0.975, 2} \approx 4.303$):**
$$\mathbf{Validation\ MSE} = \mathbf{0.000121 \pm 0.000001}$$
$$\mathbf{Validation\ MAE} = \mathbf{0.007869 \pm 0.000102}$$

---

## 8. 📁 Artefatti, Modelli e Figure Esportate

* **Checkpoint del Modello Vincitore**:
  * `checkpoints/best_transformer_model.keras` (salvato e verificato con ricaricamento tramite `CUSTOM_OBJECTS`).
* **Log Sperimentale Completo**:
  * [`results/results_log_riccardo.csv`](file:///c:/Users/ernes/Desktop/Deep%20Learning/dl-quantum/results/results_log_riccardo.csv): 18 run Transformer registrati (12 griglia HP + 4 Sfida 1 + 2 Sfida 2).
* **9 Figure ad Alta Risoluzione ($\ge 300\text{ DPI}$) in [`figures/04_transformer/`](file:///c:/Users/ernes/Desktop/Deep%20Learning/dl-quantum/figures/04_transformer/)**:
  1. `start_small_and_autoencoder.jpeg`: confronto rapido *Start Small* e ricostruzione con Autoencoder (Lab 6);
  2. `loss_curves_hybrid.jpeg`: curve di apprendimento M3 e M4 lungo le 3 fasi ibride (`TF -> MM -> SS`);
  3. `free_running_error_accumulation.jpeg`: accumulo dell'errore sull'orizzonte $t \in [1, 40]$ (M4 Teacher Forcing vs M4 Free-Running vs M3 One-Shot);
  4. `challenge1_tin_scaling.jpeg`: scaling dell'errore di validazione e test in funzione di $T_{in} \in \{5, 10, 20, 40\}$;
  5. `challenge2_super_resolution.jpeg`: ricostruzione ad alta frequenza ($\Delta t = 0.02\text{ s}$) da input sottocampionato ($\Delta t = 0.04\text{ s}$);
  6. `challenge3_cross_attention_heads.jpeg`: mappe di calore multi-head della `CrossAttention` ($T_{out} \times T_{in}$);
  7. `challenge3_causal_mask_verification.jpeg`: verifica visiva della maschera causale triangolare inferiore del Decoder vs Encoder bidirezionale;
  8. `challenge3_attention_mean_profile.jpeg`: profilo temporale medio dell'attenzione incrociata correlato alle oscillazioni fisiche;
  9. `disaggregated_error_analysis.jpeg`: errore disaggregato per orizzonte, singolo qubit e matrice $10 \times 10$ delle coppie di correlazione.

