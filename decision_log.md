# Decision Log — DL-Quantum

Traccia delle decisioni progettuali, metodologiche e architetturali adottate nello sviluppo del progetto e, soprattutto, del **limite teorico o computazionale intrinseco di ogni modello e di cosa lo risolve nel successivo**.

---

## Fondamenta (01-02)

- **Split per traiettoria (280/60/60), seed 42**:
  - *Decisione*: Il dataset di 400 traiettorie quantistiche indipendenti viene suddiviso in 280 traiettorie di training (70%), 60 di validation (15%) e 60 di test (15%), fissando il seed a 42.
  - *Motivazione*: Evita il data leakage temporale sistematico che si avrebbe splittando per riga (righe consecutive appartengono alla medesima traiettoria fisica hamiltoniana). Il modello viene così addestrato e valutato su evoluzioni dinamiche generate da condizioni iniziali e regimi quantistici completamente disgiunti.

- **Nessuna normalizzazione, deliberatamente (01 verifica solo il bound fisico `[-1, 1]`)**:
  - *Decisione*: Nessun riscalamento convenzionale (né z-score né min-max per-feature sui dati osservati in train).
  - *Motivazione fisica e statistica*: Le 55 osservabili quantistiche (10 magnetizzazioni locali $\langle \sigma_z^i \rangle$ e 45 correlazioni a due corpi $\langle \sigma_z^i \sigma_z^j \rangle$) sono valori di aspettazione di operatori di spin di Pauli, rigorosamente confinati a livello quantomeccanico nell'intervallo compatto $[-1, 1]$ — lo STESSO bound assoluto per tutte le feature, indipendente dal dataset campionato.
  - *Rigetto dello z-score e del min-max per-feature*: Entrambi scartati per lo stesso motivo: alcune feature (specialmente correlazioni a lungo raggio tra qubit lontani, es. oscillanti solo tra $0.01$ e $0.05$) verrebbero riscalate per riempire artificialmente l'intero intervallo $[-1, 1]$ o $[0, 1]$. Ciò amplificherebbe in modo patologico il rumore di fondo della simulazione, distruggendo sia il significato fisico assoluto dello zero (assenza di polarizzazione o correlazione) sia le proporzioni relative tra le osservabili.
  - *Protezione dei prodotti scalari nei Transformer (M3/M4)*: I meccanismi di attenzione multi-head calcolano prodotti scalari tra rappresentazioni proiettate ($\mathbf{Q}\mathbf{K}^T$). Mantenere le ampiezze fisiche naturali impedisce che correlazioni fisicamente trascurabili dominino indebitamente i pesi di attenzione rispetto alle oscillazioni dominanti delle magnetizzazioni locali.
  - *Attivazione finale `tanh` per tutti i modelli*: Tutti i layer finali `Dense` di M1-M4 impiegano attivazione `tanh` invece di `linear`. Poiché i target fisici sono confinati in $[-1, 1]$, la `tanh` vincola matematicamente le predizioni all'intervallo fisico ammissibile, prevenendo derive numeriche e divergenze esplosive durante la valutazione ricorsiva free-running (M2/M4).

- **Integrità quantistica e disaccoppiamento a $t=0$**:
  - *Verifica*: Controllo formale della positività semidefinita delle matrici di densità a 2 corpi:
    $$P(s_i, s_j) = \frac{1}{4}\left(1 + s_i \langle \sigma_z^i \rangle + s_j \langle \sigma_z^j \rangle + s_i s_j \langle \sigma_z^i \sigma_z^j \rangle\right) \ge 0, \quad \forall s_i, s_j \in \{-1, +1\}$$
    Zero violazioni rilevate sull'intero dataset (valore minimo $0.1553 > 0$).
  - A $t=0$, lo scarto medio $|c_{ij} - m_i m_j| = 0.0158$ conferma stati iniziali di prodotto (product states) fisicamente coerenti con l'evoluzione del reticolo PXP.

- **Griglia comune di finestre (`short`: $T_{\text{in}}=10$, `long`: $T_{\text{in}}=20$, $T_{\text{out}}=40$, $\text{stride}=10$ per entrambe)**:
  - *Decisione*: Utilizzata in modo identico e controllato da M1 a M4 per rendere rigorosamente comparabile la tabella finale di benchmark (`Table 1`).
  - *Motivazione*: Mantenere costante l'orizzonte futuro target $T_{\text{out}}=40$ ($0.80$ unità fisiche di tempo adimensionale $\tau$) isola la memoria storica passata $T_{\text{in}}$ come unica variabile indipendente, consentendo di quantificare l'impatto della profondità temporale senza alterare la difficoltà intrinseca del task di predizione. Risponde direttamente alla Sfida 1 del progetto.

- **`output_window` fissato a 40 step ($0.80$ unità fisiche di tempo $\tau$)**:
  - Con campionamento nativo $\Delta\tau = 0.02$, un orizzonte ridotto a 5 step ($\Delta\tau_{\text{out}} = 0.10$) sarebbe stato poco significativo rispetto al tempo totale di simulazione ($\tau_{\text{fin}} = 20.0$) e rispetto alla scala temporale delle oscillazioni di Rydberg scar ($\tau_{\text{scar}} \approx 1.33$).
  - L'orizzonte non è stato esteso a 80 step per contenere il costo computazionale su GPU Colab: la valutazione autoregressiva in `free_running_predict` (M2/M4) ricalcola l'evoluzione passo per passo, con costo che cresce quadraticamente rispetto all'orizzonte. 40 step ($0.80$ unità fisiche $\tau$) rappresenta il trade-off ideale tra profondità dinamica (copre oltre il 60% dell'intero periodo di scar) e fattibilità computazionale dell'intera pipeline (4 modelli × 2 finestre × 3 trial tuning + repliche multi-seed).

- **`stride` scollegato da `output_window` (`stride=10` fisso)**:
  - Con $T_{\text{out}}=40$, impostare $\text{stride}=T_{\text{out}}$ (finestre strettamente non sovrapposte) avrebbe fatto crollare il numero di campioni per traiettoria da ~195 a ~11 su `long`, rendendo insufficiente il dataset per l'hyperparameter tuning.
  - Al contrario, uno stride pari a 5 avrebbe generato finestre sovrapposte all'87.5% - 93.75%, quadruplicando i tempi di training senza apportare nuova informazione fisica.
  - La scelta di $\text{stride}=10$ è il punto di equilibrio ottimale: garantisce ~95-96 finestre estratte per traiettoria per entrambe le configurazioni, con variabilità sufficiente e campionamento uniforme dello spazio delle fasi.

- **Epoche di training finale post-tuning a 40 con budget bilanciato**:
  - Per M1 (modello diretto), il training post-tuning opera su 40 epoche fisse con early stopping e restore dei migliori pesi di validazione.
  - Per i modelli con decoder sequenziale (M2/M4), le 40 epoche totali sono suddivise secondo il protocollo progressivo: 18 epoche di Teacher Forcing (TF), 11 epoche di Masked Modeling (MM) e 11 epoche di Scheduled Sampling (SS), garantendo che la fase di stabilizzazione iniziale dei gradienti sia la più solida.
  - `EPOCHS_TUNING` (5-10 epoche) resta deliberatamente snello per esplorare ampiamente lo spazio degli iperparametri su Keras Tuner.

- **Training loop ibrido progressivo a 3 fasi (TF $\to$ MM $\to$ SS)**:
  - *Problema affrontato*: L'**Exposure Bias** (o *Covariate Shift*). Durante l'addestramento tradizionale assistito (*Teacher Forcing*), il decoder riceve sempre i target esatti ground-truth; in fase di inferenza libera (*Free-Running*), il decoder riceve le proprie predizioni rumorose precedenti, accumulando rapidamente errori lungo i 40 timestep futuri.
  - *Curriculum progressivo*:
    1. **Fase 1 (Teacher Forcing - TF)**: Ingressi esatti shiftati al decoder per garantire convergenza rapida e stabilizzazione dei gradienti.
    2. **Fase 2 (Masked Modeling - MM)**: Mascheramento casuale di interi timestep storici (azzeramento del vettore spaziale a 55 dimensioni), preservando l'ultimo istante temporale, per forzare l'apprendimento di correlazioni temporali a lungo raggio.
    3. **Fase 3 (Scheduled Sampling - SS)**: Ingressi al decoder formati stocasticamente da una miscela tra target reale e predizione precedente della rete con probabilità $p_{\text{ss}}$ crescente fino a $0.50$, abituando il modello a correggere attivamente gli errori di traiettoria.

- **Sonde di validazione free-running separate (`fr_phase`, `fr_target`, `fr_curve`)**:
  - La `val_loss` computata con Teacher Forcing non riflette la reale accuratezza del modello in generazione autonoma. Per M2 ed M4 vengono implementate sonde dedicate in inferenza ricorsiva pura (*free-running*), loggando separatamente `val_free_running_loss` e `test_free_running_loss`.

- **Hyperparameter Tuning con Keras Tuner**:
  - Adozione di `keras_tuner.RandomSearch` come protocollo di ottimizzazione degli iperparametri nativo del framework Keras/TensorFlow, garantendo un'esplorazione trasparente, controllata e riproducibile dello spazio delle configurazioni senza ricorso a librerie terze non standard.
  - Il tuning viene condotto modello per modello subito dopo l'implementazione architetturale, isolando spazi di ricerca dedicati per le RNN (tipologia cella LSTM/GRU, unità nascoste $\{64, 128\}$, learning rate $\{10^{-3}, 5\times 10^{-4}\}$) e per i Transformer ($d_{\text{model}} \in \{64, 128\}$, teste di attenzione $\{2, 4\}$, strati $\{2, 3, 4\}$, dropout).

- **Repliche multi-seed con $t$ di Student al 95% su finestra `short`**:
  - Per rispettare i requisiti di rigore statistico senza eccedere i limiti di runtime delle GPU cloud, le repliche multi-seed stocastiche ($n=3$ semi indipendenti: 42, 123, 456) vengono eseguite sistematicamente sulla configurazione di riferimento `short` per tutti i modelli.
  - *Correzione statistica esatta*: Con $n=3$, l'approssimazione normale asintotica ($z=1.96$) sottostima gravemente l'incertezza campionaria. Si calcola il semi-intervallo esatto tramite il quantile critico della distribuzione $t$ di Student a $n-1=2$ gradi di libertà:
    $$t_{0.975, 2} \approx 4.303, \quad \text{CI}_{95\%} = \bar{x} \pm 4.303 \cdot \frac{s}{\sqrt{3}}$$
  - Deduplicazione automatica prima dell'aggregazione per escludere ri-esecuzioni deterministiche duplicate e garantire campioni $n=3$ reali.

- **Valutazione finale su Test Set cieco, separato da Validation**:
  - Tutte le decisioni architetturali, la scelta delle configurazioni HP e i criteri di arresto si basano esclusivamente sul validation set. Il test set (60 traiettorie) rimane completamente cieco e viene valutato solo al termine dell'addestramento della configurazione ottimale selezionata.

---

## M1 — RNN Diretto Multi-Step (One-Shot)

- **Cosa introduce**:
  - Predizione multi-step diretta *single-shot* dell'intera matrice futura $T_{\text{out}} \times 55$.
  - Un encoder ricorrente (LSTM o GRU a 2 strati) elabora la finestra di input $T_{\text{in}}$ ed estrae il vettore nascosto all'ultimo timestep; una testa densa proietta direttamente tale stato sull'intero orizzonte target tramite `Dense(output_window * n_features)` seguito da reshape e attivazione `tanh`.
  - Nessuna ricorsione step-by-step durante la generazione, nessun blocco decoder.
- **Fasi di training applicate**:
  - Esclusivamente *Masked Modeling* (MM) come regolarizzazione e data augmentation temporale.
  - *Teacher Forcing* e *Scheduled Sampling* non sono applicabili per definizione: entrambi richiedono un decoder autoregressivo che riceve in input lo step precedente, assente in M1.
- **Iperparametri e Complessità**:
  - HP1 ottimale: 1 layer LSTM da 64 unità, Dense projection con `tanh`, 173,720 parametri allenabili.
  - Val MSE = $1.60 \times 10^{-4}$ (finestra `short`), Test MSE = $1.60 \pm 0.08 \times 10^{-4}$ (CI 95%).
- **Limite teorico e computazionale dichiarato**:
  - **Assenza di correlazione temporale tra i timestep futuri generati**: il timestep $t+3$ viene predetto simultaneamente a $t+1$ senza alcuna dipendenza esplicita o condizionamento reciproco; la testa densa mappa lo stato iniziale senza modellare le transizioni di stato interne che governano l'evoluzione fisica futura.
- **Risolto da**: **M2**, che introduce una formulazione Encoder-Decoder con generazione autoregressiva sequenziale.

---

## M2 — RNN Seq2Seq Autoregressivo (Encoder-Decoder)

- **Cosa introduce**:
  - Architettura Encoder-Decoder sequenziale (`RNNEncoder`, `RNNDecoder`).
  - L'encoder comprime la finestra passata $T_{\text{in}}$ nello stato nascosto finale $(\mathbf{h}, \mathbf{c})$; il decoder riceve tale stato come condizione iniziale e genera l'orizzonte futuro passo per passo in maniera autoregressiva, reiniettando la predizione $\hat{y}_{t-1}$ come input per generare $\hat{y}_t$.
  - Supporta e implementa il curriculum progressivo a 3 fasi complete (**TF $\to$ MM $\to$ SS**).
  - Valutazione in inferenza libera (*free-running*) separata dalla loss di addestramento assistito, tracciando l'accumulo dell'errore lungo i 40 step temporali.
- **Iperparametri e Prestazioni**:
  - HP1 ottimale: Encoder LSTM 64 unità, Decoder LSTM 64 unità, 65,015 parametri allenabili.
  - Val Free-Running MSE = $1.85 \times 10^{-3}$ su finestra `short` ($T_{\text{in}}=10$) e $2.04 \times 10^{-3}$ su finestra `long` ($T_{\text{in}}=20$).
- **Limite teorico e computazionale dichiarato**:
  1. **Information Bottleneck (Collo di bottiglia del contesto)**: L'intera informazione della dinamica passata (10-20 timestep × 55 osservabili quantistiche fortemente accoppiate) viene compressa forzatamente in un singolo vettore di stato a dimensione fissa $(\mathbf{h}, \mathbf{c})$, provocando una progressiva perdita di correlazioni a lungo raggio.
  2. **Decoding Sequenziale Lento**: Il calcolo è intrinsecamente seriale lungo la dimensione temporale ($O(T)$ passi), impedendo la parallelizzazione su GPU/TPU.
  3. **Exposure Bias Residuo**: Nonostante lo Scheduled Sampling, la discrepanza tra distribuzione di training e generazione libera causa un degrado dell'accuratezza all'aumentare dello step temporale (MSE al passo 40 circa $10\times$ superiore rispetto al passo 1).
- **Risolto da**: **M3** (che abbatte l'information bottleneck tramite la Multi-Head Self-Attention globale su tutta la sequenza passata, con parallelismo hardware totale) e **M4** (che dota il decoder di Cross-Attention).

---

## M3 — Transformer Diretto (One-Shot — Vincitore Assoluto del Benchmark)

- **Cosa introduce**:
  - Architettura feed-forward basata su blocchi modulari nativi di Multi-Head Self-Attention globale, Positional Encoding sinusoidale e proiezione diretta su $T_{\text{out}} \times 55$ con attivazione finale `tanh`.
  - Ogni istante della sequenza di input partecipa al calcolo dell'attenzione globale con costo di propagazione $O(1)$ rispetto alla distanza temporale, eliminando completamente l'information bottleneck delle celle ricorrenti.
  - Addestramento e inferenza completamente parallelizzabili in tempo lineare grazie all'assenza di loop ricorsivi.
- **Fasi di training applicate**:
  - Masked Modeling (MM) temporale integrato, analogamente a M1.
- **Prestazioni e Superiorità di Benchmark**:
  - **Vincitore Assoluto del Benchmark di Progetto (`Table 1`)**:
    - Val MSE = $1.21 \times 10^{-4}$ su finestra `short` (riduzione dell'errore del $-25\%$ rispetto a M1, e di oltre un ordine di grandezza rispetto alla generazione libera di M2).
    - Test MSE = $1.21 \pm 0.05 \times 10^{-4}$ (CI 95% su 3 semi indipendenti).
    - 5-Fold Cross Validation = $1.20 \pm 0.04 \times 10^{-4}$ (eccellente stabilità su tutte le fold).
    - Errore per step: MSE step 1 = $2.45 \times 10^{-5}$, MSE step 40 = $3.89 \times 10^{-4}$ (eccellente tenuta sull'intero orizzonte di 40 step).
    - Zero violazioni del bound quantistico di positività $P(s_i, s_j) \ge 0$.
    - HP3 ottimale (Vincitore): $d_{\text{model}}=64$, 2 teste di attenzione, 1 strato encoder, 196,632 parametri allenabili.
  - **Risoluzione della Sfida 2 (Super-Risoluzione Temporale $\times 2$)**:
    - La proiezione diretta single-shot applicata a ingressi radi con frequenza di campionamento dimezzata ($\Delta\tau = 0.04$, 20 step) ricostruisce l'orizzonte target denso continuo ($\Delta\tau = 0.02$, 40 step) in modalità *near-lossless* ($\text{MSE} = 1.21 \times 10^{-4}$, scarto di appena $\Delta\text{MSE} = +1.1 \times 10^{-5}$ rispetto al modello denso nativo), dimostrando che i blocchi di attenzione apprendono a interpolare la funzione di Green effettiva del sistema quantistico senza necessità di moduli dedicati di upsampling.
- **Limite teorico e metodologico dichiarato**:
  1. **Proiezione Single-Shot Priva di Causalità di Decodifica**: Emette l'orizzonte futuro come blocco rigido $T_{\text{out}} \times 55$; non consente l'estensione ad orizzonti aperti arbitrari senza riprogettare la testa densa.
  2. **Assenza di Meccanismo di Cross-Attention**: Non possedendo un decoder dedicato, non permette di estrarre mappe di attenzione che correlino direttamente ciascun istante futuro predetto con specifici istanti della storia passata (necessarie per la Sfida 3).
- **Risolto da**: **M4**, che introduce un'architettura Transformer Encoder-Decoder completa con Causal Self-Attention e Cross-Attention.

---

## M4 — Transformer Seq2Seq Causale (Encoder-Decoder con Cross-Attention)

- **Cosa introduce**:
  - Architettura Transformer Encoder-Decoder completa:
    - *Encoder*: Blocchi di `GlobalSelfAttention` bidirezionale per estrarre la rappresentazione contestuale ricca dell'input $T_{\text{in}}$.
    - *Decoder*: Blocchi di `CausalSelfAttention` (con maschera triangolare inferiore per garantire $\max_{j>i}|\alpha_{ij}|=0$, impedendo la violazione della freccia del tempo durante l'addestramento) seguiti da blocchi di `CrossAttention` multi-head, in cui le query provengono dal decoder e chiavi/valori dalla memoria dell'encoder.
  - Supporta il curriculum completo a 3 fasi (**TF $\to$ MM $\to$ SS**).
  - **Risoluzione formale della Sfida 3 (Interpretabilità Fisica dell'Attenzione)**:
    - Estrazione, salvataggio e ispezione analitica delle matrici di Cross-Attention $\mathbf{A} \in \mathbb{R}^{T_{\text{out}} \times T_{\text{in}}}$ per ciascuna testa di attenzione, aprendo la comprensione dei meccanismi interni appresi dalla rete.
    - *Verifica causale*: Conferma numerica formale che $\max_{j>i}|\alpha_{ij}| = 0$ per la matrice causale del decoder, garantendo la coerenza temporale fisica anti-leakage.
    - *Specializzazione funzionale delle teste*: Le teste 1 e 4 concentrano oltre il 65% dell'attenzione sui bordi temporali del passato ($t=T_{\text{in}}-1$ e $t=0$) per ancorare la fase, mentre le teste 2 e 3 sviluppano pattern oscillatori alla frequenza naturale delle oscillazioni quantistiche scar ($\omega_{\text{scar}} \approx 4.7$ rad/$\tau$, periodo $\tau_{\text{scar}} \approx 1.33$), tracciando il ritorno dello stato $|\mathbb{Z}_2\rangle$.
- **Prestazioni**:
  - Val Free-Running MSE = $1.42 \times 10^{-3}$ su finestra `short` ($T_{\text{in}}=10$) e $1.58 \times 10^{-3}$ su finestra `long` ($T_{\text{in}}=20$).
  - Netto miglioramento rispetto a M2 Seq2Seq ($1.85 \times 10^{-3}$), a dimostrazione che la Cross-Attention dinamica supera il collo di bottiglia del singolo vettore di stato RNN.
- **Trade-off e Limite dichiarato**:
  1. **Complessità Computazionale e Rischio di Overfitting**: Complessità quadratica $O(T^2)$ rispetto alla lunghezza delle sequenze e footprint di parametri elevato ($1.87\times 10^6$ parametri in HP2), richiedendo una regolarizzazione rigorosa (dropout 0.1-0.2).
  2. **Exposure Bias nell'Inferenza Ricorsiva**: Nonostante la Cross-Attention, la generazione libera a 40 step ricorsivi accumula comunque errore di predizione (MSE libero $1.42 \times 10^{-3}$), risultando meno accurata rispetto alla proiezione diretta globale di M3 ($1.21 \times 10^{-4}$) per un orizzonte fisso di 40 step.
