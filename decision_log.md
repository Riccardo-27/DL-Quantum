# Decision Log — DL-Quantum

Traccia formale delle decisioni progettuali, metodologiche e architetturali adottate nello sviluppo del progetto. La struttura documenta le scelte fisiche e di machine learning, evidenziando per ogni modello il **limite teorico o computazionale intrinseco e come viene superato dal modello successivo** (roadmap incrementale per report e presentazione).

---

## Fondamenta Fisiche e Preprocessing

- **Partizionamento per Traiettoria (280 Train / 60 Validation / 60 Test, Seed 42)**:
  - *Decisione*: Lo split è eseguito a livello di intere traiettorie fisiche indipendenti (70% train, 15% validation, 15% test).
  - *Motivazione*: Uno split casuale per riga genererebbe un massiccio data leakage temporale, poiché timestep consecutivi appartengono alla medesima evoluzione dinamica quantistica. Il partizionamento per traiettoria assicura che il modello sia valutato su condizioni iniziali e parametri hamiltoniani mai visti durante il training.

- **Dati Grezzi (Nessuna Normalizzazione) e Attivazione Finale `tanh`**:
  - *Decisione*: Nessun riscalamento standard (né z-score né min-max per feature) e layer di output vincolato con attivazione `tanh`.
  - *Motivazione Fisica e Statistica*: Le 55 osservabili quantistiche (10 magnetizzazioni $\langle \sigma_z^i \rangle$ e 45 correlazioni a due corpi $\langle \sigma_z^i \sigma_z^j \rangle$) sono valori di aspettazione di operatori di spin di Pauli, rigorosamente confinati a livello quantomeccanico nell'intervallo compatto $[-1, 1]$.
  - *Rigetto dello Z-Score e Min-Max*: Le correlazioni a lungo raggio oscillano su ampiezze fisiche ridotte (es. tra $0.01$ e $0.05$). Una normalizzazione per-feature riscalerebbe tali oscillazioni amplificandone artificialmente il rumore di fondo, distruggendo il significato assoluto dello zero (assenza di polarizzazione o correlazione) e alterando i rapporti di grandezza tra osservabili.
  - *Preservazione delle Mappe di Attenzione per i Transformer*: Nei Transformer, i meccanismi di attenzione calcolano prodotti scalari $\mathbf{Q}\mathbf{K}^T$. Preservare le scale naturali impedisce che correlazioni fisicamente trascurabili dominino indebitamente i pesi di attenzione rispetto alle magnetizzazioni locali dominanti.
  - *Prevenzione del Drift in Free-Running*: L'attivazione `tanh` nell'ultimo layer impone analiticamente che ogni valore predetto resti in $[-1, 1]$, impedendo esplosioni numeriche e divergenze non fisiche durante l'inferenza autoregressiva ricorsiva a lungo orizzonte.

- **Integrità Quantistica e Disaccoppiamento Iniziale**:
  - *Verifica*: Controllo della positività semidefinita delle matrici di densità a 2 corpi:
    $$P(s_i, s_j) = \frac{1}{4}\left(1 + s_i \langle \sigma_z^i \rangle + s_j \langle \sigma_z^j \rangle + s_i s_j \langle \sigma_z^i \sigma_z^j \rangle\right) \ge 0, \quad \forall s_i, s_j \in \{-1, +1\}$$
    Zero violazioni rilevate sull'intero dataset (minimo $0.1553 > 0$).
  - A $t=0$, lo scarto medio $|c_{ij} - m_i m_j| = 0.0158$ conferma stati iniziali di prodotto (product states) fisicamente coerenti con l'evoluzione del modello PXP.

---

## Finestrature Temporali e Metodologia di Valutazione

- **Griglia Comune di Finestre (`short` vs `long`)**:
  - *Configurazione*: Finestra `short` ($T_{in}=10 \to T_{out}=40$) e finestra `long` ($T_{in}=20 \to T_{out}=40$), entrambe con $\text{stride}=10$ fisso e campionamento nativo $\Delta t = 0.02$.
  - *Motivazione*: Mantenere costante l'orizzonte futuro target $T_{out}=40$ ($0.80$ unità fisiche di tempo) isola la memoria passata $T_{in}$ come unica variabile indipendente, consentendo un confronto diretto e controllato tra le architetture.

- **Orizzonte di Predizione $T_{out}=40$ e Scelta dello Stride**:
  - *Orizzonte*: Un orizzonte $T_{out}=40$ garantisce un intervallo temporalmente significativo (copre molteplici oscillazioni del ciclo limite quantistico), bilanciando l'accumulo di errore dell'inferenza autoregressiva e il costo computazionale quadratico rispetto a orizzonti più lunghi.
  - *Stride = 10*: Disaccoppiare lo stride dalla lunghezza della finestra evita la sovrapposizione estrema (che con $\text{stride}=5$ produrrebbe finestre duplicate al 93.75%, gonfiando il training senza apportare nuova dinamica) e previene al contempo la perdita di campioni statistici tipica di finestre non sovrapposte ($\text{stride}=40$).

- **Allineamento Target per la Sfida 1 (`align_t_in = 40`)**:
  - *Problema*: Nel testare la minima finestra di osservazione $T_{in} \in \{5, 10, 20, 40\}$, finestre non allineate valuterebbero regimi dinamici differenti (finestre brevi valuterebbero il transiente iniziale $t \in [0.10, 0.90]$, finestre lunghe regimi asintotici $t \in [0.80, 1.60]$), introducendo un forte bias comparativo.
  - *Soluzione*: Imposizione di un punto di inizio comune per il target futuro $Y$ ($t_{start} \ge 40 + k \cdot 10$). In questo modo, tutte le finestre predicono rigorosamente lo stesso intervallo fisico futuro a parità di target $Y$.

- **Rigore Statistico: Intervalli di Confidenza al 95% con $t$ di Student**:
  - *Decisione*: Valutazione multi-seed su 3 semi stocastici indipendenti ($n=3$) riportando media, deviazione standard e semi-intervallo di confidenza al 95%.
  - *Correzione Metodologica*: Per $n=3$, l'approssimazione gaussiana ($z=1.96$) sottostima gravemente l'incertezza campionaria. Si adotta il quantile critico esatto della distribuzione $t$ di Student a $n-1=2$ gradi di libertà:
    $$t_{0.975, 2} \approx 4.303, \quad \text{CI}_{95\%} = \bar{x} \pm 4.303 \cdot \frac{s}{\sqrt{3}}$$

- **Valutazione Finale su Test Set Cieco**:
  - Le decisioni di model selection, scelta degli iperparametri e soglie di arresto si basano esclusivamente sull'errore sul validation set. Il test set rimane isolato e viene valutato una sola volta a pipeline completata, garantendo l'onestà statistica della generalizzazione.

---

## Strategia di Addestramento Ibrido (TF → MM → SS)

I modelli autoregressivi soffrono tipicamente di **Exposure Bias** (o *Covariate Shift*): durante il training convenzionale ricevono in ingresso gli input storici esatti (*Teacher Forcing*), mentre in fase di inferenza ricorsiva (*Free-Running*) operano alimentandosi delle proprie predizioni precedenti, accumulando rapidamente errori lungo l'orizzonte futuro.

Per mitigare tale discrepanza, l'addestramento segue un protocollo curriculum progressivo a 3 fasi:

1. **Fase 1: Teacher Forcing (TF)**:
   - *Meccanismo*: Il modello riceve come ingresso al decoder il ground-truth shiftato temporalmente.
   - *Obiettivo*: Stabilizzazione iniziale dei gradienti e convergenza rapida verso la traiettoria globale media.
2. **Fase 2: Masked Modeling (MM)**:
   - *Meccanismo*: Mascheramento casuale di interi timestep nel contesto passato dell'input (azzeramento del vettore spaziale delle 55 feature), preservando sempre l'ultimo istante temporale.
   - *Obiettivo*: Data augmentation temporale che forza la rete a catturare la memoria a lungo raggio della catena quantistica senza dipendere unicamente da transizioni locali immediate.
3. **Fase 3: Scheduled Sampling (SS)**:
   - *Meccanismo*: Curriculum learning in cui l'ingresso al decoder al timestep $t$ è una combinazione stocastica tra il target reale $y_{t-1}$ e la predizione della rete $\hat{y}_{t-1}$, con probabilità $p_{ss}$ che cresce linearmente da $0.0$ a $p_{end} \in [0.4, 0.7]$.
   - *Obiettivo*: Abituare il modello a correggere le proprie imperfezioni predittive, rendendo la dinamica robusta durante l'inferenza libera a 40 step.

---

## Roadmap Incrementale dei Modelli

La transizione tra i modelli è guidata dall'identificazione dei limiti intrinseci di ciascuna architettura:

```
M1 (RNN Diretto)
  │  Limita: Nessuna modellazione temporale tra i target futuri
  ▼
M2 (RNN Seq2Seq Autoregressivo)
  │  Limita: Bottleneck del contesto a stato fisso; decoding sequenziale lento
  ▼
M3 (Transformer Diretto)
  │  Limita: Proiezione single-shot senza decodifica autoregressiva
  ▼
M4 (Transformer Seq2Seq Causale)
     Supera: Attenzione multi-head globale + causal decoding + interpretabilità fisica
```

### M1 — Modello Ricorrente Diretto (Multi-Step Baseline)

- **Architettura**: Encoder ricorrente (LSTM o GRU a 2 strati) che processa la finestra di input $T_{in}$ ed estrae il vettore di stato all'ultimo timestep; una testa densa proietta direttamente lo stato finale sulla matrice completa di output $T_{out} \times 55$ tramite un layer `Dense(output_window * n_features)` seguito da un reshape e attivazione `tanh`.
- **Fasi di Training**: Si applica esclusivamente il *Masked Modeling*. *Teacher Forcing* e *Scheduled Sampling* non sono applicabili poiché M1 non possiede un decoder passo-passo.
- **Limite Teorico**: **Assenza di correlazione temporale tra le predizioni future**. Il timestep futuro $t+k$ è generato simultaneamente a $t+1$ senza alcuna dipendenza esplicita; il modello non ha modo di condizionare lo stato al tempo $t+k$ sulla traiettoria che esso stesso ha generato negli istanti precedenti.
- **Superamento**: Risolto da **M2**, che introduce un decoder autoregressivo ricorrente.

---

### M2 — Modello Ricorrente Seq2Seq (Encoder-Decoder LSTM/GRU)

- **Architettura**: Architettura a due stadi. L'encoder LSTM/GRU comprime la sequenza passata nello stato nascosto $(\mathbf{h}, \mathbf{c})$; il decoder riceve tale stato come condizione iniziale e genera l'orizzonte futuro passo per passo in maniera autoregressiva, alimentando la predizione al tempo $t$ come ingresso al tempo $t+1$.
- **Fasi di Training**: Supporta l'intero ciclo a 3 fasi (**TF → MM → SS**). Valutato in inferenza pura *free-running* senza alcun accesso ai target futuri.
- **Limite Teorico**:
  1. **Information Bottleneck**: L'intera informazione della storia passata è compressa forzatamente in un vettore di dimensione fissa $(\mathbf{h}, \mathbf{c})$, limitando la capacità di preservare dipendenze temporali complesse su orizzonti lunghi.
  2. **Decoding Sequenziale**: Il calcolo è intrinsecamente seriale lungo la dimensione temporale, impedendo la parallelizzazione su acceleratori hardware GPU/TPU.
- **Superamento**: Risolto da **M3** e **M4**, che eliminano il collo di bottiglia tramite l'attenzione su tutta la sequenza passata.

---

### M3 — Transformer Diretto (Self-Attention Baseline)

- **Architettura**: Encoder a blocchi di Multi-Head Self-Attention e feed-forward con Positional Encoding sinusoidale; la sequenza di embedding viene aggregata ed espansa sull'orizzonte futuro tramite proiezione diretta.
- **Punto di Forza**: L'attenzione globale permette a ogni istante futuro di osservare l'intera storia passata senza compressione di stato e con addestramento completamente parallelizzabile.
- **Limite Teorico**: Pur catturando le correlazioni passate in modo superiore rispetto alle RNN, la proiezione finale rimane diretta (single-shot), mancando di un meccanismo di decodifica autoregressivo che guidi l'evoluzione causale passo per passo.
- **Superamento**: Risolto da **M4**, che introduce un decoder con causal masking e cross-attention.

---

### M4 — Transformer Seq2Seq Causale (Architettura Avanzata)

- **Architettura**: Transformer Encoder-Decoder completo. Il decoder impiega masked self-attention causale (per impedire la violazione della freccia del tempo durante il training) e cross-attention multi-head per focalizzarsi dinamicamente sugli istanti salienti della sequenza passata.
- **Punti di Forza**:
  - Combina la ricchezza rappresentativa dell'attenzione globale con la coerenza temporale della generazione sequenziale causale.
  - Consente l'estrazione e la visualizzazione diretta delle matrici di attenzione (Sfida 3), aprendo la "scatola nera" del modello.
- **Trade-off e Limite**: Complessità computazionale quadratica $O(T^2)$ rispetto alla lunghezza delle sequenze e maggior numero di parametri, che richiede un attento controllo della regolarizzazione (dropout, weight decay) per prevenire l'overfitting su piccoli volumi di dati.

---

## Sfide Specifiche di Progetto

### Sfida 1: Lunghezza Minima della Finestra di Osservazione ($T_{in}$)
- **Obiettivo**: Determinare quanti istanti di storia passata siano strettamente necessari per prevedere accuratamente la dinamica a lungo termine.
- **Metodologia**: Valutazione comparativa su una griglia sistematica $T_{in} \in \{5, 10, 20, 40\}$ (corrispondenti a span temporali da $0.10$ a $0.80$ unità fisiche), mantenendo invariato $T_{out}=40$ e garantendo l'invarianza del target tramite allineamento $T_{in}^{align}=40$.
- **Ipotesi Fisica**: Sotto una soglia critica di osservazione (inferiore al periodo delle oscillazioni di Rabi a molti corpi), la rete non può distinguere lo stato quantistico di fase corrente, portando a un rapido degrado predittivo.

### Sfida 2: Super-Risoluzione Temporale ed Imputazione
- **Obiettivo**: Valutare la capacità del modello di ricostruire la dinamica quantistica continua ad alta frequenza ($\Delta t = 0.02$) a partire da campionamenti radi a frequenza dimezzata ($\Delta t = 0.04$).
- **Metodologia**: Subsampling $2\times$ dell'input temporale $X_{sr}$ con target $Y$ mantenuto a risoluzione densa nativa ($\Delta t = 0.02$).
- **Rilevanza Fisica**: Simula contesti sperimentali reali in cui la frequenza di lettura dei qubit Rydberg è limitata dall'hardware di misura, testando la capacità dell'architettura di apprendere la funzione di Green effettiva del sistema.

### Sfida 3: Interpretabilità Fisica tramite Mappe di Attenzione
- **Obiettivo**: Mappare i pattern di attenzione multi-head appresi da M4 per verificare se la rete riscopre autonomamente principi fisici della dinamica quantistica many-body.
- **Ipotesi di Validazione**:
  1. *Cono di Luce di Lieb-Robinson*: La propagazione dell'informazione nella catena 1D deve mostrare una velocità finita di diffusione delle correlazioni spaziali.
  2. *Blocco di Rydberg*: I pesi di attenzione tra atomi primi vicini devono riflettere l'inibizione dell'eccitazione simultanea dovuta al forte potenziale di interazione $V_{ij} \sim C_6 / R^6$.
