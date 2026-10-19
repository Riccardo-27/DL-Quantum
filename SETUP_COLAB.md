# 🚀 Guida Rapida Colab per Riccardo (DL-Quantum)

Questa guida spiega passo-passo come configurare Google Colab in 3 minuti per eseguire i notebook del progetto senza dover modificare percorsi di file o configurazioni nel codice.

---

## 📌 Requisiti Preliminari (Da fare 1 sola volta)

### 1. Collegare la Cartella Condivisa di Google Drive
La cartella `DL-Quantum` su Google Drive contiene il dataset grezzo `trajectories.csv` (464 MB), la cache binaria `data/cache/trajectories.npy` (88 MB) e le partizioni congelate (`split_indices.npz`).

Per fare in modo che Colab la trovi sempre nel percorso standard `/content/drive/MyDrive/DL-Quantum`:
1. Apri **[Google Drive](https://drive.google.com)**.
2. Nel menu a sinistra, clicca su **"Condivisi con me"** (Shared with me).
3. Trova la cartella **`DL-Quantum`**.
4. Fai **click destro** sulla cartella -> seleziona **"Organizza"** (o "Aggiungi a") -> **"Aggiungi scorciatoia"** (Add shortcut).
5. Scegli come destinazione **"Il mio Drive"** (MyDrive) e conferma.

> **Perché è fondamentale**: Con questa scorciatoia, quando Colab monta Drive in `/content/drive`, vedrà la cartella esattamente in `/content/drive/MyDrive/DL-Quantum`. Entrambi useremo gli stessi identici percorsi automatici senza dover cambiare una riga di codice nei notebook!

---

### 2. Configurare il Token GitHub (Colab Secrets)
Per permettere a Colab di scaricare il codice aggiornato e caricare figure e risultati su GitHub senza dover inserire password ogni volta:

1. Su GitHub, vai in **Settings** (in alto a destra sulla tua foto profilo) -> **Developer Settings** (in fondo alla barra sinistra) -> **Personal access tokens** -> **Tokens (classic)**.
2. Clicca su **"Generate new token"** -> **"Generate new token (classic)"**:
   - **Note**: `Colab-DL-Quantum`
   - **Expiration**: 60 o 90 giorni (o a tua scelta)
   - **Scopes**: spunta la casella **`repo`** (Full control of repositories)
   - Scorri in fondo e clicca **Generate token**.
   - **Copia subito la stringa generata** (inizia con `ghp_...`).
3. Apri **[Google Colab](https://colab.research.google.com)** su un qualsiasi notebook.
4. Nella barra laterale sinistra di Colab, clicca sull'icona della **Chiave 🔑 (Segreti / Secrets)**.
5. Clicca su **"Aggiungi nuovo segreto"** (Add new secret):
   - **Nome**: `GH_TOKEN`
   - **Valore**: incolla il token (`ghp_...`)
   - **Accesso al notebook**: attiva la levetta (toggle su **ON**) per autorizzare l'accesso al notebook.
6. *(Opzionale ma utile)* Puoi aggiungere anche due altri segreti:
   - `GIT_USERNAME` con valore il tuo username GitHub (es. `Riccardo-27`)
   - `GIT_EMAIL` con valore la tua email associata a GitHub.

---

## 💻 Come Eseguire i Notebook su Colab

### 1. Aprire il Notebook
- Vai sul repository GitHub **[Riccardo-27/DL-Quantum](https://github.com/Riccardo-27/DL-Quantum)**.
- Entra nella cartella `notebooks/` e apri il notebook di lavoro (es. `01_data.ipynb`, `02_windowing.ipynb`, `03_rnn.ipynb`).
- Clicca sul badge in cima al notebook **"Open in Colab"** per aprirlo direttamente nel browser.
- Verifica che il runtime sia impostato su GPU:
  - Menu **Modifica** -> **Impostazioni blocco note** -> **Acceleratore hardware: GPU (T4)**.

### 2. Cella 0: Setup Automatico dell'Ambiente
- Esegui la prima cella (**Cella 0**):
  - Ti apparirà il popup per autorizzare l'accesso a Google Drive: clicca **"Connetti a Google Drive"**.
  - La cella rileverà automaticamente la cartella `DL-Quantum` su Drive.
  - Sincronizzerà l'ultimo codice dal repository Git via `git pull`.
  - Configurerà `sys.path`, imposterà il determinismo (`seed = 42`) e verificherà la presenza della GPU T4.

### 3. Cella Finale: Sincronizzazione dei Risultati
- Al termine dell'esecuzione del notebook, l'ultima cella si occupa di:
  - Verificare lo stato dei file modificati (`git status`).
  - Salvare i log degli esperimenti e le figure ad alta risoluzione (300 dpi).
  - Effettuare commit e push su GitHub utilizzando il token sicuro `GH_TOKEN` impostato nei Secrets, senza esporre password.

---

## 🛡️ Regole e Best Practice di Collaborazione

1. **Log concorrente separato**:
   - Ogni autore registra i propri esperimenti nel file dedicato `results/results_log_<autore>.csv` (es. `results_log_riccardo.csv` ed `results_log_ernesto.csv`). Il codice unisce automaticamente i risultati in fase di analisi senza mai rischiare conflitti di scrittura su Drive.
2. **File pesanti sempre fuori da Git**:
   - File pesanti come `trajectories.csv`, cache binaria `.npy` e pesi `.keras` risiedono solo su Google Drive e sono già esclusi dal versionamento in `.gitignore`.
3. **Salvataggio output nel notebook**:
   - In Google Colab, prima di chiudere la sessione o prima di inviare un commit, premi **`Ctrl + S`** per salvare il notebook completo di grafici e tabelle renderizzate.
