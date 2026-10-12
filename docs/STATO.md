# ChargeShield-FL — Stato del progetto

> **Stato: documento CANONICO.** Aggiornato il 2026-10-02 (revisione: costo sul modello
> rilasciato, analisi per record corretta, E-C, segnalazioni 46-59, controlli del protocollo, celle delle matrici ricomposte, RQ2 senza DP,
> punto operativo a 5 seed, RQ2 con DP, E-B a σ = 1 e 2 con il riferimento GroupNorm, RQ3 senza DP; il 2026-09-29 E-B a σ = 5, RQ3 sul Mac principale con e senza DP, prove su mu e FedAvg con DP su Windows; il 2026-09-30 lo screening di C per FedAvg con DP e il solo clipping di E-B; il 2026-10-01 il braccio B del canary su più siti, il braccio FedProx con DP di Windows e il braccio A del canary su Caltech della quarta macchina, il test appaiato per record su tutte le 24 celle a 5 seed, il seed 42 del canary su Caltech; il 2026-10-02 la linea per il paper DSN e l'analisi per round di RQ3; dai JSON e non ancora nelle matrici). Solo numeri
> presenti in `risultati/` o letti dai JSON grezzi alla data, e lo dice dove.
> Linea scientifica: la guida. Cosa fare: `ESPERIMENTI.md`. Cosa esiste nel codice:
> `SISTEMA.md`. Questo file risponde a una sola domanda: a che punto siamo.

## 1. In una frase

Nel regime naturale non c'è un'esposizione misurabile da ridurre: in media nessun
attacco distingue membri da non membri, con o senza DP, e il test appaiato per record
non trova segnale in nessuna delle celle analizzate (sezioni 3.2 e 3.9). L'eccesso per
record senza DP, letto finora come l'unico segnale, compare identico fra i non membri:
è stabilità del ranking, non appartenenza (segnalazione 47). Il costo, misurato sul
modello globale rilasciato (segnalazione 46), supera la soglia di 3 volte in ogni cella
client-level: a ε = 16 è 60 volte il no-DP appaiato, e il punto operativo scelto con la
griglia, C = 1 con ε = 64, vale 4.5 volte a 5 seed (sezione 3.1b). La DP per record costa
1.9 volte a σ = 1 e 2.1 volte a σ = 2 rispetto al no-DP con la stessa normalizzazione
(GroupNorm), sotto la soglia, con un ε per record di circa 30 e 10, e 2.3 volte a σ = 5 con
ε di circa 3; il solo clipping per esempio non ha un costo misurabile (1.04 volte), quindi il
costo viene dal rumore (sezione 3.5). La partizione IID o per sito non cambia il
successo degli attacchi, con o senza DP (sezione 3.9). Senza DP FedAvg ha la loss
sull'holdout circa 3.7-3.9 volte più bassa di FedProx su due macchine; con DP a C = 1 l'ordine
si inverte: su Windows FedProx ha 0.34 volte la loss di FedAvg, 5 seed su 5 (t = −4.62 sul
logaritmo, p = 0.0099), sul Mac principale 0.53, 4 seed su 5, non significativo; alzare C fino
a 8 non aiuta FedAvg (seed 42, sezione 3.10). Il canary su più siti, a un seed, vede l'appartenenza sull'update di
Office 1 al round 1 e nessun segnale sul modello globale (sezione 3.3); il canary bilanciato su
Caltech al seed 42 vede l'appartenenza in tutti e tre i round (somme A+B fra 1.57 e 1.65
contro 1.00, sezione 3.3); gli altri 4 seed sono in corso (`ESPERIMENTI.md`).

## 2. Glossario minimo

I termini che non si leggono da soli. La descrizione tecnica completa è in
`SISTEMA.md`, sezioni 4 e 7.

| termine | significato |
|---|---|
| client-level / record-level | due meccanismi DP diversi. Il primo protegge il contributo del client e il suo ε è un parametro di calibrazione; il secondo è DP-SGD e protegge il record, la stessa unità che gli attacchi misurano. Si attivano uno alla volta; il record-level richiede `--no-dp`. |
| A0, A1, A2, A3 | cosa vede l'avversario sull'update del client: grezzo senza DP; grezzo prima di clip e rumore (dp-fedavg); clippato (central); clippato e rumorizzato (local). Ordinati per informazione. Solo per il client-level. |
| loss grezza | errore di ricostruzione senza calibrazione. È la metrica che regge; LiRA calibrato è degenere (3.4). |
| record segnalati, z | sessioni membro in almeno 2 seed con percentile medio ≥ 90 e minimo ≥ 75; z misura quante deviazioni standard il conteggio dista dal livello di caso ottenuto per permutazione dentro il seed. Misura la stabilità del ranking, non l'appartenenza (segnalazione 47). |
| test appaiato | per le sessioni membro in un seed e non membro in un altro, differenza fra il percentile medio da membro e da non membro; z = media / errore standard. È la lettura primaria per record dal 2026-09-24. |
| loss sull'holdout del modello rilasciato / loss locale | la prima è l'MSE del modello globale finale sull'holdout ed è il costo; la seconda è la loss di addestramento locale dell'ultimo round, solo diagnostica (segnalazione 46). |
| regime naturale / canary | dati come sono / memorizzazione indotta con template duplicati, capacità 3.3x, feature temporale quasi univoca, 1000 epoche. Il canary valida lo strumento, non certifica il null naturale. |

## 3. Cosa è stato fatto e regge

### 3.1 Campagna principale, regime naturale, client-level

Fonte: `risultati/Matrice_sintesi.xlsx`, foglio `Utility_privacy_limite`, rigenerato il
2026-09-24 con il costo corretto (segnalazione 46) e con la composizione delle celle di
`check_significance.py` (segnalazioni 44, 45, 53): una run per seed, la più recente,
cartelle `_*`, NVFlare, entity-split e fedmia-gradient escluse. Il riferimento no-DP sono
le 5 run di `nodp-sweep2` (holdout 0.00158); prima della correzione erano 49 run, fra cui
le calibrazioni di overfitting, con holdout 0.00244 e LiRA composto 0.513, e tutti i
rapporti risultavano più bassi. AUC-ROC media sulle run; `n` = run nella cella. Le colonne
di costo sono rapporti sul no-DP: la prima sul modello rilasciato (il costo), la seconda
sulla loss locale (diagnostica).

| cella | n | costo: holdout del modello rilasciato | loss locale (diagnostica) | LiRA composto | Yeom | Shadow | TPR a FPR 1% |
|---|---|---|---|---|---|---|---|
| no-DP (A0) | 5 | 1 | 1 | 0.5004 | 0.5003 | 0.5016 | 0.0105 |
| dp-fedavg ε=64 (punto operativo) | 5 | 4.5 | 1.6 | 0.5011 | 0.5004 | 0.5013 | 0.0105 |
| dp-fedavg ε=16 (A1) | 5 | 59.7 | 3.7 | 0.4992 | 0.5015 | 0.5011 | 0.0097 |
| dp-fedavg ε=8 | 5 | 119.9 | 39.1 | 0.4978 | 0.5001 | 0.5002 | 0.0101 |
| dp-fedavg ε=4 | 5 | 188.4 | 76.2 | 0.4959 | 0.4999 | 0.5002 | 0.0092 |
| dp-fedavg ε=2 | 5 | 230.6 | 169.9 | 0.4957 | 0.5004 | 0.5005 | 0.0096 |
| dp-fedavg ε=1 | 5 | 249.1 | 241.1 | 0.4971 | 0.5002 | 0.5000 | 0.0099 |
| dp-fedavg ε=0.5 | 5 | 257.2 | 252.2 | 0.4980 | 0.4998 | 0.4993 | 0.0095 |
| dp-fedavg ε=0.1 | 5 | 266.8 | 251.9 | 0.4996 | 0.4995 | 0.4994 | 0.0089 |
| central ε=1 (A2) | 5 | 261.5 | 238.1 | 0.5024 | 0.5006 | 0.5001 | 0.0102 |
| central ε=0.5 | 5 | 278.6 | 284.1 | 0.5003 | 0.5002 | 0.4996 | 0.0096 |
| central ε=0.1 | 5 | 274.9 | 287.1 | 0.5020 | 0.5000 | 0.4992 | 0.0105 |
| local ε=1 (A3) | 5 | 249.1 | 241.1 | 0.4983 | 0.5002 | 0.5000 | 0.0108 |
| local ε=0.5 | 5 | 257.2 | 252.2 | 0.4975 | 0.4998 | 0.4993 | 0.0101 |
| local ε=0.1 | 5 | 266.8 | 251.9 | 0.4986 | 0.4995 | 0.4994 | 0.0092 |

Tre cose da tenere insieme leggendo la tabella. Il nullo aggregato vale anche
senza DP. Nelle celle di E-A (ε ≤ 16 per round) il modello rilasciato costa almeno 60 volte il
riferimento, e al punto operativo ε = 64 in media 4.5 volte, sopra la soglia di 3 (sezione 3.1b):
il nullo sotto DP non distingue "la DP protegge" da "non c'è nulla
da proteggere"; fino al 2026-09-24 la colonna di costo usava la loss locale e faceva
sembrare ε = 16 e 8 celle con utility residua. Nota storica che
conta per la Tabella 2 del paper: dp-fedavg e local hanno coinciso bit a bit fino
al 15 settembre perché eseguivano lo stesso codice; da allora dp-fedavg attacca
l'update grezzo (A1), e quali celle siano post modifica va verificato nel
registro. Il test di equivalenza TOST a margine 0.02 è soddisfatto ovunque, calcolato
però sul LiRA composto e non sulla metrica primaria (segnalazione 50); il Wilcoxon a 5
seed non può essere significativo per costruzione.

### 3.1b Sweep di ε, client-level `dp-fedavg` (A1): E-A

Fonte: i JSON di `experiments/rq1-eps{2,4,8,16}/` e `nodp-sweep2` letti il 2026-09-24,
una run per seed (42, 123, 456, 789, 1234), la più recente: per ε = 2 seed 789 è il
rilancio del 24 settembre, con gli attacchi e la stessa loss del primo tentativo bit per
bit. Costo = loss sull'holdout del modello globale rilasciato al round 10
(segnalazione 46), rapportata a `nodp-sweep2` (media 0.00158), sulla media e appaiata
per seed. La loss locale è la vecchia colonna, tenuta come diagnostica.

| ε | seed | holdout del modello rilasciato | rapporto su no-DP | appaiato, min-max | loss locale | Yeom | Shadow | LiRA composto | TPR a FPR 1% |
|---|---|---|---|---|---|---|---|---|---|
| 16 | 5 | 0.0945 | 60 | 30-157 | 0.00614 | 0.5015 | 0.5011 | 0.4992 | 0.0097 |
| 8 | 5 | 0.190 | 120 | 67-490 | 0.0655 | 0.5001 | 0.5002 | 0.4978 | 0.0101 |
| 4 | 5 | 0.299 | 188 | 131-490 | 0.128 | 0.4999 | 0.5002 | 0.4959 | 0.0092 |
| 2 | 5 | 0.366 | 231 | 153-495 | 0.285 | 0.5004 | 0.5005 | 0.4957 | 0.0096 |

Nessuna cella sta sotto la soglia di 3 volte: è la terza riga della tabella di lettura
di `ESPERIMENTI.md` ("nessun punto operativo utile per il client-level in questo
regime"). Sull'holdout la variabilità fra seed è di circa 3 volte in ogni cella (a ε = 8
0.109-0.341): l'incoerenza di ε = 8, che veniva dalla loss locale (0.014-0.195), non c'è
più e i seed aggiuntivi non servono. TOST soddisfatto in tutte e quattro le celle, sul LiRA
composto (segnalazione 50). I JSON registrano `git_commit`: f145b79 ed ed0e1f9, e
478d471 per il rilancio; il codice di training e attacco è lo stesso.

Punto operativo: griglia C ∈ {0.25, 0.5, 1} × ε ∈ {16, 64, 256}, seed 42 (JSON in
`experiments/_op_*` letti il 2026-09-25, fuori dalle matrici perché prove a un seed). Costo
in volte `nodp-sweep2`:

| C (righe), ε (colonne) | 16 | 64 | 256 |
|---|---|---|---|
| 0.25 | 11.9 | 10.2 | 6.9 |
| 0.5 | 8.8 | 5.3 | 2.6 |
| 1 | 41.6 (E-A) | 2.6 | 2.9 |

A parità di rumore C più grande costa meno: domina la distorsione del clipping. Tre celle
sotto 3 volte a un seed; il punto operativo per la regola era C = 1, ε = 64. **Su 5 seed non
regge**: cella `rq1-eps64` (commit 8d44ae3, nelle matrici), holdout 0.00712, 4.5 volte la media
del no-DP; per seed, rispetto al no-DP dello stesso seed, da 1.9 a 14.7 volte. Il seed 42 della
griglia era il più favorevole. ε = 32, seed 42: 7.3 volte. Attacchi al caso in tutte le celle,
test per record senza segnale anche a ε = 64 (sezione 3.2). È la terza riga della tabella di
lettura di `ESPERIMENTI.md`: nessun punto operativo utile per il client-level in questo regime,
con tre client.

### 3.1c Controlli del protocollo (segnalazione 48)

Fonte: `logs/ctrl_riproduzione_s42.log` e il JSON di `experiments/_ctrl_common_init/`
(commit `6d31e21`), letti il 2026-09-24, confrontati con `nodp-sweep2`; no-DP, seed 42.
La run ha un'etichetta di cella propria ("no-DP baseline, init comune") e, come
cartella `_*`, resta fuori dalle celle delle matrici.

| | holdout al round 1 | holdout al round 10 | Yeom | LiRA composto | TPR a FPR 1% |
|---|---|---|---|---|---|
| `nodp-sweep2`, seed 42 | 0.0651 | 0.00219 | 0.4988 | 0.5018 | 0.0117 |
| stessi, 5 seed (min-max) | | 0.00070-0.00219 | 0.4975-0.5039 | 0.4961-0.5036 | 0.0086-0.0117 |
| inizializzazione comune, seed 42 | 0.0080 | 0.00108 | 0.4989 | 0.4991 | 0.0099 |

Il codice attuale riproduce `nodp-sweep2` (controllo (a): loss di addestramento uguale
alla sesta cifra nei 10 round). Con l'inizializzazione comune il danno del round 1
sparisce e al round 10 il modello e' migliore, ma dentro la variabilita' fra seed;
gli attacchi restano al caso. Le campagne restano valide, lo scostamento dal protocollo
standard si dichiara. Norme dei delta dal round 3: 0.16-0.29 (caltech, jpl) e
0.51-0.59 (office1), sotto C = 1 (`ESPERIMENTI.md`, punto operativo).

### 3.2 Analisi per record: nessun segnale di appartenenza (E-C)

Fonte: `risultati/worst_case/livello_di_caso.json`, rigenerato il 2026-09-24 con il
controllo sui non membri e il test appaiato (segnalazione 47). 5 seed per cella, 17 561
sessioni membro in almeno 2 seed, 200 permutazioni; test appaiato su 28 129 sessioni
membro in un seed e non membro in un altro.

| cella | segnalati (membri) | attesi | z membri | z non membri | test appaiato | z appaiato |
|---|---|---|---|---|---|---|
| no-DP | 412 | 289.9 | 7.56 | 6.85 | +0.06 ± 0.22 | 0.26 |
| dp-fedavg ε=16 | 326 | 290.4 | 2.12 | 0.22 | +0.07 ± 0.22 | 0.31 |
| dp-fedavg ε=8 | 293 | 291.5 | 0.09 | 3.45 | +0.12 ± 0.22 | 0.56 |
| dp-fedavg ε=4 | 355 | 288.9 | 4.31 | 3.86 | −0.31 ± 0.22 | −1.38 |
| dp-fedavg ε=2 | 337 | 290.3 | 2.88 | 2.71 | −0.30 ± 0.22 | −1.34 |
| dp-fedavg ε=1 | 290 | 289.8 | 0.02 | 0.60 | −0.21 ± 0.22 | −0.94 |
| central ε=1 | 299 | 289.9 | 0.62 | 1.91 | +0.20 ± 0.22 | 0.91 |
| local ε=1 | 291 | 289.8 | 0.07 | 1.53 | −0.29 ± 0.22 | −1.32 |
| dp-fedavg ε=64 | 473 | 290.3 | 11.61 | 10.35 | +0.15 ± 0.22 | 0.69 |
| record-DP σ=1 | 415 | 289.0 | 7.21 | 9.39 | +0.04 ± 0.22 | 0.18 |
| record-DP σ=2 | 455 | 286.0 | 10.66 | 11.64 | −0.11 ± 0.22 | −0.50 |
| no-DP, GroupNorm | 458 | 289.9 | 10.12 | 8.98 | −0.20 ± 0.22 | −0.91 |
| record-DP σ=5 | 431 | 286.2 | 8.72 | 7.73 | −0.06 ± 0.22 | −0.27 |
| record-DP σ=0 (solo clipping) | 435 | 290.2 | 8.64 | 9.56 | +0.35 ± 0.22 | 1.59 |
| FedAvg no-DP, Mac (`rq3-mu0`) | 516 | 290.1 | 14.55 | 12.64 | +0.19 ± 0.22 | 0.86 |
| FedAvg ε=64, Mac | 441 | 290.4 | 9.03 | 9.81 | +0.23 ± 0.22 | 1.04 |
| FedAvg no-DP, Windows | 508 | 289.6 | 15.31 | 9.07 | +0.11 ± 0.22 | 0.51 |
| FedProx no-DP, Windows | 422 | 290.2 | 8.42 | 5.72 | +0.14 ± 0.22 | 0.62 |
| FedAvg ε=64, Windows | 483 | 291.3 | 12.46 | 9.76 | +0.23 ± 0.22 | 1.04 |
| FedProx ε=64, Windows | 461 | 290.2 | 11.44 | 9.98 | +0.15 ± 0.22 | 0.69 |

**Celle aggiunte il 2026-10-01** (le ultime otto righe; stesso script e stessi parametri, 200
permutazioni a seme fisso per cella; ricalcolata la cella no-DP, i valori salvati sono riprodotti
identici). Il test appaiato è ora calcolato su tutte le 24 celle a 5 seed di
`livello_di_caso.json`, comprese le quattro di RQ2 (sezione 3.9): z appaiato fra −1.49 (RQ2
per sito) e +1.59 (solo clipping), nessun segnale di appartenenza per record. Nel FedAvg
senza DP di Windows l'eccesso dei membri sui non membri ha z = 3.43, l'unico sopra 3 fra le 24
celle, mentre il test appaiato della stessa cella è nullo (z = 0.51). Le run a un seed
(screening di C, griglia del punto operativo, prove su mu) e le run canary non hanno il test:
richiede la stessa sessione membro in un seed e non membro in un altro.

Nella cella record-DP le sessioni sono 17 389 e 27 877 invece di 17 561 e 28 129: il dump
per campione contiene i record che LiRA punteggia, e con record-DP sono circa 60 in meno per
seed (segnalazione 58).

Il conteggio sui membri, letto fino al 2026-09-24 come "l'unico segnale su dati
naturali", compare quasi uguale fra i non membri: senza DP 406 segnalati contro 412.
Misura quanto un modello ordina i record in modo coerente fra seed, cosa che la DP
distrugge insieme all'utility; non misura l'appartenenza. Il test appaiato non trova un
effetto dell'appartenenza in nessuna cella. La segnalazione 6 (percentile dei due
script, 411 contro 412) riguarda ora solo il conteggio, che non è più la lettura
primaria.

### 3.3 Lo strumento è validato a Office 1 e a Caltech (5 seed ciascuno)

Fonte: `CanaryPositiveControl.md`, sezione 6, campagna bilanciata dopo la
correzione del 16 settembre. Office 1, 20 template membro e 20 non membro, 30
duplicati, scambio dei ruoli, baseline a inizializzazione casuale, 5 seed, regime
canary.

| metrica, AUC media sui 3 round | braccio A | braccio B |
|---|---|---|
| loss grezza | 0.745 | 0.715 |
| LiRA calibrato | 0.651 | 0.696 |

Trenta round su trenta sopra 0.5 sulla loss grezza, tutte le somme A+B sopra 1,
Δ medio per seed +0.23 con t(4) = 9.9: il segnale è simmetrico allo scambio dei
ruoli, quindi è appartenenza. Tre artefatti intercettati dai controlli prima di
arrivarci: pool diversi per i due gruppi, scoring asimmetrico fra le classi,
effetto dell'assegnazione sotto DP. Caltech e JPL hanno solo run a seed 42 senza
il protocollo bilanciato; il config per ChargePlace Scotland non è mai stato
eseguito. Il paper riporta ancora i numeri precedenti alla correzione
(segnalazione 35).

**Canary su più siti, prova a un seed (2026-09-30, bracci A e B, `experiments/_canary_multisite_s42`
e `_canary_multisite_swap_s42`, commit b3feee4 e 567d32e, che differiscono solo in `docs/`).** Tre
siti, canary solo in Office 1, senza DP. Loss grezza sull'update di Office 1: braccio A 0.79,
0.5775, 0.65, braccio B 0.7925, 0.5025, 0.5475 nei tre round; somme A+B 1.5825, 1.08 e 1.1975,
contro 1.00 delle baseline a init casuale (0.7255 e 0.2745, dal log). Al round 1 il segnale è
appartenenza; dopo l'aggregazione, in cui Office 1 pesa circa il 3.6%, si riduce molto e con un
solo seed non si può dire se resta sopra il rumore. Media sui 3 round 0.6725 e 0.6142, somma 1.29
contro 1.46 del sito singolo. Sul modello globale le somme di Yeom sono 1.0025, 1.0125 e 0.995:
nessun segnale, e i valori sotto 0.5 del braccio A erano l'effetto dei template. Con più client
LiRA e Shadow sui canary non sono interpretabili (segnalazioni 63 e 64): le somme di LiRA ai
round 2 e 3 sono 0.49 e 0.70, sotto 1.

**Pilota canary su A3 (2026-10-06/07, Mac principale, seed 42 e 123, commit ec27181 pulito,
`experiments/_canary_multisite*_A3_s*` e `_canary_multisite*_nodp_s*`).** Stessi tre siti e canary in Office 1, con DP per
client al punto operativo e rumore lato client (`local`, ε = 64, C = 1): l'attaccante vede l'update di Office 1 dopo
taglio e rumore. Somme A+B della loss grezza nei tre round: seed 42 1.0075, 0.975, 1.0275 (senza DP 1.58, 1.08, 1.20);
seed 123 0.9875, 0.9925, 1.1125 (senza DP 1.38, 1.06, 1.05); media sui 3 round 1.00 e 1.03 contro 1.29 e 1.16 senza DP.
Al round 1, dove senza DP il segnale c'è in tutti e due i seed, con DP i due bracci stanno sulle baseline a init casuale.
Sul modello globale nessun segnale, con e senza DP. La DP per client al punto operativo toglie il segnale dei canary sulla
superficie dell'aggregatore; costo in questo modello a 3 round: loss del modello globale circa 3-3.5 volte quella senza
DP al round 3. Con LiRA ridotta la loss grezza senza DP è identica a quella con LiRA completa (stesso modello target).
Aggiornamento 2026-10-07: seed 1234 (Mac principale) con DP 1.005, 0.998, 1.005 e senza DP 1.50, 1.09, 1.02; seed 456
(quarta macchina, solo con DP) 1.015, 0.995, 1.08. Con DP quattro seed su quattro intorno a 1.00 in ogni round, senza DP
il segnale al round 1 c'è in tutti i seed misurati (1.58, 1.38, 1.50). Seed 789 (Windows, 2026-10-08): con DP 0.98, 0.98, 1.02,
senza DP 1.48, 1.08, 1.06. Cinque seed su cinque senza segnale con DP. Seed 456 senza DP (quarta macchina, JSON importati il 2026-10-09): 1.555, 1.04,
1.1025. Limite importante (2026-10-09, seed 42 e 123): anche il solo taglio, senza rumore, toglie il segnale (somme fra 0.975
e 1.02), come ε = 512 ed ε = 64. Per questo attaccante il segnale sparisce perché l'update tagliato lascia il modello
osservato vicino all'inizializzazione, non per il rumore; il taglio da solo non è un meccanismo di privacy (un attaccante
che riscala l'update tagliato potrebbe recuperare il segnale). Il pilota non può essere scritto come "la DP toglie il
segnale grazie al rumore" finché non si prova un attaccante che riscala l'update (`ESPERIMENTI.md`).
Seed 456 e 789 (2026-10-09): solo taglio 1.0025, 1.0525, 1.1425 e 0.9975, 1.015, 1.0425; ε = 512 1.01, 0.99, 1.0475 e
0.9925, 0.9925, 1.0175. Al round 1 il solo taglio toglie il segnale in quattro seed su quattro. Attaccante che riscala
l'update: codice pronto (`--canary-rescale`, Sprint 10zz+170), coda di 24 run sul Mac principale con previsione R1-R4.
Esito (2026-10-10, Mac principale, seed 42, 123, 1234): riscalando l'update di 16 al round 1, con solo taglio il segnale
torna nei seed 42 e 1234 (1.518 e 1.335) ma non nel 123 (0.938), dove la norma dell'update è 21 e non 16; con ε = 64
(1.015, 1.008, 1.047) ed ε = 512 (0.748, 0.935, 1.042) non torna. La loss dei non-membri è minima proprio a s = 16 con
solo taglio, quindi l'attaccante può trovare s senza etichette. Lettura provvisoria: il taglio da solo non protegge da un
attaccante che riscala; il rumore sì. In attesa: seed 456 e 789 (quarta macchina), s fitto fra 12 e 32 (Windows).
Fra macchine i numeri non coincidono cifra per cifra (seed 1234, scarto massimo 0.045 fra Windows e Mac): si confrontano
condizioni solo dentro la stessa macchina (`ESPERIMENTI.md`).
Esito a cinque seed (2026-10-12, griglia fitta sul Mac principale, s fitto su Windows, seed 456 e 789 sulla quarta
macchina): con solo taglio l'attaccante trova senza etichette l's uguale alla norma dell'update e il segnale torna in
cinque seed su cinque (1.16-1.52, senza DP 1.38-1.58); con ε = 64 non torna in quattro seed su cinque (1.00-1.08), nel
seed 456 resta 1.195; con ε = 512 nessun segnale nel verso atteso (0.98-1.12). Il calo sotto 1 visto a ε = 512 nel seed 42
non si ripete sugli altri seed. Frase per il paper: il taglio da solo non protegge da un attaccante che riscala; il rumore
al punto operativo toglie quasi tutto il segnale.

**Canary bilanciato su Caltech, seed 42 (2026-10-01, Mac principale,
`experiments/_canary_balanced_caltech_s42` e `_canary_balanced_caltech_swap_s42`, commit 11c7fdf e
1c0d0c0 con `-dirty` dovuto solo a documenti).** Un client, 561 duplicati per template (la stessa
amplificazione per record di Office 1), senza DP. Loss grezza: braccio A 0.80, 0.7875, 0.6925,
braccio B 0.7725, 0.865, 0.8925; somme A+B 1.5725, 1.6525 e 1.585 contro 1.00 delle baseline a
init casuale (0.4616 e 0.5384): appartenenza in tutti e tre i round. Media sui 3 round 0.76 e
0.8433, somma 1.60 contro 1.46 di Office 1. Con un solo client LiRA lavora su tutte le 20 x 20
coppie e dà somme 1.415, 1.54 e 1.5075. Il braccio A della quarta macchina ha la stessa media
(0.7608 contro 0.7600) con scarti per round fino a 0.08.

**Canary bilanciato su Caltech, 5 seed (2026-10-02).** Seed 42, 456, 789 e 1234 sul Mac principale
(`experiments/_canary_balanced_caltech{,_swap}_s*`), seed 123 su Windows
(`experiments_altre_macchine/_canary_balanced_caltech{,_swap}_s123`, commit 1c0d0c0 pulito); i due
bracci di ogni seed sulla stessa macchina. Commit: 456 e 789 a 316cbf8 pulito; 1234 a 316cbf8-dirty
(braccio B) e ae8b9c9-dirty (braccio A), con modifiche aperte solo a documenti e file di analisi,
codice identico. Loss grezza, media sui 3 round: braccio A 0.770, braccio B 0.808. Trenta round su
trenta sopra 0.5 (minimo 0.6625), tutte le 15 somme A+B per round sopra 1 (da 1.42 a 1.74).
Δ medio per seed (media dei due bracci, come per Office 1): 0.302, 0.267, 0.310, 0.324, 0.242;
media +0.289, sd 0.034, t(4) = 19.3, p < 0.001; test dei segni 5 su 5, p = 0.031. Office 1 dava
+0.23 con t(4) = 9.9, alla stessa amplificazione per record. LiRA sui canary, medie sui 3 round:
0.694 e 0.738, 29 round su 30 sopra 0.5 (sotto: seed 789, braccio B, round 3, 0.437). Δ dei
singoli bracci tutti positivi, da +0.13 a +0.40 (seed 123: baseline 0.4271 e 0.5729 dal log di
Windows, Δ +0.335 e +0.200); come a Office 1, dove la baseline parte molto bassa (seed 123 e
1234) il Δ di quel braccio è più grande. Lo strumento è validato su due siti con un solo client; su più client resta la prova a un seed
qui sopra.

### 3.4 LiRA è degenere

Il floor di varianza è colpito nel 97-99% dei record, il controllo di simmetria
dello scoring su central dà 0.21-0.34 contro uno 0.50 riportato, e il punteggio
non è riproducibile a seed fisso mentre la loss grezza lo è. Per questo la loss
grezza è la metrica primaria e lo scorer resta congelato finché lo sweep non è
chiuso (`ESPERIMENTI.md`, regole).

### 3.5 Record-level DP

Implementato il 16 settembre, usato solo in regime canary a Office 1: a σ = 5 la
loss grezza dei canary scende da 0.68 a 0.53 con il modello ancora addestrabile (loss
sull'holdout del modello rilasciato 0.006, simile alla loss locale). Un seed, nessun
braccio swap. L'ε di 7.15 è corretto sotto l'ipotesi di campionamento di Poisson (3
round, n = 1924: segnalazione 4, rettifica); il training usa shuffle, e il limite valido
con lo shuffle, adiacenza per sostituzione e senza amplificazione, è 343. Su dati
naturali: la prova del 24 settembre sul Mac principale è arrivata in fondo con gli
attacchi saltati (segnalazione 49, JSON in `_prova_recorddp` da non usare); il rilancio
`_prova_recorddp_v2` sul secondo Mac esegue tutti gli attacchi. A σ = 1, seed 42, la loss
sull'holdout è 1.29 volte il no-DP della stessa macchina, gli attacchi sono al caso, l'ε
per record di Poisson è 30.2 (Office 1) e il limite con lo shuffle 1212 (JSON in
`experiments_altre_macchine/_prova_recorddp_v2`, riverificato il 2026-09-26). Il confronto col no-DP mescola rumore e cambio di
normalizzazione (GroupNorm).

**Campagna E-B, σ = 1, completa** (`experiments/rq1-recorddp-nm1`, Mac principale, 5 seed
dal 26 al 27 settembre, zero righe `[ERROR]` nel log; JSON letti il 2026-09-27; cella
"record-DP, nm=1.0" nei fogli `Utility_privacy_limite` e `Worst_case_per_record`, confronto
C16). Commit 0dedd34 (seed 123), 1519d16 con `-dirty` dovuto solo a documenti (seed 456,
segnalazione 57), 943c795 (789, 1234, 42): il codice di training e attacco è lo stesso.

| seed | holdout | volte lo stesso seed no-DP | Yeom, ultimo round | Shadow, media | LiRA composto | TPR a FPR 1% | ε Poisson, Office 1 |
|---|---|---|---|---|---|---|---|
| 42 | 0.00240 | 1.10 | 0.498 | 0.497 | 0.502 | 0.0104 | 30.2 |
| 123 | 0.00239 | 1.46 | 0.503 | 0.503 | 0.498 | 0.0100 | 30.4 |
| 456 | 0.00130 | 1.86 | 0.501 | 0.503 | 0.503 | 0.0081 | 31.0 |
| 789 | 0.00233 | 1.48 | 0.503 | 0.504 | 0.497 | 0.0085 | 30.4 |
| 1234 | 0.00247 | 1.35 | 0.501 | 0.504 | 0.503 | 0.0110 | 30.4 |

Loss sull'holdout media 0.00218, 1.4 volte la media di `nodp-sweep2` (1.43 in media
geometrica dei rapporti per seed, tutti sopra 1): sotto la soglia di 3 volte, dove nessuna
cella client-level arriva (la migliore, ε = 64, vale 4.5). Attacchi al caso, LiRA composto
equivalente a 0.5 col TOST (`check_significance.py`), test appaiato per record +0.04 ± 0.22,
z = 0.18 (sezione 3.2). ε per record di Poisson fra 30.2 e 31.0 a Office 1, 4.8 a JPL e 5.0 a
Caltech; con lo shuffle il limite valido è 1212, cioè nessuna garanzia formale utile.

**σ = 2 e riferimento no-DP con GroupNorm, completi il 2026-09-28** (`experiments/rq1-recorddp-nm2`
e `rq1-nodp-groupnorm`, Mac principale, 5 seed ciascuna, zero righe `[ERROR]`, commit puliti
4c0d0ae, fe22124, 42b4561; JSON letti il 2026-09-28; celle "record-DP, nm=2.0" e "no-DP
baseline, norm=group", confronti C17 e C15). Loss sull'holdout del modello rilasciato:

| seed | no-DP, BatchNorm (`nodp-sweep2`) | no-DP, GroupNorm | σ = 1 | σ = 2 | σ = 1 / GroupNorm | σ = 2 / GroupNorm |
|---|---|---|---|---|---|---|
| 42 | 0.00219 | 0.00109 | 0.00240 | 0.00244 | 2.19 | 2.23 |
| 123 | 0.00164 | 0.00153 | 0.00239 | 0.00293 | 1.56 | 1.92 |
| 456 | 0.00070 | 0.00092 | 0.00130 | 0.00137 | 1.41 | 1.49 |
| 789 | 0.00157 | 0.00086 | 0.00233 | 0.00264 | 2.71 | 3.07 |
| 1234 | 0.00184 | 0.00141 | 0.00247 | 0.00297 | 1.75 | 2.10 |
| media (rapporti fra le medie) | 0.00158 | 0.00116 | 0.00218 | 0.00247 | 1.87 | 2.12 |

Il passaggio a GroupNorm, senza DP, non costa: 0.73 volte BatchNorm sulle medie, non
significativo a 5 seed (t = −1.51 sul logaritmo). Il costo della DP per record va quindi letto
sul riferimento GroupNorm ed è più alto dell'1.4 letto su `nodp-sweep2`: 1.87 volte a σ = 1 e
2.10 a σ = 2 in media geometrica dei rapporti per seed, 5 seed su 5 sopra 1 (t = 5.30 e 6.37
sul logaritmo), ancora sotto la soglia di 3 volte con entrambi i riferimenti. Da σ = 1 a
σ = 2 il costo cresce di 1.12 volte, mentre l'ε per record di Poisson scende da circa 30 a
circa 9.7 a Office 1 (1.8 a JPL e Caltech; con lo shuffle 355). Attacchi al caso in tutte e
tre le celle, test per record senza segnale (sezione 3.2). Che il costo quasi non cambi col
rumore faceva pensare che venisse soprattutto dal clipping per esempio (C = 1); la cella con
clipping e σ = 0 lo smentisce (paragrafo «Solo clipping» sotto). Riserve: gli shadow di LiRA non usano
DP-SGD (segnalazione 51); il confronto con il client-level è fra unità protette diverse a
costo comparabile, non allo stesso ε. Lettura per RQ1: a σ = 1 e 2 la DP per record ha un
costo accettabile e la DP per client no; in nessuna delle due il regime naturale offre un
segnale che la DP possa ridurre.

**σ = 5 e solo clipping, 2026-09-29.** σ = 5 è completo (quarta macchina, 5 seed, commit
64757fd pulito, `experiments_altre_macchine/rq1-recorddp-nm5`, JSON letti il 2026-09-29):
2.32 volte il riferimento GroupNorm in media geometrica dei rapporti per seed (da 1.82 a
3.27; t = 7.74 sul logaritmo) e 1.10 volte σ = 2 (da 1.01 a 1.22; t = 2.44, non
significativo a 4 gdl). L'ε per record di Poisson scende a 3.1 (massimo sui client, Office 1;
0.6 a Caltech e JPL), il limite valido con lo shuffle a 81. Yeom al caso (0.497-0.503). Il
seed 789 supera la soglia di 3 volte già a σ = 2 (3.07) e a σ = 5 (3.27): ha il riferimento
GroupNorm più basso dei cinque. Lettura: da σ = 1 a σ = 5 l'ε per record di Poisson scende
di circa dieci volte e il costo passa da 1.87 a 2.32 volte; la media resta sotto la soglia,
non tutti i seed. Il controllo a 2 round sulla quarta macchina riproduce il Mac principale
(`PROVENIENZA.txt`).

**Solo clipping (σ = 0), 2026-09-30.** Completo (quarta macchina, 5 seed, commit 8dceb39
pulito, `config/experiment_rq1_recorddp_nm0.yaml`, `experiments_altre_macchine/rq1-recorddp-nm0`,
JSON letti il 2026-09-30): 1.04 volte il riferimento GroupNorm in media geometrica dei rapporti
per seed (da 0.72 a 1.40; 3 seed su 5 sopra 1; t = 0.35 sul logaritmo, nessuna differenza).
Sugli stessi seed, rispetto al solo clipping, il rumore costa 1.80 volte a σ = 1 (da 1.45 a
2.18, t = 8.60, 5 seed su 5), 2.02 a σ = 2 e 2.23 a σ = 5. Il costo della DP per record viene
quindi dal rumore e non dal clipping per esempio a C = 1: l'ipotesi formulata con σ = 1 e 2
non regge. Il costo si concentra nel passaggio da σ = 0 a σ = 1 e poi cresce poco (1.24 volte da
σ = 1 a σ = 5). ε per record `None`: la cella è diagnostica, senza garanzia. Attacchi al caso
(Yeom, Shadow e LiRA fra 0.496 e 0.509 al round 10); test per record non eseguito su questa
cella.

Il seed 42 riproduce la prova `_prova_recorddp_v2` del secondo Mac (commit 2c4a0f0): loss di
addestramento per round e holdout coincidono (0.00239868088 contro 0.00239868097), Yeom alla
settima cifra; LiRA composto no (0.5025 contro 0.5043), come atteso (sezione 3.4). Con
GroupNorm il risultato non dipende dalla macchina (segnalazione 55).

### 3.6 Infrastruttura

Simulazione e deployment NVFlare a 5 container convergono sullo stesso nullo a
ε = 1. Venticinque dump NVFlare a ε ≤ 1 attendono rianalisi, rinviata perché nel
regime con utility distrutta; la rianalisi precedente era invalida per il seed
dello split (segnalazione 1). Privacy Auditor: contatore di budget, telemetria
identica fra cella con memorizzazione e senza. ByzantineDetector: mai eseguito
end-to-end. Nessuno dei due entra nel paper come contributo.

### 3.7 Unità protetta e persona

Identificativo utente presente nel 72.6% delle sessioni: JPL 93.5%, Caltech
52.1%, Office 1 34.8% (`risultati/decision_matrix_ACN_membership_DP.xlsx`,
misura del 2026-09-21; il paper li ha invertiti, segnalazione 36). 1 028 utenti,
mediana 14 sessioni, 8.5% multi-sito. Né record-level né client-level limitano il
contributo di una persona.

### 3.8 La scheda scientifica

I campi della sezione 5 della guida sono compilati nella stessa scheda
`decision_matrix_ACN_membership_DP.xlsx`; le decisioni di `ESPERIMENTI.md` sezione 0
sono state congelate il 2026-09-22, prima di leggere E-A. L'ambiguità sul riferimento
di costo (campagna o `nodp-sweep2`) non decide più nulla: con la loss del modello
rilasciato ε = 16 sta a 47 volte il primo e 60 volte il secondo (segnalazione 46). Due
punti del §0 sono cambiati il 2026-09-24 e sono documentati lì come deviazioni: il
costo, per un errore di implementazione (la definizione era giusta), e la metrica
primaria per record, che va sostituita dal test appaiato (segnalazione 47): la seconda
è decisa col supervisore il 2026-10-01 (test appaiato primario). Il braccio B1 della Fase B, clipping senza rumore, è
eseguito per il record-level (σ = 0, sezione 3.5); per il client-level manca il flag
(segnalazione 38).

### 3.9 RQ2: partizione IID contro per sito (E-D)

Fonte: foglio `RQ2_partizione` di `risultati/Matrice_sintesi.xlsx` e
`risultati/worst_case/livello_di_caso.json`, dai JSON di
`experiments_altre_macchine/rq2-per_site` e `rq2-iid` (secondo Mac, commit 3815da7),
letti il 2026-09-24. Senza DP, 5 seed per braccio appaiati per seed; i bracci differiscono
solo per `partition.strategy`, con le stesse numerosità per client.

| metrica | per sito | IID | IID − per sito, appaiata (± sd) | t, 4 gdl |
|---|---|---|---|---|
| loss sull'holdout del modello rilasciato | 0.00162 | 0.00133 | −0.00029 ± 0.00051 | −1.26 |
| divario holdout − membri | 0.000278 | 0.000274 | −0.000004 ± 0.000079 | −0.12 |
| Yeom, ultimo round | 0.5001 | 0.5011 | +0.0010 ± 0.0041 | 0.57 |
| Shadow, media sui round | 0.5014 | 0.5012 | −0.0001 ± 0.0015 | −0.19 |
| LiRA composto | 0.4982 | 0.4993 | +0.0011 ± 0.0044 | 0.57 |
| TPR a FPR 1% (LiRA composto) | 0.0091 | 0.0089 | −0.0002 ± 0.0020 | −0.21 |

Test appaiato per record: per sito −0.33 ± 0.22 (z = −1.49), IID −0.03 ± 0.22
(z = −0.15), nessun segnale in nessuno dei due. Senza DP, passare dalla partizione per
sito a una IID con le stesse numerosità non cambia il successo degli attacchi valutati,
al caso in entrambi, né il divario fra membri e non membri. Il modello IID ha la loss
sull'holdout più bassa in 4 seed su 5, in media del 18%, non significativa a 5 seed. Nel
regime naturale RQ2 non misura quindi un effetto dell'eterogeneità sulla membership,
perché non c'è segnale da modulare; l'effetto sul costo della DP si misura col braccio
con DP, sotto. Mancano le statistiche delle feature per client nelle due partizioni, che
la guida chiede per mostrare che il fattore è cambiato.

**Con DP.** dp-fedavg, ε = 64 per round, C = 1: il punto operativo scelto con la regola
della griglia prima di guardare RQ2 con DP. Stessi foglio e file, dai JSON di
`experiments_altre_macchine/rq2-per_site-eps64` e `rq2-iid-eps64` (secondo Mac, commit
3d4d318, zero righe `[ERROR]` nel log), letti il 2026-09-26.

| metrica | per sito | IID | IID − per sito, appaiata (± sd) | t, 4 gdl |
|---|---|---|---|---|
| loss sull'holdout del modello rilasciato | 0.00633 | 0.00936 | +0.0030 ± 0.0066 | 1.03 |
| divario holdout − membri | 0.000299 | 0.000311 | +0.000011 ± 0.000050 | 0.51 |
| Yeom, ultimo round | 0.4990 | 0.5014 | +0.0023 ± 0.0048 | 1.08 |
| Shadow, media sui round | 0.5014 | 0.5010 | −0.0003 ± 0.0012 | −0.64 |
| LiRA composto | 0.5015 | 0.4997 | −0.0018 ± 0.0054 | −0.74 |
| TPR a FPR 1% (LiRA composto) | 0.0109 | 0.0082 | −0.0027 ± 0.0015 | −3.88 |

Costo della DP per partizione, come rapporto fra la loss sull'holdout con DP e senza DP
allo stesso seed e sulla stessa macchina: per sito da 2.1 a 8.6, media geometrica 3.9;
IID da 3.2 a 15.0, media geometrica 6.1. Il rapporto è più alto con IID in 4 seed su 5,
di 1.57 volte in media geometrica, ma l'interazione partizione × DP, sul logaritmo del
rapporto, non è significativa a 5 seed (+0.45 ± 0.45, t = 2.26 contro 2.776). Test
appaiato per record: per sito +0.18 ± 0.22 (z = 0.82), IID +0.19 ± 0.22 (z = 0.86),
nessun segnale. Sul rapporto delle medie anche questa macchina supera la soglia di costo
al punto operativo: 3.9 volte per sito e 7.0 volte IID, coerente con la sezione 3.1b.

Lettura. Con e senza DP la partizione non cambia il successo degli attacchi valutati,
che resta al caso, né il divario fra membri e non membri; la DP tende a costare di più
con la partizione IID, che senza DP partiva da una loss più bassa, ma a 5 seed la
differenza non è distinguibile dalla variabilità fra seed. L'unico test del foglio sopra
la soglia è la TPR del LiRA composto con DP (p = 0.018, per sito sopra il caso e IID
sotto in tutti e 5 i seed): è una metrica secondaria con calibrazione degenere (3.4), i
test appaiati del foglio sono 18 e a α = 0.05 senza correzione se ne attende circa uno
per caso, quindi non lo leggiamo come effetto della partizione. RQ2 risponde così sul
costo del modello, non su un beneficio di privacy, per la stessa ragione di RQ1: nel
regime naturale non c'è un segnale da ridurre.

Il braccio per sito ha lo stesso config di `nodp-sweep2` ma gira sull'altra macchina, e
la loss sull'holdout differisce per seed fino al 23% (segnalazione 55, confermata: dipende
dalla macchina, non dal codice): il confronto di RQ2
usa perciò i due bracci della stessa macchina, non `nodp-sweep2`.

### 3.10 RQ3: FedAvg contro FedProx (E-E)

Fonte: JSON di `experiments_altre_macchine/rq3-mu0` e `rq3-mu0.01` (Windows, commit 478d471,
i due bracci in parallelo), letti il 2026-09-27; 5 seed per braccio appaiati per seed, zero
righe `[ERROR]`, round 1 identico nei due bracci per ogni seed. Foglio `RQ3_algoritmo` di
`risultati/Matrice_sintesi.xlsx` (con la curva della loss sull'holdout per round) e confronto
C-RQ3 di `matrice_confronti.xlsx`, dal 2026-09-27.

| metrica | FedProx mu = 0.01 | FedAvg mu = 0 | FedAvg − FedProx, appaiata (± sd) | t, 4 gdl |
|---|---|---|---|---|
| loss sull'holdout del modello rilasciato | 0.00156 | 0.00040 | −0.00117 ± 0.00039 | −6.63 |
| divario holdout − membri | 0.000265 | 0.000213 | −0.000052 ± 0.000072 | −1.64 |
| Yeom, ultimo round | 0.5000 | 0.5011 | +0.0011 ± 0.0025 | 1.01 |
| Shadow, media sui round | 0.5014 | 0.4995 | −0.0019 ± 0.0035 | −1.22 |
| LiRA composto | 0.5015 | 0.5008 | −0.0007 ± 0.0039 | −0.39 |
| TPR a FPR 1% (LiRA composto) | 0.0103 | 0.0105 | +0.0002 ± 0.0014 | 0.34 |

Con FedAvg il modello rilasciato ha la loss sull'holdout 3.9 volte più bassa in media
geometrica (da 3.7 a 4.5 per seed, 5 su 5); attacchi al caso in entrambi, divario membri −
non membri non diverso. Le curve per round: FedAvg si ferma a 0.0004-0.0005 dal round 4-6,
FedProx scende ancora al round 10 in tutti i seed (seed 42: 0.0041 al round 2, 0.0018 al
round 10). Con l'inizializzazione comune (sezione 3.1c, seed 42, Mac principale) FedProx
arriva a 0.0011: metà del divario viene dal round 1 (segnalazione 48), il resto no, a meno
dell'effetto macchina (fino al 23%, segnalazione 55). Nei round 3-10 di FedProx gli update
di Caltech e JPL hanno norma 0.16-0.29 (controllo (a)) e il termine prossimale a fine round
vale il 5-15% della loss locale: non domina la loss, limita lo spostamento per round.
Ipotesi, da verificare con le prove su mu (`ESPERIMENTI.md` E-E): con mu = 0.01 dieci round
non bastano a FedProx. La loss di addestramento locale registrata include il termine
prossimale e non è confrontabile fra i due bracci (segnalazione 59). Tutta la campagna usa
FedProx mu = 0.01 come base: se cambiarla è una decisione del supervisore.

**Mac principale, senza DP, 2026-09-29** (JSON di `experiments/rq3-mu0`, commit 64757fd,
contro `nodp-sweep2`; non ancora nelle matrici). FedProx ha la loss sull'holdout 3.69 volte
quella di FedAvg in media geometrica (da 2.74 a 4.31, 5 seed su 5), coerente con Windows
(3.90). Il round 10 della run a 30 round (seed 42, commit 64757fd) riproduce `nodp-sweep2`
(0.002186): il confronto regge fra i due commit. Attacchi al caso.

**Prove su mu** (seed 42, Mac principale, loss sull'holdout al round 10): mu = 0: 0.000508;
0.001: 0.000536; 0.01: 0.002186; 0.1: 0.003791. Il costo cresce con mu e il salto sta fra
0.001 e 0.01. Con mu = 0.001 gli update hanno norma 0.23-0.52 al round 10 (FedAvg 2.2-6.8) e
la qualità di FedAvg: la dimensione del passo da sola non spiega il costo. A 30 round FedProx
mu = 0.01 scende a 0.00127 al round 20 e a 0.00094 al round 30, ancora in discesa: 1.85 volte
FedAvg a 10 round. FedProx è soprattutto più lento; se raggiunga il livello di FedAvg non è
stabilito.

**Con DP, Mac principale** (ε = 64, C = 1, dp-fedavg; `experiments/rq3-mu0-eps64`, commit
64757fd e 8dceb39, contro `rq1-eps64`, commit 8d44ae3; fra 8d44ae3 e 8dceb39 `src/` e
`run_experiments.py` non cambiano). FedProx ha la loss sull'holdout 0.53 volte quella di
FedAvg in media geometrica (da 0.28 a 1.62; 4 seed su 5 sotto 1; t = −1.93 sul logaritmo,
non significativo a 4 gdl). La DP costa a FedAvg 30.4 volte (da 20 a 42) e a FedProx 4.4
volte (da 1.9 a 14.7). Norme degli update: FedAvg fra 2.5 e 8.5 in tutti i round, quindi
tagliate da C = 1 a ogni round; FedProx fra 0.2 e 0.5 dal round 5, sotto C (tagliate solo al
round 1, quando il termine prossimale non è attivo). Yeom al caso in tutte le celle.

**Screening di C per FedAvg con DP** (seed 42, ε = 64, `experiments/_rq3_mu0_eps64_C{2,4,8}_s42`,
completo il 2026-09-29; lanciate a 7a21bb0, commit 98589f1 nei JSON, fra i due cambiano solo
documenti). C = 1 era stato scelto sulla griglia di FedProx; non è quella scelta a penalizzare
FedAvg. Loss sull'holdout al round 10: C = 1: 0.01298; C = 2: 0.01308 (1.01 volte C = 1);
C = 4: 0.04888 (3.8); C = 8: 0.24306 (18.7). A ε fisso il rumore cresce con C (σ = C √(2 ln(1.25/δ)) / ε dal
config: 0.151, 0.303, 0.606 per C = 2, 4, 8) e gli update di Caltech e JPL restano sopra C anche a C = 8
(fino a 15.4): alzare C non migliora il rapporto fra update tagliato e rumore, e sposta di più
il modello a ogni round. Con C = 2 la loss scende più in fretta nei round 2-9 (0.0081 al round 8
contro 0.0304) e al round 10 torna allo stesso livello; con C = 4 e 8 l'andamento è instabile
(C = 8: 0.553 al round 3). Per la regola scritta prima dei risultati (`ESPERIMENTI.md` E-E)
nessun C batte C = 1: niente 5 seed a un altro C, niente FedProx allo stesso C. Un seed solo: a
C = 1 la loss varia fra seed di un fattore 2.9 (0.00634-0.01816), quindi il 3.8 di C = 4 è
appena sopra quella variabilità, il 18.7 di C = 8 ben oltre. Attacchi al caso (Yeom, Shadow e
LiRA fra 0.49 e 0.51 in tutti i round).

**Con DP, Windows** (FedAvg, `experiments_altre_macchine/rq3-mu0-eps64`, 5 seed, fe22124
con `-dirty` da file non tracciati e codice identico, `PROVENIENZA.txt`). La DP costa a
FedAvg 52 volte (da 34 a 83) sul riferimento Windows senza DP a 478d471 (percorso di
training identico: loss del round 1 uguale a 12 cifre). Con stesso seed e stesso config la
loss con DP su Windows è 1.61 volte quella del Mac (da 1.00 a 2.45), contro 0.94 senza DP:
sotto DP l'effetto macchina è più grande (segnalazione 60), quindi i confronti con DP si
fanno solo sulla stessa macchina.

**FedProx con DP su Windows, 2026-10-01** (`experiments_altre_macchine/rq3-mu0.01-eps64`, 5
seed, commit fe22124 pulito, stesso codice e stessa macchina del braccio FedAvg). FedProx ha la
loss sull'holdout 0.34 volte quella di FedAvg in media geometrica (da 0.18 a 0.75, 5 seed su 5
sotto 1; t = −4.62 sul logaritmo, 4 gdl, p = 0.0099): stesso verso del Mac principale (0.53, 4
seed su 5, non significativo), qui netto. La DP costa a FedProx 4.6 volte (da 2.2 a 17.0, contro
`rq3-mu0.01` a 478d471), come sul Mac principale (4.4), e a FedAvg 52. Attacchi al caso in
entrambi i bracci: Yeom all'ultimo round 0.501 (FedAvg) e 0.499 (FedProx), LiRA composto 0.501
in entrambi, TPR a FPR 1% 0.010.

**Analisi per round con DP (2026-10-02, dai JSON: `delta_norm_per_client`, `n_train_per_client`,
loss sull'holdout per round; Mac principale e Windows, 5 seed).** Al round 1 i due algoritmi sono identici
(norme 4.2-9.4, fattore di taglio circa 0.12, loss 0.157). Gli aggiornamenti di FedProx scendono a 1.7-2.0 al
round 2, 0.9-1.3 al round 3 e sotto C = 1 dal round 4; quelli di FedAvg restano fra 2.5 e 11 in tutti i
round. Al round 2 FedProx trattiene il 55% dell'aggiornamento e FedAvg il 13%, e le loss con DP si separano
già lì (0.059 contro 0.133); dopo il round 1 il modello globale è lo stesso per i due algoritmi, quindi il
confronto dal round 2 non dipende dall'inizializzazione. FedAvg con DP scende ancora al round 10 (0.026 al
round 9, 0.013 al round 10). Messa contro la quota cumulata di soluzione locale trattenuta,
Σ_t min(1, C/‖Δ_t‖), la loss con DP dei due algoritmi cade su una curva sola su tutte e due le macchine
(FedAvg al round 10, 1.48, 0.013; FedProx al round 3, 1.58, 0.017). È un'associazione: la causa si verifica
col solo taglio (`ESPERIMENTI.md`, linea per il paper DSN). A seed 42 FedAvg ha 0.01298 a C = 1 e 0.01308 a
C = 2. Figure di lavoro in `Claude outputs/` (`fig_rq3_v2_*`).

**Clipping su tutti i JSON con le norme (2026-10-02; per il paper, a supporto di RQ3).** Stessa analisi
su tutte le run con DP per client che hanno `delta_norm_per_client` nel JSON: 18 impostazioni, 6 a 5 seed (C = 1,
ε = 64: FedAvg e FedProx su Mac principale e Windows, FedProx con partizione per sito e IID sul secondo Mac), le
altre al solo seed 42 (FedProx con C = 0.25, 0.5, 1 e ε da 16 a 256; FedAvg con C = 2, 4, 8 a ε = 64). La griglia
di ε da 2 a 16 a 5 seed non ha le norme nel JSON né nel log (il campo `sensitivity` dell'IDS è un rapporto fra
le dimensioni dei dati, non una norma) ed è esclusa. (1) A parità di rumore aggregato
(C/ε = 1/64) le sette impostazioni seguono la stessa curva fino a una quota cumulata di circa 1.5: loss sull'holdout
0.057-0.072 a quota 0.65 e 0.014-0.018 a quota 1.5. (2) FedProx con C = 0.25 a ε = 16 (stesso C/ε, un seed; rivisto a 5
seed nel paragrafo successivo)
trattiene quanto FedAvg (13% contro 12% al round 2) e sta sulla curva di FedAvg (0.067 contro 0.067 a quota 0.65,
0.014 contro 0.014 a 1.5) con passi applicati quattro volte più piccoli: conta la quota di soluzione locale
trattenuta, non il motivo del taglio né l'algoritmo. (3) Sotto quota circa 1 il rumore conta poco (FedAvg seed 42 a
quota 1.0: 0.033 a C = 1, 0.044 a C = 2, 0.030 a C = 4; solo C = 8 sta sopra, 0.139); oltre quota 2 la curva si
appiattisce su un pavimento che sale col rumore (mediane per round a quota ≥ 2: 0.004-0.011 con meno rumore del punto
operativo, a un seed; 0.010 al punto operativo, 0.018 a rumore doppio, 0.058 a quadruplo, 0.24 a otto volte). Lettura in due
regimi: fino a quota circa 1.5 la loss segue la quota trattenuta, da quota 2 la fissa il rumore. FedAvg al punto
operativo è ancora nel primo regime al round 10 (quota 1.36 al seed 42, 1.48 in media), FedProx è nel secondo dal round 4-5. C = 1 e
C = 2 coincidono al round 10 per motivi diversi: C = 2 arriva prima a quota 1.3 (stessa loss, 0.017 contro 0.018) e
poi resta sul suo pavimento più alto, C = 1 sta ancora scendendo. Limiti: un solo seed fuori da C = 1, ε = 64; a un
seed i confronti fra ε muovono la stessa direzione di rumore scalata e non si leggono (a 5 seed più rumore costa di
più: ε = 16 contro 64 al round 3, 0.13 contro 0.016); associazione, la causa resta all'esperimento con solo taglio.
Fonte: `risultati/clipping/` (`per_round.csv`, `per_run.csv`), rigenerabile con `scripts/analisi_clipping_json.py`;
figura di lavoro in `Claude outputs/clipping_tutti_json/`. Uso nel paper: il punto (1) è già a 5 seed per sei
impostazioni e si può usare così; il punto (2), che è la frase più forte (il vantaggio di FedProx sta nel fatto
che i suoi aggiornamenti passano il taglio, non nell'algoritmo in sé), e il pavimento del punto (3) aspettano i
seed mancanti (`ESPERIMENTI.md`, norme della griglia di ε e seed di FedProx a C = 0.25). Previsione per la prova
a 30 round in `ESPERIMENTI.md` (linea per il paper DSN).

**Aggiornamento a 5 seed (2026-10-05).** Rifatte sul Mac principale, con LiRA ridotta, le run di `rq1-eps16` e
`rq1-eps8` (5 seed ciascuna) e i 4 seed mancanti di FedProx a C = 0.25, ε = 16 (`experiments/_rq1_eps16_norme`,
`_rq1_eps8_norme`, `_op_C0.25_eps16_seed`; commit 0b1d8a9 e ff23a1c puliti). Per ε = 16 e 8 la loss globale e la
loss sull'holdout per round coincidono in ogni cifra con i JSON originali: le norme sono quelle delle run originali.
Punto (2) rivisto. FedProx tagliato quanto FedAvg perde il vantaggio di FedProx round per round: al round 2 vale
0.153 (FedAvg 0.132, FedProx a C = 1 0.056), al round 10 0.0142 (FedAvg 0.0122, FedProx a C = 1 0.0065), cioè 9.6
volte FedProx senza DP contro 4.4. Il vantaggio di FedProx viene dal fatto che i suoi aggiornamenti passano il
taglio, non dall'algoritmo in sé: questa frase regge a 5 seed. Non regge invece la coincidenza esatta con la curva
comune vista a un seed: a 5 seed FedProx a C = 0.25 sta sopra (0.076 contro 0.057-0.072 a quota 0.65, 0.029 contro
0.014-0.018 a quota 1.5). Con lo stesso rumore e passi quattro volte più piccoli conta anche il passo applicato
rispetto al rumore: la quota trattenuta spiega la differenza fra i due algoritmi, ma a rumore fisso un C più piccolo
costa di più. Punto (3) confermato a 5 seed per FedProx a C = 1. A ε = 16 (rumore quadruplo) gli aggiornamenti
scendono sotto C dal round 4 (norma mediana circa 0.75, contro circa 0.27 a ε = 64) e la quota cumulata arriva a
8.7 come a ε = 64: il costo in più non passa dal taglio, è il pavimento del rumore (mediana per round a quota ≥ 2
0.080, contro 0.010). A ε = 8 (otto volte) gli aggiornamenti restano intorno a C (quota per round 0.72-0.93) e la
loss resta fra 0.15 e 0.30 in tutti i round (mediana 0.23): il modello quasi non impara, il rumore domina e passa
anche un po' dal taglio. Il pavimento dipende dal livello di rumore e non da come lo si alza: a rumore quadruplo
0.080 (FedProx, ε = 16, 5 seed) e 0.058 (FedAvg, C = 4, un seed), a otto volte 0.23 (FedProx, ε = 8) e 0.24 (FedAvg,
C = 8). Uso nel paper: il punto (1) così com'è; il punto (2) nella forma rivista (FedProx tagliato come FedAvg perde
il vantaggio, 5 seed), non nella forma "conta solo la quota"; il punto (3) a 5 seed per FedProx. Resta
un'associazione: la causa si prova con le run a solo taglio e solo rumore.

**Prova a 30 round con DP, seed 42 (2026-10-05; quarta macchina, `experiments_altre_macchine/_rq3_mu0_eps64_r30_s42`
e `_rq3_mu0.01_eps64_r30_s42`, commit 2341da6 pulito, LiRA ridotta).** FedAvg al round 30 ha una quota cumulata di
5.38 (prevista fra 5 e 6) e ai round 26-30 vale 0.0102 (media geometrica), dentro la fascia di FedProx allo stesso
rumore (0.006-0.011): a 10 round gran parte del costo di FedAvg è velocità. FedProx ai round 26-30 vale 0.0142, più di
FedAvg, e la sua loss sale piano dopo il round 10 (0.0075 ai round 6-10, 0.0087 agli 11-15, 0.0129 ai 21-25), con
aggiornamenti fra 0.19 e 0.29 contro un rumore aggregato di norma circa 1.25 per round: è il segnale che il pavimento di
FedProx è di rumore e che con abbastanza round FedAvg lo supera. Un seed solo, con oscillazioni per round fino a un
fattore 4 (FedProx 0.046 al round 25): l'inversione è un'ipotesi da verificare a 5 seed (`ESPERIMENTI.md`). Per il paper
RQ3 va detta a budget fisso di round: a 10 round FedProx costa circa un terzo di FedAvg; con più round il vantaggio si
riduce e può invertirsi. La quarta macchina non riproduce il Mac principale cifra per cifra (round 1: norme 8.8969,
8.4754, 4.6364 contro 8.8392, 8.4689, 4.6367; round 10 FedAvg 0.0198 contro 0.0130): vale il confronto fra bracci
della stessa macchina, non lo scambio di numeri fra macchine.

**30 round con DP, 5 seed sul Mac principale (2026-10-05 e 06; `experiments/_rq3_mu0_eps64_r30`,
`_rq3_mu0.01_eps64_r30`, commit b4285e9 pulito per quattro seed e c88a4c5 pulito per il seed 42, LiRA ridotta).** I
round 1-10 coincidono in ogni cifra con le run a 10 round. Ai round 26-30 (media geometrica) FedAvg vale 0.0050-0.0097
e FedProx 0.0051-0.0160: FedAvg raggiunge il pavimento di FedProx in tutti i seed e sta sotto in 5 su 5, ma di poco:
rapporto 0.76 in media geometrica, t(4) = -2.15, p = 0.098. Medie per blocchi di 5 round: dal blocco 11-15 i due
algoritmi stanno sullo stesso livello (circa 0.007-0.011); in media FedProx non sale con i round (sale nei seed 42 e
123, scende negli altri tre). Per il paper RQ3 si scrive a budget fisso: a 10 round FedProx costa circa la metà di FedAvg (0.0065
contro 0.0122), perché il taglio rallenta FedAvg; con abbastanza round i due arrivano allo stesso pavimento di rumore.

**Solo taglio e solo rumore (2026-10-06; Mac principale, 5 seed, commit c88a4c5 pulito; `experiments/_rq3_mu0_eps64_*`
e `_rq3_mu0.01_eps64_*` con `clip-only`/`noise-only`).** Prova causale del meccanismo, con previsione scritta prima
(`ESPERIMENTI.md`). Loss al round 10, media geometrica: FedAvg 0.0122 con la DP completa, 0.0119 con il solo taglio,
0.0044 con il solo rumore, 0.0004 senza DP; FedProx 0.0065, 0.0022, 0.0079, 0.0015. Il costo di FedAvg è il taglio
(il solo taglio riproduce la DP completa, rapporto 0.97); quello di FedProx è il rumore (solo rumore 1.21 volte la DP
completa, solo taglio 1.5 volte il senza DP). Allo stesso rumore e senza taglio FedAvg arriva più in basso di FedProx
(0.0044 contro 0.0079, p = 0.07): i suoi aggiornamenti grandi correggono meglio il rumore, coerente con i 30 round. Al
variare di C per FedAvg il costo del solo taglio scende (al seed 42 0.068 a C = 0.5, 0.0004 a C = 4) e quello del solo
rumore sale (0.001 a C = 0.5, 0.35 a C = 8); si incrociano fra C = 1 e 2 in tutti e tre i seed provati (42, 123, 456): a
C = 2 la DP completa coincide con il solo rumore (rapporti 0.94-1.15), a C = 1 sta vicino al solo taglio. A C = 2 la DP
completa è 1.8 volte quella a C = 1 ai seed 123 e 456 (uguale solo al seed 42): dei due, C = 1 è il punto migliore. Frase causale per RQ3: sotto DP per client, al punto operativo, il costo di FedAvg viene dal taglio e quello di
FedProx dal rumore. Replica su Windows (2026-10-06, 5 seed, DP completa rifatta e identica alle run vecchie): FedProx
uguale al Mac (solo rumore 1.15 volte la DP completa, solo taglio 0.33); per FedAvg il solo taglio resta molto sopra il
solo rumore (5.8 volte sui round 6-10, sul Mac 6.5) ma vale 0.42 volte la DP completa al round 10 e 0.66 sui round 6-10
(sul Mac 0.97 e 0.91): su due macchine la frase va scritta "il costo di FedAvg viene soprattutto dal taglio". Placement del rumore: con `central` (quarta macchina, 5 seed) FedProx vale quanto con il rumore lato
client (rapporto 1.00), quindi la frase vale per la DP per client in generale. Prova su μ (quarta macchina, 5
seed, commit c88a4c5 pulito, solo Yeom): con DP la loss al round 10 vale 0.0158, 0.0086, 0.0059 e 0.0103 per μ = 0, 0.001,
0.01 e 0.1 (quota cumulata 1.5, 8.0, 8.6, 9.1); senza DP μ = 0.001 costa poco rispetto a μ = 0 (rapporto 1.16, entro 1.5
in 5 seed su 5). Un μ minimo toglie il taglio, ma con DP resta sopra μ = 0.01 (in 5 seed su 5 sulla media dei round
6-10) e non sta sotto μ = 0 in tutti i seed al round 10; μ = 0.1 rallenta. Con la DP per client il μ migliore a questo
budget è intorno a 0.01, non il più piccolo possibile. Sul Mac principale (2026-10-06, 5 seed) la curva si ripete quasi
uguale: 0.0122, 0.0081, 0.0065, 0.0104; sulla media dei round 6-10 μ = 0.01 sta sotto μ = 0.001 e μ = 0.1 in tutti i seed.

**Inizializzazione comune, con DP (2026-10-05; Windows, `experiments_altre_macchine/rq3-ci-*`, commit ff23a1c pulito, 5
seed per condizione; FedProx senza DP completato il 2026-10-06).** Con tutti i client che partono dagli
stessi pesi, FedProx con DP batte ancora FedAvg in 5 seed su 5: loss al round 10 0.0081 contro 0.0166 (medie
geometriche; a init casuale sulla stessa macchina 0.0068 contro 0.0196), rapporto 0.49 contro 0.34, t(4) = -4.3 sul
logaritmo del rapporto. Il regime del taglio non cambia: FedAvg ha aggiornamenti fra 8 e 6 per tutto il training
(quota cumulata 1.42 contro 1.46), FedProx scende sotto C dal round 3 (8.72 contro 8.58). Le norme grandi di FedAvg
non vengono quindi dalla media di tre reti indipendenti al round 1, ma dall'addestramento locale (50 epoche) che porta
ogni client lontano dal modello globale; il termine prossimale le tiene piccole. Senza DP FedAvg non cambia (0.00041
contro 0.00038), mentre FedProx migliora (0.00081 contro 0.00147): senza DP FedProx costa 1.96 volte FedAvg invece di
3.90, in 5 seed su 5. Senza DP FedProx resta peggio di FedAvg, con DP meglio, in tutti i seed. Costo della DP rispetto al proprio modello senza DP: FedAvg 40 volte (52 a init casuale), FedProx 10 volte (4.6); il costo di FedProx è un quarto di quello di FedAvg (0.25; 0.09 a init casuale). RQ3 regge
all'inizializzazione; il riferimento in simulazione per la validazione sul deployment ha ora le celle con DP a 5 seed.

## 4. Cosa manca

L'elenco ordinato, con comandi e prerequisiti, è `ESPERIMENTI.md`. E-A è chiuso (sezione
3.1b), con la griglia del punto operativo e ε = 64 a 5 seed; E-C è eseguito sulle celle
esistenti, ε = 64 compresa (sezione 3.2). E-D (RQ2) è analizzato con e senza DP (sezione
3.9) e sta in `experiments_altre_macchine/`, come E-E senza DP di Windows (sezione 3.10,
segnalazione 45). I controlli del protocollo sono
eseguiti (sezione 3.1c). E-B a σ = 0, 1, 2 e 5 e il riferimento GroupNorm sono completi (sezione
3.5); E-E sul Mac principale, con e senza DP, le prove su mu e lo screening di C sono completi
(sezione 3.10). Il braccio FedProx con DP di Windows è completo e analizzato (sezione 3.10). Restano: l'aggiornamento delle matrici
con le celle del 2026-09-29, dello screening e del solo clipping, il canary su più siti (prova a un seed completa nei due bracci, campagna
da decidere col supervisore), il canary bilanciato su
Caltech (seed 42 completo sul Mac principale, sezione 3.3; seed 123 su Windows e 456, 789, 1234 sul Mac principale in corso), la validazione sul deployment NVFLARE e
Containerlab (`ESPERIMENTI.md`, gruppo minimo di celle, a campagne chiuse), le
statistiche delle feature per client di E-D, la rianalisi NVFlare. Deciso col supervisore il 2026-10-01: il test appaiato è la metrica primaria per record
(segnalazione 47). Deciso il 2026-10-02 (linea per il paper DSN, `ESPERIMENTI.md`): RQ riformulate,
inizializzazione comune solo per il rerun controllato e la validazione sul deployment, FedProx μ = 0.01
come base con i costi anche rispetto al miglior modello non protetto, solo clipping come esperimento
centrale di RQ3, TOST su Yeom, `central` al punto operativo, superficie A3 per i canary con DP per client.
Da decidere: quanto è centrale l'ε formale per record (Poisson), il perimetro della campagna canary sui
tre siti dopo il pilota A3. I bug che toccano i numeri sono in
`Segnalazioni_tecniche_2026-09-22.md`, punti 1, 5, 6, 9, 35, 36, 38, 45, 48, 50, 51, 55. Fuori dal paper, come infrastruttura o lavoro futuro: ML Plane,
Privacy Auditor, PES, ByzantineDetector, FedMIA-gradient, secondo dataset. Per le
frasi da non scrivere senza evidenza: guida, sezione 11.
