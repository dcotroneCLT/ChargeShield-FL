# ChargeShield-FL — segnalazioni tecniche e bug (revisione del 2026-09-22)

Contesto: revisione indipendente di codice, config, `risultati/*.xlsx`, paper e docs
allo stato del commit `f89c1cc`. Ogni voce indica file e riga; verificala sul codice e
sui JSON prima di correggere, non fidarti della lista. L'ambiente di revisione non aveva
`experiments/`, `logs/`, torch, pytest, quindi nessun numero grezzo è stato rieseguito.

## A. Da fare subito: tocca l'analisi NVFlare in corso

1. **Lo split train/holdout viene ricostruito con il seed sbagliato per 20 run su 25.**
   `scripts/run_nvflare_mia.py:686-700` legge il seed dello split da `--client-config`,
   che di default è il file live `config_fed_client.json` (oggi `seed: 1234`). Il flag
   `--seed` governa solo gli shadow. `scripts/analizza_dump_nvflare.sh` non passa
   `--client-config`, quindi i run per i seed 42/123/456/789 usano una partizione
   membri/non membri diversa da quella dei client reali: l'AUC tende a 0.5 per
   costruzione, senza errore. Nel log compare il warning "Seed ... diverso dal seed
   reale del client NVFLARE ... uso quest'ultimo".
   Fix: aggiungere a ogni riga dello script
   `--client-config nvflare/jobs/chargeshield_poc/app/config/seed_snapshots/config_fed_client_seed<S>_<mode>_eps<E>.json`
   (i 25 snapshot esistono tutti). Scartare i 20 JSON già prodotti senza snapshot.
   **Aggiornamento 2026-09-22:** script rigenerato con `--client-config` e cartelle
   invalide cancellate; rilancio rinviato a dopo lo sweep di ε (vedi `ESPERIMENTI.md`).
   Hardening: registrare `seed` in `meta` del pickle
   (`chargeshield_aggregator.py:1025-1030` oggi salva solo dp_mode/epsilon/delta/max_grad_norm)
   e far fallire `run_nvflare_mia.py` quando `--seed` non coincide con il seed del client
   config, invece di proseguire con quello live.
2. Stesso problema, retroattivo: le run `nvflare-central-seed{123,456,789,1234}` con seed
   nel JSON "DISCORDE" (`risultati/matrice_run_completati.xlsx`) vanno riverificate: il
   registro le tratta come discordanza cosmetica, ma può essere lo stesso disallineamento
   dello split.
3. Per tutte le rianalisi NVFlare resta la discrepanza dichiarata a
   `run_nvflare_mia.py` (blocco "ATTENZIONE — discrepanza nota"): statistiche di
   normalizzazione globali offline contro per-sito nel client reale. Va quantificata o
   allineata, altrimenti i numeri NVFlare restano "indicativi".

## B. Bug che cambiano numeri citati nel paper

4. **`epsilon_record_dp` è sottostimato.** `scripts/run_experiments.py:136-137` legge
   `fl_rounds` e `delta` dalla radice di `cfg`, ma vivono in `cfg["experiment"]`
   (`run_experiments.py:1169`, `:226`): `rounds` vale sempre 1, quindi i passi
   dell'accountant sono 1/3 o 1/10 di quelli reali. Inoltre `n_sessions` è
   `len(train_sessions)` globale (`:7443`), mentre ogni client addestra sul proprio
   sottoinsieme, e l'accountant compone un `PoissonSampledDpEvent` (`:152`) mentre il
   training usa `DataLoader(shuffle=True, drop_last=True)` (`autoencoder_trainer.py:374-377`).
   Stesso bug in `scripts/fix_json_recorddp_fields.py`. L'ε=7.15 a σ=5 e il "~88" a
   σ=1 vanno ricalcolati. `dp_accounting` non è dichiarato in `pyproject.toml` e
   l'ImportError degrada in silenzio a `epsilon_record_dp: None`.
   **Corretto il 2026-09-24.** Il calcolo è in `src/ml/record_dp_accounting.py`;
   `run_experiments.py::_record_dp_fields` (riga 103) lo chiama. Round e delta da
   `cfg["experiment"]` (dove `load_config` scrive anche `--rounds`), nessun default
   silenzioso; n per client passato da `main` (`n_train_per_client` di
   `save_results`), ε per client e massimo in `epsilon_record_dp`. Accanto all'ε di
   Poisson, dichiarato come approssimazione, il JSON riporta
   `epsilon_record_dp_shuffle_bound`: adiacenza per sostituzione, senza
   amplificazione, E·T gaussiani a σ/2, valido con lo shuffle. `dp-accounting` in
   `pyproject.toml`; se manca, warning nel log. JSON già prodotti:
   `scripts/ricalcola_epsilon_record_dp.py` (sola lettura; sostituisce
   `scripts/_applicati/fix_json_recorddp_fields.py`, che resta com'era). Test:
   `tests/test_record_dp_accounting.py`. **Rettifica** a questa voce: 7.15 (σ=5) e
   88 (σ=1) erano già coerenti con 3 round e n = 1924 (Office 1 con canary, dai log),
   sotto l'ipotesi di Poisson: ricalcolati, restano. Il limite valido con lo shuffle
   per quelle run è 343 a σ=5. Per E-B a σ=1 il codice vecchio avrebbe scritto
   ε ≈ 1.0; per client è JPL 4.8, Caltech 5.0, Office 1 30.4 (Poisson), 1212 il
   limite con lo shuffle. Nessuna cella già prodotta cambia confrontabilità:
   training e attacchi non sono toccati, solo i campi di budget del JSON.
5. **Le celle record-DP sono etichettate no-DP.** Il warning `[NO-DP BASELINE]`
   (`run_experiments.py:1280`) scatta con `--no-dp` anche se `record_dp` è attivo.
   Nel registro (`scripts/genera_matrici_faseA.py`) nessuna delle 237 righe riporta
   `record_dp`: le 10 run record-DP compaiono come `no_dp=True`. Il warning deve
   controllare `record_dp.enabled`; il registro deve emettere il campo.
6. **I due script worst-case non calcolano lo stesso percentile.**
   `scripts/analyze_worst_case_vulnerability.py:95-110` usa mid-rank su membri e non
   membri; `scripts/worst_case_livello_di_caso.py:62-72` usa `bisect_left` sui soli
   membri. Risultati 411 vs 412 (no-DP) e 288 vs 290 (dp-fedavg). La docstring del
   secondo dichiara il criterio "identico". Sceglierne uno e ricalcolare; il foglio
   `Worst_case_per_record` pubblica 412/290. Due righe di quel foglio
   (central-sweep5, local-sweep3) non hanno il JSON grezzo in `risultati/worst_case/`.
7. **dp-fedavg e local sono lo stesso codice in simulazione**
   (`run_experiments.py:1396-1401`): la Tabella 2 ha 7 condizioni, non 10. Il docx lo
   dice, `sections/preliminaries.tex` li presenta ancora come tre punti distinti.
8. **ε=8 e ε=16 hanno un solo seed** e sono le uniche celle con utility non distrutta;
   nel foglio `Utility_privacy_limite` stanno accanto a celle a 14 run senza distinzione.
9. **Deduplica e seed.** `scripts/check_significance.py:281` deduplica per il seed nel
   config JSON; il registro usa il nome cartella. Sei run sono discordi: le due logiche
   danno n diversi.

## C. LiRA: non è un bug da patchare alla cieca, ma va deciso

10. Il floor di varianza simmetrico (`run_experiments.py:4409-4413`, default
    `floor_mode=symmetric`) è colpito nel 97-99% dei record: il log-likelihood ratio
    degenera in una funzione monotona della loss (= `_sablayrolles_score`). Su `central`
    `matched_formula_auc` vale 0.21-0.34 contro uno 0.50 riportato: il null può essere
    cancellazione di due artefatti. `member_scoring` resta `"real"` (asimmetrico,
    `:4662-4666`). Il floor è 10 volte più largo dell'intervallo dei dati perché
    `global_in_stats_per_cluster` mescola record e shadow (varianza fra record, non fra
    shadow).
11. Lo score composto somma i log-LR su un numero variabile di round senza normalizzare
    (`_cumulative_scores[_sid] += lira_score`): campioni scorati in 10 e in 3 round
    finiscono nella stessa classifica. Impatta i dump per-campione del worst-case.
12. `lira_debug_uncalibrated_skip_rate = 0.0000` in tutti i gruppi: il filtro 8σ non
    esclude nulla, perché la soglia è calcolata su σ già floorato. Nel paper è
    presentato come adattamento sostanziale.
13. LiRA non è riproducibile a seed fisso (canary_auc_roc cambia fra due tornate
    identiche), la loss grezza sì. Decidere e scrivere quale è la metrica primaria.

## D. Privacy Auditor

14. `config/auditor.yaml` ha `total_rounds_budget: 1000` su 10 round reali:
    `budget_ratio ≈ 0.01`, le soglie `PRIVACY_BUDGET_*` non possono scattare. Solo
    `auditor_tuned.yaml` (budget 3) è coerente, ed è usato da un config solo.
15. La sensibilità vale 1.000 in ogni round perché la norma L2 è saturata dal clipping:
    telemetria identica fra cella con memorizzazione e cella record-DP. Il docstring
    `src/auditor/privacy_auditor.py:254-270` afferma il contrario ("3-4% di
    max_grad_norm") e va corretto. Il proxy non misura rischio di appartenenza.
16. Il commit `f8bb42a` dichiara `docs/AuditorEvaluation.md` e `MetricsReference §5b`:
    nessuno dei due esiste nel repo.

## E. Canary

17. `run_experiments.py:942-985`: per ogni template non membro vengono inseriti due
    record identici (originale ritaggato più clone); il commento a `:944-947` dice
    "qui non ci sono duplicati". Il pool non membro è raddoppiato artificialmente.
18. Lato NVFlare (`chargeshield_executor.py:380-451`, `run_nvflare_mia.py:150-215`) il
    canary usa ancora la semantica legacy: non membri dall'holdout, niente
    `paired_split`. Un canary su NVFlare riprodurrebbe il confondente già diagnosticato.
19. Il protocollo bilanciato (k=20, swap, baseline, 5 seed) esiste solo su Office 1.
    Caltech e JPL hanno solo run a seed 42, sbilanciati, senza blocco canary nel JSON
    (marcati "incompleta"). Scotland ha il config e nessun run.

## F. Altri bug e incoerenze nel codice

20. `run_experiments.py:2297`: `run_yeom = run_yeom` è un no-op; l'alias `run_shadow`
    promesso in `src/plugins/attacks/shadow.py:10-11` non esiste.
21. `drift_detected` è hardcoded `False` (`run_experiments.py:6471, 6487`) mentre il
    CUSUM lo calcola.
22. `GradientManager.privatize_aggregate` (`src/ml/gradient_manager.py:375-387`) usa
    sensibilità `max_i(n_i/N)·C`: vale per adiacenza add/remove; sotto replace-one manca
    un fattore 2. Dichiarare l'adiacenza scelta.
23. `_compute_sigma` (`gradient_manager.py:214-227`) non controlla ε≤1, condizione del
    bound usato; il codice lo chiama con ε=1000 (`run_experiments.py:6357, 6362`).
24. `krum_threshold=3.5` è calibrato su 4 fette dello stesso sito e mai rivalidato per
    n=5 (`run_experiments.py:6300-6329`); `byzantine_attack.enabled` è false ovunque,
    il ByzantineDetector non è mai stato eseguito end-to-end.
25. NVFlare: al round 1 di dp-fedavg il clipping è sul vettore assoluto
    (`chargeshield_aggregator.py:718-736`); `epsilon` default 1.0 silenzioso lato server
    (`:542-546`); contatori di round locali mai validati contro NVFlare.
26. `config/experiment_overfit_control.yaml` è solo commenti: `yaml.safe_load` dà `None`
    e `load_config` crasha. `config/_smoke_auditor.yaml` è un config sperimentale
    record-DP con nome fuorviante. `config/datasets.yaml` e `config/nodes/cluster_*.yaml`
    sono legacy non letti.
27. `scripts/{repair_json_fields,add_n_sessions_to_save,fix_json_recorddp_fields,add_auditor_telemetry}.py`
    riscrivono `run_experiments.py` con splice testuale e sono già stati applicati:
    rilanciarli corrompe il sorgente. Rimuoverli o proteggerli.
28. File spazzatura tracciati: `wegrvb` e `cqEFG24HF2UPIQWèCOX+a` (log identici, commit
    `f8bb42a`), `*.md.docx` in root, `fix_canary_paired_split.patch`,
    `docs/doc_update_canary.patch`.
29. `experiments/` e `logs/` sono gitignorati: nessun numero è verificabile da un clone.
    Almeno i JSON di sintesi per cella, o i loro hash, vanno versionati.

## G. Documentazione e paper

*Nota 2026-09-22 sera: i punti 30-33 riguardano documenti eliminati nella pulizia del
contesto (Architecture, MLPlane, IDS, CaseStudies, errata, Roadmap); quanto di valido
contenevano è in `SISTEMA.md`. Restano da correggere solo i commenti nel codice citati
al punto 30 e i punti 34-36 sul paper. Il config del punto 26 `_smoke_auditor.yaml` è
stato rinominato `experiment_canary_balanced_recorddp_nm5.yaml`; `datasets.yaml`,
`nodes/` e `overfit_control` sono in `config/_legacy_unused/`; gli script del punto 27
sono in `scripts/_applicati/`.*

30. Conteggio parametri incoerente: 570 (`core/autoencoder.py:72`), 618 con BatchNorm,
    "~650" (`gradient_manager.py:419`), 1200 (`Architecture.md:498`).
31. `docs/MLPlane.md` §5.5 descrive StandardScaler; il codice fa min-max [0,1]. §8 dello
    stesso file è interamente stantio (12 client, 13073 sessioni, output linear).
32. `Architecture.md:467` dice epochs 3 (reale 50); cosine threshold 0.85 nella prosa,
    0.3 nel codice; `IDS.md` abstract parla di zero alert, §12.1 ne conta 36; deadline
    DSN diverse fra `Roadmap.md` e README; Makefile target `logs` punta a un container
    inesistente; `CaseStudies.md` §6 cita make target e script che non esistono.
33. `ChargeShield-FL_errata_paper_vs_codice.md` punto 4 è superato: `check_significance.py`
    legge già `composed_lira_auc_roc`. Va datato o aggiornato.
34. Paper LaTeX: titolo "CChargeShield"; "TO BE VERIFIED" in `introduction.tex:14`;
    `preliminaries.tex` usa T=3 come "operating point" mentre la campagna è a T=10;
    δ confrontato con |D|≈1344 vale solo per Office 1; blocco autori non anonimizzato
    per il double-blind.

## H. Aggiunte dal secondo check di coerenza (2026-09-22, sera)

35. **`sections/validation.tex` riporta i numeri canary PRE correzione.** Righe 103-107
    e 126: 0.7570 / 0.7164, Δ = +0.2367, t = 11.25. Dopo il fix del 2026-09-16 (il pool
    raw non è più filtrato dalla calibrazione LiRA, `CanaryPositiveControl.md` §6.6) i
    valori sono 0.7447 / 0.7152, Δ = +0.2299, t = 9.90. Allineare a Overleaf secondo la
    regola di `CLAUDE.md`.
36. **`sections/setup.tex:128-130` inverte Caltech e JPL sulla copertura dell'identificativo
    utente.** Il paper dice 94% Caltech e 52% JPL; la scheda
    `risultati/decision_matrix_ACN_membership_DP.xlsx` (misura del 2026-09-21) dice JPL
    93.5% e Caltech 52.1%. La cifra "31.452 sessioni identificate" non può essere di
    Caltech, che ne ha 31.404 in totale. Stesso errore in
    `docs/paper/paper_privacy_unit_e_ids.md`.
37. **Le run record-DP con `--no-dp` finiscono nel gruppo "no-DP baseline" e lo
    sostituiscono.** `record_dp` non disattiva il client-level: per una cella record-DP
    pulita serve `--no-dp`, e infatti tutte le run record-DP canary hanno `no_dp=True`.
    Ma `check_significance.py::discover_groups` (righe 96-114) etichetta `no_dp=True`
    come "no-DP baseline" e deduplica per `(label, seed)` tenendo il file più recente:
    una run `rq1-recorddp-nm1` a seed 42 sostituirebbe il seed 42 di `nodp-sweep2` senza
    alcun avviso. Stesso problema in `genera_matrici_faseA.py`. Fix: includere
    `record_dp.enabled` e `noise_multiplier` nell'etichetta del gruppo. Da chiudere
    prima di E-B. Il comando E-B nel vecchio TestRoadmap non aveva `--no-dp`:
    corretto in `ESPERIMENTI.md`.
38. **Non esiste un braccio B1, clipping senza rumore.** La Fase B della guida lo
    richiede per separare l'effetto del clipping da quello del rumore; `central` clippa
    ma rumorizza l'aggregato, e `_compute_sigma` non accetta σ = 0. Serve un flag
    `--clip-only` (client clip, nessun rumore, `epsilon_cumulative_*` a `None`), oppure
    il braccio va dichiarato non eseguito.
39. **33 test leggono i dataset reali senza skip.** `tests/test_acn_dataset.py` e
    `tests/test_chargeplace_scotland_adapter.py` aprono file sotto `datasets/`, che non è
    versionato: in un clone senza dati falliscono con `FileNotFoundError` invece di
    essere saltati (verificato il 2026-09-22: 264 passed, 2 failed, 31 errors). Aggiungere
    `pytest.mark.skipif(not Path(...).exists(), reason=...)` a livello di modulo, così la
    baseline resta leggibile ovunque; la funzionalità va coperta anche da un test su un
    file sintetico piccolo committato in `tests/`.
40. **Commenti e docstring nel codice rimandano a documenti eliminati il 2026-09-22**
    (TestRoadmap, Positioning, CaseStudies, DeveloperGuide, PES_v1): 22 occorrenze in
    `scripts/run_experiments.py`, 5 in `compute_pes.py`, 3 in `check_significance.py`,
    3 in `generate_excel_report.py`, 2 ciascuno in `compare_results.py`,
    `src/core/autoencoder.py`, `src/core/base_attack.py`, `src/plugins/attacks/__init__.py`,
    `src/adapters/chargeplace_scotland_adapter.py`, una in `autoencoder_trainer.py`,
    `gradient_manager.py`, `ml_plane.py`, `privacy_auditor.py`, `fedmia.py`. Non bloccano
    nulla: sostituirli con `docs/SISTEMA.md` o `docs/STATO.md` quando si tocca il file per
    altro, non con un commit dedicato mentre la campagna gira.

## I. Trovato costruendo l'inventario dell'ambiente d'esecuzione (2026-09-22)

41. **`requests` è usato ma non dichiarato in nessuna dipendenza di
    `pyproject.toml`.** `scripts/download_acn_sessions.py:39` fa `import requests`
    a livello di modulo, senza fallback — è l'unico modo nel repo per scaricare
    ACN-Data da ev.caltech.edu. Non compare né in `[project.dependencies]` né in
    nessun `[project.optional-dependencies]` (`dev`, `flare`, `viz`). A differenza
    di `dp-accounting` (segnalazione 4), qui non c'è degradazione silenziosa: chi
    non ce l'ha installato ottiene subito un `ModuleNotFoundError` lanciando lo
    script. Fix: aggiungerlo a `[project.dependencies]` in `pyproject.toml`.
    Verificato leggendo gli `import` reali di tutto `src/` e `scripts/`, non solo
    questo file — nessun'altra libreria di terze parti usata nel codice risulta
    mancante dalla dichiarazione (oltre a `dp-accounting`, già nota).

## J. Trovato preparando la riscrittura del paper (2026-09-23)

42. **Un attacco che fallisce non ferma lo sweep e la cella perde un seed in
    silenzio.** `scripts/run_experiments.py:6174` (`run_registered_attacks`)
    intercetta l'eccezione di ogni attacco registrato, scrive comunque il JSON con
    le metriche a `None` ed esce con codice 0, quindi il `set -e` degli sweep non
    si ferma. Caso reale in E-A: `experiments/rq1-eps2/experiment_20260922_180414.json`
    (epsilon = 2, quarto seed del ciclo, cioe' 789) ha Yeom, Shadow e LiRA falliti
    con `AttributeError: module 'numpy' has no attribute '_no_nep50_warning'`
    (`logs/rq1_eps_sweep.log`, righe 1540-1697, ore 18:04 del 2026-09-22), JSON da
    40 KB contro i 90 KB degli altri e nessun `per_sample_seed789.json`. La causa e'
    d'ambiente (numpy/scipy incoerenti in quel momento: le run successive sono
    andate), ma il codice la rende invisibile. Da fare: rilanciare la sola cella
    epsilon = 2, seed 789, a fine sweep, prima di `check_significance.py`; valutare
    un codice d'uscita non nullo quando falliscono tutti gli attacchi (tocca
    `run_experiments.py`: solo dopo la campagna).

## K. Trovato rigenerando le matrici dopo E-A (2026-09-24)

43. **`genera_matrici_faseA.py` duplicava i fogli a ogni esecuzione.** Le quattro
    funzioni che aggiungono fogli a `Matrice_sintesi.xlsx` (`scrivi_costo_per_sito`,
    `scrivi_utility_privacy`, `scrivi_worst_case`, `scrivi_glossario`, righe 561-718
    prima della correzione) aprono il file esistente e chiamano `create_sheet`: alla
    seconda esecuzione openpyxl crea `Utility_privacy_limite1` accanto al vecchio
    `Utility_privacy_limite`, che resta con i numeri precedenti. Chi apre il foglio
    col nome atteso legge dati vecchi. **Corretto il 2026-09-24**: ogni funzione
    elimina il foglio omonimo prima di ricrearlo; `Matrice_sintesi.xlsx` rigenerato
    da quello committato, sei fogli senza duplicati.
44. **Il foglio `Utility_privacy_limite` non compone le celle come
    `check_significance.py`.** `scrivi_utility_privacy` (stesso script) raggruppa per
    `(dp_mode, epsilon)` TUTTI i JSON di `experiments/*/`, senza escludere le cartelle
    `_*` e `nvflare-*` e senza deduplicare per seed; `check_significance.py` fa
    entrambe le cose. Effetti visti il 2026-09-24: a ε = 8 e 16 la run pilota a un
    seed finisce insieme alle cinque di E-A (n = 6; a ε = 8 la media della loss scende
    da 0.0655 a 0.0566); `local` ε = 0.5 e 0.1 includono una run NVFlare rianalizzata
    il 2026-09-22 (la loss di `local` ε = 0.5 è passata da 0.4235 a 0.4118 senza nuove
    simulazioni); le celle a 14 run sommano sweep diversi dello stesso seed. Va deciso
    insieme alla segnalazione 9 quale regola vale; finché non è deciso, `STATO.md`
    riporta E-A dai JSON, una run per seed. **Corretto il 2026-09-24**: i fogli di
    `genera_matrici_faseA.py` compongono le celle con `check_significance.discover_groups`
    (funzione `composizione_celle`), quindi stesse esclusioni e una run per (cella, seed),
    la piu' recente. Effetto sul foglio `Utility_privacy_limite`: la cella no-DP passa da
    49 run (holdout 0.00244, LiRA composto 0.513, con le calibrazioni di overfitting) alle
    5 di `nodp-sweep2` (0.00158, 0.500), e i rapporti di costo salgono (dp-fedavg eps = 1
    da 161 a 249 volte). Le celle di `check_significance.py` non cambiano. Resta da
    dichiarare la scelta del seed del config invece del nome della cartella (segnalazione 9).

## L. Trovato preparando E-E (2026-09-24)

45. **Le run no-DP di E-D ed E-E finiscono nel gruppo "no-DP baseline" e ne
    sostituiscono i seed.** `check_significance.py::discover_groups` (righe 273-330)
    etichetta "no-DP baseline" ogni run con `no_dp=True` e senza `record_dp`, senza
    guardare `partition_strategy` ne' `ml.proximal_mu`, che pure sono registrati nel
    JSON, e deduplica per `(etichetta, seed)` tenendo il file piu' recente. Le run di
    `rq2-per_site` e `rq2-iid` (E-D, 22-23 settembre) e quelle di E-E hanno
    `no_dp=True`, gli stessi cinque seed e timestamp successivi a `nodp-sweep2`
    (8 settembre): copiate in `experiments/`, sostituirebbero in silenzio i seed del
    riferimento no-DP di RQ1, e fra loro vincerebbe l'ultima in ordine di timestamp.
    `genera_matrici_faseA.py` fa lo stesso (cella "no-DP" alle righe 379 e 633, e il
    foglio `Utility_privacy_limite` non deduplica, segnalazione 44). Stessa classe
    della 37. Fix: includere `partition_strategy` e `proximal_mu` nell'etichetta del
    gruppo in entrambi gli script (tocca il codice: da autorizzare). Finche' non e'
    corretto, le cartelle di E-D ed E-E restano fuori da `experiments/` (per esempio
    in `experiments_altre_macchine/`) e si analizzano a parte. **Corretto in parte il
    2026-09-24** (`scripts/etichetta_cella.py`): `proximal_mu`, `partition_strategy`,
    `split_strategy`, `max_grad_norm`, `common_init`, epoche, round e architettura entrano
    nell'etichetta quando differiscono dalla base, quindi i bracci IID e mu = 0 hanno celle
    proprie. Resta vero per i bracci con il config della base (E-D `per_site`, E-E
    mu = 0.01): sono repliche di `nodp-sweep2` e, con la regola "la piu' recente", ne
    sostituirebbero i seed. Finche' la segnalazione 9 non e' decisa restano fuori da
    `experiments/`. Dal 2026-09-24 E-D sta in `experiments_altre_macchine/` (non
    versionata) e `genera_matrici_faseA.py` la legge a parte (foglio `RQ2_partizione`).
    Le repliche non sono identiche (segnalazione 55): la regola "la piu' recente"
    cambierebbe davvero i numeri del riferimento.

## M. Revisione del codice e dei numeri (2026-09-24)

46. **Il costo "loss finale" non era la loss del modello rilasciato.**
    `genera_matrici_faseA.py::scrivi_utility_privacy` (righe 626-634 prima della
    correzione) prendeva `per_round[T].fl.mean_loss`, cioe' la media pesata delle loss
    di addestramento LOCALE dei client nell'ultimo round (`fedavg_aggregator.py:138-143`,
    media sulle 50 epoche, prima del rumore DP di quel round). Docstring, glossario, §0
    di `ESPERIMENTI.md` e paper dichiarano invece la loss del modello globale finale
    sull'holdout. Senza DP coincidono (`nodp-sweep2`: 0.00168 contro 0.00158); con DP
    client-level no: a ε = 16 0.0061 contro 0.0945, cioe' 60 volte il no-DP appaiato
    (30-157 per seed) invece di 2.3-3.7. **Corretto il 2026-09-24**: la colonna di costo
    e' `-non_member_score_mean` di Yeom al round finale (modello globale rilasciato,
    tutto l'holdout), gia' presente nei JSON; manca solo nel vecchio JSON del seed 789 a
    ε = 2, gia' rilanciato. La loss locale resta come diagnostica e `Costo_per_sito` e'
    rietichettato. Nessuna run da ripetere; cambia la lettura di E-A: nessun ε in 2-16
    sotto la soglia di 3 volte.
47. **Il conteggio per record non misurava l'appartenenza.**
    `worst_case_livello_di_caso.py` (percentili fra i soli membri e permutazione dentro
    il seed, righe 46-91 prima della correzione) misura la stabilita' del ranking di un
    record fra seed, che c'e' anche per un record facile da ricostruire indipendentemente
    dall'appartenenza. Controllo sui dump di `nodp-sweep2`: lo stesso criterio sulle
    apparizioni da non-membro segnala 406 sessioni contro 288.9 attese, z = 6.85 (membri
    412, z = 7.56); test appaiato sulle 28 129 sessioni membro in un seed e non-membro in
    un altro: differenza di percentile +0.06 ± 0.22. **Corretto il 2026-09-24**: lo
    script riporta anche il controllo sui non-membri, lo z della differenza e il test
    appaiato, che e' la lettura primaria; aggiornati `scrivi_worst_case` e
    `analyze_worst_case_vulnerability.py`. E-C ricalcolato su 8 celle: nessun segnale di
    appartenenza per record (z appaiato fra -1.38 e +0.91). La metrica primaria per
    record del §0 va sostituita: decisione del supervisore. **Decisa il 2026-10-01**: test
    appaiato primario, conteggio secondario col controllo sui non membri; calcolato su tutte
    le 24 celle a 5 seed (z appaiato fra -1.49 e +1.59).
48. **I client non partono dallo stesso modello iniziale.**
    `run_experiments.py::run_fl_rounds` crea i trainer uno dopo l'altro (righe 1211-1219
    prima della correzione) senza distribuire un modello comune: nel round 1 FedAvg media
    tre reti con inizializzazioni diverse. In `nodp-sweep2` il modello globale dopo il
    round 1 ha holdout 0.057-0.065 contro 0.0012 di loss locale; al round 3 e' recuperato
    (0.003) e al round 10 coincide con la loss di addestramento. Non tocca le run canary,
    che hanno un solo client. A parita' di seed l'inizializzazione e' la stessa in tutte
    le celle, quindi i confronti restano omogenei. **Aggiunta il 2026-09-24** l'opzione
    `ml.common_init` (default False: run esistenti invariate) con il config di controllo
    `experiment_ctrl_common_init.yaml`; il JSON registra `common_init`. **Misurato il
    2026-09-24** (`experiments/_ctrl_common_init`, seed 42, commit `6d31e21`): holdout
    del modello rilasciato 0.0080 contro 0.0651 al round 1 e 0.00108 contro 0.00219 al
    round 10, dentro la variabilita' fra seed di `nodp-sweep2` (0.00070-0.00219). Tutti
    gli attacchi restano al caso (Yeom 0.4989, LiRA composto 0.4991, TPR a FPR 1%
    0.0099). Le campagne restano valide e lo scostamento dal protocollo si dichiara;
    se passare al protocollo standard lo decide il supervisore.
49. **Con la DP a livello di record gli attacchi saltavano ogni round.** Il trainer forza
    `norm: group` quando `record_dp` e' attivo (`autoencoder_trainer.py:146-152`), ma
    `_autoencoder_arch_kwargs` ricostruiva BatchNorm se il config non diceva altro, e i
    config `experiment_rq1_recorddp_nm*.yaml` non lo dicevano: FedMIA e Shadow saltano
    ogni round ("global_weights ha 16 elementi, state_dict ne richiede 22"). Visto nella
    prova E-B del 2026-09-24, fermata (`logs/prova_recorddp_s42_attacchi_saltati.log`,
    nessun JSON). Le run canary record-DP non sono toccate: i loro config hanno
    `norm: group`. **Corretto il 2026-09-24**: `norm: group` nei quattro config e la
    stessa regola del trainer in `_autoencoder_arch_kwargs`. **Verificato il 2026-09-24**
    con la prova `_prova_recorddp_v2` sul secondo Mac: Yeom, Shadow e LiRA producono un
    AUC in ogni round, nessun `skip` oltre al round 0; la loss di addestramento dei 10
    round coincide alla sesta cifra con il primo tentativo.
50. **Il TOST gira sul LiRA composto, la metrica primaria e' Yeom.**
    `check_significance.py` applica `tost_equivalence` a `composed_lira_auc_roc` (blocco
    TOST in `main`), mentre il §0 fissa la loss grezza (Yeom) come metrica primaria; l'IC
    e' bootstrap percentile su 5 seed, piu' stretto di uno basato su t. Con Yeom fra
    0.4999 e 0.5015 la conclusione non cambia. Da allineare.
51. **In E-B gli shadow non replicano l'addestramento del bersaglio.** Con `--no-dp`,
    necessario per il record-level, gli shadow di LiRA e dello Shadow attack si
    addestrano senza DP-SGD (`run_lira`, ramo `if not no_dp` della privatizzazione degli
    shadow): la calibrazione LiRA sotto record-level confronta il bersaglio con shadow non
    privati. LiRA e' secondaria; da dichiarare, o da correggere prima di leggere E-B con
    LiRA.

## N. Trovato eseguendo `make test` sul Mac (2026-09-24)

52. **Un test torch fallisce: LiRA da' lo stesso AUC con gli update scalati per 1000.**
    `make test` sul Mac dopo il commit `ff28f50`: 447 passati, 1 fallito,
    `tests/test_run_experiments_integration.py::TestRunLiRA::test_lira_attacks_post_dp_updates_not_raw`
    (righe 343-390): AUC 0.498437 sia con gli update normali sia con quelli scalati.
    Il test protegge la correzione 2026-07-21c, cioe' LiRA che legge `updates` e non
    `raw_updates` (`run_lira`, ramo `else` intorno alla riga 3864). Con il config del test
    le modifiche di `ff28f50` non toccano quel percorso: niente `record_dp`, niente
    `common_init`, il log delle norme legge soltanto i pesi. Nello stesso passaggio
    `pip install dp-accounting` ha aggiornato numpy da 1.26.4, la versione con cui sono
    state prodotte le run sul Mac, a 2.5.3 (dm-tree 0.1.10, dipendenza di dp-accounting,
    chiede numpy >= 2.3 su Python 3.14; dp-accounting lo usa solo nei suoi test). numpy
    riportato a 1.26.4: il test fallisce uguale (1 fallito, 35 passati nei due file), quindi
    numpy non c'entra. **Causa probabile, nel test e non in LiRA**: le sessioni sintetiche
    del test non sono normalizzate (`_make_sessions`, `minutes_available` fino a 600),
    mentre il decoder ha una Sigmoid in uscita (`src/core/autoencoder.py:178-180`) e
    ricostruisce in [0,1]. La loss e' dominata dalla scala dei dati e quasi non dipende
    dai pesi, quindi l'AUC non cambia neanche moltiplicandoli per 1000. Le run vere
    normalizzano (`run_experiments.py::main`, righe 7260-7262), e il codice di `run_lira`
    con `no_dp` legge `updates` (righe 3854-3867). **Corretto nel test il 2026-09-24**:
    normalizzazione come in `main`, e il test accetta come cambiamento anche il round
    saltato o una media dei punteggi diversa. **Verificato sul Mac il 2026-09-24**:
    `tests/test_run_experiments_integration.py` 23 passati su 23, con numpy 1.26.4.
53. **La soglia di clipping C non e' nel JSON, e le celle non distinguono C ne'
    `common_init`.** `save_results` (`run_experiments.py`, blocco `config`) salvava
    epsilon, delta e round ma non `experiment.max_grad_norm`. `genera_matrici_faseA.py`
    (`scrivi_confronti`, riga 379; `scrivi_utility_privacy`, riga 679) forma le celle con
    `(dp_mode, epsilon)` o "no-DP", e `check_significance.py::discover_groups` deduplica
    per `(etichetta, seed)`. Una run con C = 0.25, eps = 16 e seed 42 finirebbe nella cella
    di E-A a eps = 16 e, in `check_significance.py`, ne sostituirebbe il seed 42. Lo stesso
    vale per `experiments/_ctrl_common_init`: il foglio `Utility_privacy_limite`, che non
    esclude le cartelle `_*` (segnalazione 44), la metterebbe nella cella no-DP e
    cambierebbe il denominatore di tutti i rapporti di costo. Stessa classe della 45.
    **Corretto il 2026-09-24 nel JSON**: `config.max_grad_norm` salvato da questo commit;
    tutti i config nel repository a quella data hanno C = 1.0. **Corretto negli script
    di analisi il 2026-09-24**: etichetta di cella comune in `scripts/etichetta_cella.py`,
    usata da `check_significance.py` e, tramite `composizione_celle`, da
    `genera_matrici_faseA.py` (test in `tests/test_etichetta_cella.py`). Una run con
    C = 0.25 va nella cella "dp-fedavg, eps=16.0, C=0.25"; `_ctrl_common_init` avrebbe
    "no-DP baseline, init comune" ed e' comunque esclusa come cartella `_*`.
    **Stessa classe, 2026-09-27, prima che capitasse**: `norm` non era fra i campi
    dell'etichetta, e la cella no-DP con GroupNorm di E-B sarebbe finita in "no-DP baseline"
    sostituendo i seed di `nodp-sweep2`. Aggiunto a `BASE` in `scripts/etichetta_cella.py`
    (non conta con record_dp, dove e' sempre group); nessuna cella esistente cambia
    etichetta, solo sette run di cartelle `_*` gia' escluse. Test in
    `tests/test_etichetta_cella.py`.
54. **Nei log l'attacco di Yeom si chiama ancora "FedMIA".** `run_yeom`
    (`run_experiments.py`, righe 1978, 2013 e 2184 prima della correzione) scriveva
    `FedMIA — members`, `FedMIA pool bilanciato` e `Round N — FedMIA AUC-ROC`: residuo del
    nome `run_fedmia`, rinominato nello Sprint 10zz+107. Chi legge un log crede che giri
    l'attacco FedMIA della letteratura, che non e' nel registro (Yeom, Shadow, LiRA) ne' nel
    paper. Nessuno script legge quelle righe: JSON e matrici non cambiano. **Corretto il
    2026-09-24**: le tre stringhe dicono Yeom. I log prodotti prima restano con la vecchia
    etichetta, comprese le prime due celle della griglia del punto operativo.
55. **Stesso config su due Mac: la loss sull'holdout differisce fino al 23% per seed.**
    Il braccio per sito di E-D (secondo Mac, Python 3.13, commit 3815da7) e `nodp-sweep2`
    (Mac principale) hanno lo stesso config, ma la loss sull'holdout al round 10 differisce
    per seed fino al 23% (seed 123: 0.00202 contro 0.00164) e la loss di addestramento fino
    a 3e-4 in valore assoluto. Il codice di training e' lo stesso: `git diff 3815da7 HEAD`
    su `src/ml` aggiunge solo l'accountant, le modifiche a `run_fl_rounds` non toccano il
    percorso no-DP, e il controllo (a) riproduce `nodp-sweep2` alla sesta cifra sul Mac
    principale. La differenza viene quindi dalla macchina, sul percorso no-DP con
    BatchNorm; con la DP per record (GroupNorm) i due Mac coincidono alla sesta cifra.
    **Confermato il 2026-09-24** con il controllo (a) sul secondo Mac (codice di oggi,
    `logs/ctrl_riproduzione_mac2.log` su quella macchina): la loss dei 10 round coincide
    alla sesta cifra con il braccio per sito di E-D, girato li' con il commit 3815da7, e
    differisce da `nodp-sweep2` gia' al round 1 (0.001195 contro 0.001200). Conseguenza: le celle vanno
    confrontate sulla stessa macchina, e ogni confronto fra macchine va dichiarato; vale
    per E-D, per E-E su Windows e per E-B sul secondo Mac contro il client-level del
    principale. La differenza resta dentro la variabilita' fra seed (0.00070-0.00219).
    **Riscontro del 2026-09-27 con GroupNorm**: E-B σ = 1, seed 42, sul Mac principale
    (commit 943c795) e `_prova_recorddp_v2` sul secondo Mac (2c4a0f0) hanno loss di
    addestramento per round identiche, holdout 0.00239868088 contro 0.00239868097 e Yeom
    uguale alla settima cifra; LiRA composto no (0.5025 contro 0.5043), come nello stesso
    Mac (`STATO.md` 3.4). L'effetto macchina riguarda il percorso con BatchNorm.

## O. Trovato aggiornando il registro dopo il secondo seed di E-B (2026-09-26)

56. **Il registro delle run descrive male le run record-DP e non vede le altre macchine.**
    In `scripts/genera_matrici_faseA.py`, prima della correzione: `superficie` (riga 130)
    metteva ogni run con `no_dp=True` in "A0 — update non privatizzato", comprese quelle
    record-DP, che hanno `no_dp=True` solo per spegnere il client-level e osservano un
    update da DP-SGD; `leggi_run` scriveva per tutte "nessun accountant RDP" anche quando il
    JSON ha `epsilon_record_dp` dall'accountant (segnalazione 4); `regime` (riga 146) le
    chiamava "diagnostico", che non vale piu' per la campagna E-B; `algoritmo` usava un test
    di verita' su `proximal_mu`, quindi con mu = 0 (E-E) avrebbe scritto "non registrato";
    il motivo dello stato diceva "commit non registrato" anche per le run che hanno
    `git_commit`; e `leggi_run` leggeva solo `experiments/`, quindi le 21 run di
    `experiments_altre_macchine/` (E-D e la prova E-B), usate nei fogli, mancavano dal
    registro. Nessun numero dei fogli di sintesi cambia: tocca solo le colonne descrittive di
    `matrice_run_completati.xlsx`. **Corretto il 2026-09-26** (test in
    `tests/test_costo_e_per_record.py`): il registro passa da 276 a 297 righe. Restano in A0
    le run canary record-DP del 16-17 settembre, che non hanno il blocco `record_dp` nel
    JSON: dal file non si ricostruisce. **Stessa classe, trovato il 2026-09-27** a E-B
    completa: `scrivi_utility_privacy` saltava le celle record-DP (`if
    et.startswith("record-DP"): continue`, dal commit 2c4a0f0, quando c'era solo la prova a
    un seed) e `scrivi_confronti` le escludeva perche' il loro regime non e' "naturale"; in
    `scrivi_sintesi` le righe RQ1 e RQ3 e in `scrivi_confronti` la riga C-RQ3 avevano testo
    fisso del 2026-09-21 ("C01..C11", "proximal_mu = 0.01 su 235 run su 235"). **Corretto
    il 2026-09-27**: la cella "record-DP, nm=1.0" entra in `Utility_privacy_limite` con l'ε
    per record al posto di eps_tot e la sua lettura, e in `matrice_confronti.xlsx` come C15
    con le differenze non controllate dichiarate (GroupNorm, clipping per esempio); le righe
    RQ1 e RQ3 e C-RQ3 si calcolano dal registro e dalle celle.
57. **Una modifica ai documenti durante una campagna marca la run come `-dirty`.**
    `_git_commit_corrente` (`scripts/run_experiments.py`, righe 333-358) legge
    `git status --porcelain` al salvataggio: qualunque file modificato o non tracciato, anche
    un documento, aggiunge `-dirty`. E-B σ = 1, seed 456, salvato alle 15:18 del 2026-09-26,
    ha `1519d16...-dirty`: la stima di salvataggio data in chat (15:35-15:40) era in ritardo,
    e le modifiche a README, `CLAUDE.md`, `config/README.md`, `docs/` e il nuovo
    `config/experiment_rq3_mu0_eps64.yaml` sono state committate dopo, in a93feca. `git diff
    1519d16 a93feca` tocca solo quei file e nessuno e' letto dalla run: il codice eseguito e'
    quello di 1519d16, e il training e gli attacchi sono gli stessi di 0dedd34 (seed 123;
    fra i due commit cambiano solo `genera_matrici_faseA.py` e i test). **Regola di lavoro da
    oggi**: modifiche al repository solo lontano da un salvataggio, con margine di un'ora
    sull'ora stimata. Da valutare a campagna chiusa, perche' tocca `run_experiments.py`:
    registrare nel JSON anche l'elenco dei file modificati, cosi' il `-dirty` si verifica
    dal file.
58. **Il dump per campione contiene solo i record che LiRA punteggia, e con record-DP sono
    meno.** Il dump (`scripts/run_experiments.py`, righe 5449-5460) si costruisce da
    `_cumulative_scores` di LiRA, non dall'elenco dei membri e non membri bilanciati. A
    seed 42: 26 678 record in `nodp-sweep2` e in `rq1-eps64`, 26 616 in
    `rq1-recorddp-nm1` (13 313 membri contro 13 340). Il test per record della cella
    record-DP gira quindi su 27 877 sessioni invece di 28 129, e il conteggio su 17 389
    invece di 17 561. Quale passo di LiRA escluda i record in piu' e' da verificare; la
    differenza e' dello 0.9% e non cambia la lettura (z appaiato 0.18). Non si tocca durante
    le campagne: e' codice di scoring.
59. **La loss di addestramento locale di FedProx include il termine prossimale.**
    `train_step` (`src/ml/autoencoder_trainer.py`, righe 221-229) aggiunge
    (mu/2)||w − w_global||² alla loss e restituisce la somma, e `per_round[r].fl.mean_loss` e'
    la media di quei valori. Con mu > 0 la colonna "loss addestramento locale (diagnostica)"
    delle matrici non e' quindi solo ricostruzione: in E-E senza DP, al round 2, FedProx
    registra 0.0057-0.0073 contro 0.0004-0.0006 di FedAvg, in parte per il termine stesso.
    Fra mu diversi quella colonna non si confronta; si confronta la loss sull'holdout del
    modello rilasciato, che e' solo ricostruzione. Fra celle con lo stesso mu (tutta RQ1) la
    colonna resta coerente. Non si tocca durante le campagne: e' codice di training; il
    glossario delle matrici va aggiornato.

## P. Trovato importando le run di Windows e della quarta macchina (2026-09-29)

60. **Con DP l'effetto macchina è più grande che senza.** FedAvg a ε = 64, C = 1, stessi
    seed e config: la loss sull'holdout su Windows (`experiments_altre_macchine/rq3-mu0-eps64`,
    fe22124) è 1.61 volte quella del Mac principale (`experiments/rq3-mu0-eps64`) in media
    geometrica, da 1.00 a 2.45 per seed; senza DP lo stesso rapporto vale 0.94 (da 0.81 a
    1.05). Le versioni differiscono: Windows Python 3.13.14, torch 2.14.0, numpy 2.5.3; Mac
    principale Python 3.14.4, torch 2.11.0, numpy 1.26.4. Estende la segnalazione 55: i
    confronti con DP client-level si fanno solo sulla stessa macchina. Con GroupNorm e DP per
    record il controllo a 2 round sulla quarta macchina (torch 2.11, numpy 1.26.4) riproduce
    il Mac principale.
61. **Sul clone Windows `logs/` non è ignorato e marca le run `-dirty`.** `.gitignore` (riga
    19) esclude `*.log`, non la cartella: `logs/` conteneva file con altra estensione e, con
    uno zip nella radice, `git status --porcelain` in `_git_commit_corrente`
    (`scripts/run_experiments.py`, righe 333-358) non era vuoto. Le 5 run FedAvg con DP di
    Windows hanno `fe22124-dirty` con codice identico a fe22124 (`PROVENIENZA.txt`): valide
    come nella segnalazione 57. Rimedio applicato: `logs/` in `.git/info/exclude` sul clone
    Windows. Da fare a campagna chiusa: `logs/` in `.gitignore`.
62. **Su macOS `nohup` non stacca lo standard input dal terminale.** Un ciclo lanciato con
    `nohup caffeinate ... bash -c 'for s in ...; do python3 ...; done' &` da un terminale
    che poi si chiude (qui VS Code, 2026-09-29, quarta macchina) sopravvive, ma ogni nuovo
    `python3` fallisce all'avvio con `Fatal Python error: init_sys_streams`, `Errno 9 Bad
    file descriptor`: la run gia' partita finisce, le successive no, senza Traceback e senza
    righe `[ERROR]` (`logs/rq1_recorddp_nm0.log`, righe 508-530). Il `nohup` di macOS
    reindirizza solo standard output ed errore. Rimedio: aggiungere `< /dev/null` al comando
    di lancio e usare l'app Terminale; per i cicli gia' partiti senza, non chiudere la
    finestra da cui sono stati lanciati. Controllo utile: contare anche "Fatal Python error"
    nei log, non solo "Traceback".

## Q. Trovato con la run di prova del canary su più siti (2026-09-30)

63. **Con più client gli shadow di LiRA non riproducono il modello del client dal round 2.**
    Nella run `experiments/_canary_multisite_s42` (tre siti, canary in Office 1, shadow cold)
    gli shadow partono da pesi casuali e si allenano solo sul cluster, mentre dal round 2 il
    client riparte dal modello globale e FedProx lo trattiene vicino. Le loss vere finiscono
    oltre 8 sigma da entrambe le distribuzioni degli shadow e il campione viene scartato
    (`scripts/run_experiments.py`, righe 4688-4693, `_UNCALIBRATED_Z_THRESHOLD = 8.0` alla
    riga 4348): `lira_debug_uncalibrated_skip_rate` vale 0.0, 0.092 e 0.031 nei tre round
    (2419 e 816 campioni ai round 2 e 3), contro 0 in `_canary_balanced_s42` (un solo sito).
    Fra i canary lo scarto è molto più forte: al round 2 entrano nell'AUC di LiRA 403 record
    membro su 620 e 20 non membro su 40, 13 x 10 gruppi distinti invece di 20 x 20. Lo scarto
    non compare in `_diag_canary_skip_reason` (riga 4229), che conta solo
    `tensor_extraction`, `cross_cluster_guard` e `insufficient_calibration` e riporta 0
    salti: la riga `[CANARY DIAG]` del log sottostima i canary esclusi. La loss grezza
    (`canary_raw_mse_auc_roc`) è raccolta prima dello scarto e usa sempre 20 x 20 coppie.
    Conseguenza: con più client `canary_auc_roc` dal round 2 non è interpretabile e la
    metrica del canary è la loss grezza. Non corretto: codice di scoring, campagna in corso.
    Da decidere dopo la campagna: contare lo scarto nella diagnostica dei canary e, per la
    superficie del client, shadow che ripartono dal modello globale del round.
64. **Con più client la calibrazione di Shadow sui canary è inerte.** Nella stessa run
    `shadow_canary_auc_roc` coincide con `yeom_canary_auc_roc` in tutti e tre i round
    (0.4625, 0.3825, 0.39). Il punteggio calibrato è `shadow_mse - target_mse`
    (`scripts/run_experiments.py`, righe 2754-2755): lo shadow model, centralizzato e
    addestrato 500 epoche, ha sui canary loss di 1e-5 - 1e-4, il modello globale federato di
    3e-3 - 8e-2 (`shadow_canary_debug_group_raw` nel JSON), quindi la differenza ordina come
    la sola loss del target. Non è un errore di codice ma la stessa discrepanza fra shadow e
    target della segnalazione 63; su un solo sito i due valori differiscono. Da riportare se
    si usa Shadow sui canary con più client.

## R. Revisione profonda di LiRA, RMIA, DP e script persona (2026-10-16)

Tre revisori indipendenti (LiRA/RMIA; DP per soggetto, per record e per client; `mia_persona.py` e
`canary_persona.py`), poi controllo dei punti gravi. Righe di `scripts/run_experiments.py` riferite
al file di 88dfae8 (md5 226fc5d4). "Codice" = verificato leggendo il codice; "misura" = verificato
eseguendo. Decisione della dottoranda, 16 ottobre: si correggono tutti i punti; GroupNorm in tutte le
celle, DP per record con campionamento di Poisson, soggetti ricostruiti dalle abitudini con profili
appresi fuori dal training.

65. **Gli shadow di LiRA non si allenano come il client reale, in nessuna modalità** (codice; misura
    su dati sintetici). Ciclo Adam a righe 4216-4234: niente termine FedProx (i config hanno
    mu = 0,01; norma del delta 0,51 con mu contro 1,41 senza, revisore), Adam nuovo a ogni round
    mentre il client conserva lo stato, insieme IN di (membri + holdout)/2 (riga 4022), circa il 62%
    dei dati del client. Tocca tutti i numeri LiRA e RMIA prodotti.
66. **Con DP per record o per soggetto gli shadow erano senza DP** (codice; i log dicono
    `dp_su_shadow=no (--no-dp)`, per esempio `logs/rq1_recorddp_nm1.log`). Non validi i numeri di
    LiRA di `rq1-recorddp-nm0/nm1/nm2/nm5` (20 run) e `_final_rdp_*` (6 run). Corretto il 16
    ottobre nel ramo con `AutoencoderTrainer.train_local` (`_shadow_via_trainer`), test
    `tests/test_lira_shadow_meccanismo.py`; restano i punti della 65.
67. **dp-fedavg con DP: il bersaglio e' l'update grezzo, gli shadow sono privatizzati** (codice): riga
    4117 contro riga 4267. La calibrazione non corrisponde al bersaglio.
68. **La varianza di LiRA e' quasi sempre quella di ripiego** (log: floor_hit circa 0,997; misura del
    revisore con 16 shadow: la soglia vale 4-17 volte la dispersione fra shadow). Righe 4678-4720.
    Con sigma_in = sigma_out = soglia il punteggio diventa una forma di Sablayrolles pesata; la
    scala 1/sigma^2 diversa per client entra nel ROC aggregato.
69. **Superficie "global": shadow di un solo client contro il modello aggregato** (codice, righe 4141
    e 4631).
70. **RMIA** (codice, righe 3019-3070): popolazione z definita con le etichette (`~memb`), innocua
    con gamma = 1 ma sbagliata in generale; versione offline con a = 0,3 fisso, non invariante di
    scala; RMIA e LiRA valutati su insiemi diversi; RMIA composto calcolato solo se LiRA composto
    non e' vuoto.
71. **I numeri dipendono dal numero di thread** (misura: con 1 e 2 thread cambiano 22 campi, AUC
    RMIA 0,5383 contro 0,5389). Thread da fissare e scrivere nel JSON.
72. **Partizione IID: membri rimescolati fra i client, non membri ancora per sito** (codice: righe
    7858-7875 contro 3774-3796). L'AUC delle celle IID mescola spostamento di distribuzione e
    memorizzazione.
73. **Minori di LiRA** (codice): con `common_init` gli shadow del round 1 partono da pesi casuali;
    `torch.manual_seed(_s)` (riga 4199) reimposta il generatore globale per gli attacchi successivi;
    l'avviso "shadow skippato" (circa riga 4055) e' falso, lo shadow resta e non si allena;
    `shadow_init: cold` nel ramo trainer toglie l'ancora FedProx che il client ha.
74. **DP per client con BatchNorm: `running_mean` e `running_var` escono senza taglio ne' rumore**
    (codice, `src/ml/gradient_manager.py` righe 535-548 e 610-622; `norm` di default `batch`). La
    garanzia formale non copre quei valori.
75. **I pesi di FedAvg (`n_samples`) dipendono dai dati** (`src/ml/autoencoder_trainer.py` righe 559
    e 575, `src/ml/fedavg_aggregator.py`); anche la loss del client esce senza protezione.
76. **Normalizzazione min-max calcolata sul training** (`normalize_sessions`, riga 1058): fuori dalla
    DP; in `mia_persona.py` le statistiche vengono anche dai membri (riga 203).
77. **DP per record: training con shuffle (`DataLoader`, riga 537 del trainer), accounting di
    Poisson** (codice). L'epsilon di Poisson non e' una garanzia; il limite valido con lo shuffle e'
    quasi vuoto (81 con sigma = 5).
78. **Sigma della DP per client: formula classica valida solo per epsilon < 1**
    (`gradient_manager.py` riga 223) mentre i config usano 16-256; adiacenza non dichiarata; la
    modalita' central e' corretta solo con pesi pubblici e adiacenza add/remove a denominatore fisso.
79. **Input non finiti aggirano il taglio** (misura del revisore: un `inf` rende NaN la somma): righe
    317 e 392 del trainer (`min(1.0, nan) == 1.0`), `row_valid` in `src/ml/subject_dp.py` riga 118.
80. **`record_dp.enabled` letto con `bool()`** (trainer riga 109): la stringa "false" lo accende.
81. **Soggetti della DP per soggetto**: le sessioni senza userID sono soggetti singoli (Caltech 12.068
    su 12.613, JPL 1.729 su 2.239, Office 1 879 su 901); per una persona senza userID la protezione
    vale per sessione.
82. **`scripts/mia_persona.py`** (misura del revisore): cache del riferimento indicizzata solo da seed,
    round ed epoche (righe 211-221), riusata con config o `--min-sessioni` diversi; intervallo per
    sessione con bootstrap sulle sessioni (righe 290-293: ampiezza 0,010 contro 0,096 per persona);
    BatchNorm nel braccio senza DP e GroupNorm nei bracci con DP (righe 68-81); niente epsilon nel
    JSON; riferimento addestrato su uno sfondo poco simile alle persone valutate.
83. **`scripts/canary_persona.py --combina`** (misura del revisore): non controlla braccio, round,
    epoche, numero di sessioni, feature ne' config (righe 175-179, 224-225, 297-300); il braccio A
    copiato come B da' un AUC 0,5 falso; i nomi dei file non contengono round ed epoche.
84. **Confronti multipli senza correzione** in `mia_persona.py` e `canary_persona.py`.

Falso allarme registrato: gli epsilon della DP per soggetto in `ESPERIMENTI.md` (6,5 / 19,3 / 34,6
con sigma = 2) sono giusti per il canary multisito (1000 epoche x 3 round); ricalcolati 6,55 /
19,33 / 34,56.

Stato delle voci 65-84 al 16 ottobre, sera (codice nel repository, non ancora committato; suite 574
test passati; run di prova su dati ACN reali, 2 round x 2 epoche, in tutte le modalita'): 65-70, 72, 73
corrette con LiRA e RMIA v2 (`src/plugins/attacks/lira2.py`, default; `lira.version: 1` per le run
vecchie); 71 numero di thread fissabile (`--threads`) e scritto nel JSON; 74 GroupNorm di default e
errore con DP per client e BatchNorm; 75 pesi di aggregazione = numerosita' dei client, pubbliche con
l'adiacenza per sostituzione, e loss trattenuta nel client nelle celle con DP; 76 limiti pubblici
(`LIMITI_PUBBLICI` in `scripts/run_experiments.py`, proposta da confermare); 77 DP per record con
campionamento di Poisson e vmap; 78 sigma dal meccanismo gaussiano analitico; 79-80 corrette; 81
profili di abitudine (`src/ml/profili_abitudini.py`), con la misura della voce 88; 82-84 corrette negli
script persona e canary.

## S. Seconda revisione indipendente delle correzioni (2026-10-16)

85. **Central DP: rumore meta' del necessario per l'adiacenza dichiarata** (misura del revisore): il
    FedAvg rinormalizza sui client presenti, quindi aggiungere o togliere un client cambia anche i
    pesi degli altri. Corretto: adiacenza per SOSTITUZIONE dei dati di un client in tutte le
    modalita', D = 2C per client e D = 2 max_i(w_i) C in central, controllo che tutti i client
    partecipino a ogni round.
86. **n e i pesi di aggregazione sono osservabili dal server**, quindi l'adiacenza add/remove non e'
    coerente per DP per record e per soggetto. Corretto: epsilon principale per SOSTITUZIONE di un
    record o di un soggetto (PLD, `ml.subject_dp.accountant`), add/remove (RDP) riportato per
    confronto. Esempio Office 1, sigma = 1, B = 32, 10 x 50: 55,8 contro 30,2.
87. **Minori corretti**: opzioni della LiRA v1 (`shadow_init: cold`, `floor_mode`, `member_scoring`,
    `canary_rescale`) bloccate prima del FL con LiRA v2; canary fuori dalle metriche principali;
    `composed_rmia_loglr_*` al posto di `composed_rmia_*` (somma dei log-rapporti); n di shadow
    eseguiti nel JSON; p di Mann-Whitney unilaterale al posto di 1000 permutazioni, metrica primaria
    senza correzione e Holm sulle secondarie; norma effettiva nel JSON; limiti per le feature assenti
    in Scozia e per `start_time_epoch`; matrici che riconoscono la LiRA v2; profili scalati sulle
    sessioni di training, errore se un sito non ha profili o se la normalizzazione non e' pubblica;
    hash del codice nella cache del riferimento persona; normalizzazione pubblica anche in
    `run_nvflare_mia.py` e `check_canary_init_confound.py`.
88. **Profili di abitudine: protezione debole per la persona vera** (misura, `scripts/misura_profili.py`,
    seed 42, persone con userID del training): con profili di 20-500 sessioni di training una persona
    (mediana 12 sessioni) cade in 5-9 profili, p90 17-51. Con sostituzione e 10 round x 50 epoche:
    sigma = 5 Caltech 7,8-10,9, JPL 10,1-11,2, Office 1 53-105; sigma = 2 Caltech 25-36, JPL 34-38,
    Office 1 oltre 200. Decisione aperta.
89. **Con IN della taglia del client gli shadow OUT per campione sono pochi**: universo membri +
    non membri, quota IN circa 0,85; con 8 shadow il 28% dei campioni non ha shadow OUT e non si
    valuta, con 32 circa lo 0,5%. Da fissare `lira.n_shadow: 32`.

## T. Controlli della campagna v2b (2026-10-19)

90. **Avviso fuorviante con la DP per soggetto** (`scripts/run_experiments.py`, avviso di avvio; `scripts/canary_persona.py`
    con `no_dp=True`): con `subject_dp` attiva compare "Rumore Differential Privacy DISABILITATO (sigma=0)". La riga
    `[SUBJECT-DP] ... sigma=5.0` mostra che il rumore e' attivo. Nessun effetto sui numeri; va corretto il testo
    dell'avviso.
91. **Raccolta dei risultati canary**: nella copia del raccoglitore del Mac di Domenico `canary_persona.csv` non e' stato
    scritto. Con la versione corrente di `raccogli_risultati.py` lo stesso zip produce il file (45 righe). Da
    riverificare sul Mac di Domenico con la copia aggiornata.
92. **Pacchetto senza dati**: `aggiornamento_v2b` non contiene `datasets/`; ogni run fallisce dopo circa 36 secondi con
    `FileNotFoundError: Nessun dataset trovato`. Da indicare nel LEGGIMI del pacchetto (copiare `datasets/acn`).

