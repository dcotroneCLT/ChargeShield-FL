# Indice dei config

> **Stato: documento OPERATIVO.** Una riga per file, aggiornato il 2026-09-30. Ogni
> config è una cella: prima di crearne uno nuovo cercare qui se esiste già. La base
> di tutti è `experiment.yaml`; le colonne dicono cosa cambia rispetto a quella base
> (3 siti, 10 round, 50 epoche, 6 feature, ε = 1, n_shadow 16, FedProx mu 0.01).
> `--epsilon`, `--rounds`, `--epochs`, `--seed`, `--dp-mode`, `--no-dp` da riga di
> comando hanno la precedenza sul file.

## Regime naturale: RQ1, RQ2, RQ3 (`docs/ESPERIMENTI.md`)

| file | cosa cambia | uso |
|---|---|---|
| `experiment.yaml` | niente: è la base | no-DP con `--no-dp`; client-level con `--dp-mode` ed `--epsilon` |
| `experiment_rq1_eps{2,4,8,16}.yaml` | solo ε | E-A, sweep di ε |
| `experiment_rq1_eps{32,64,256}.yaml` | solo ε | punto operativo client-level, C = 1: griglia a un seed in `_op_*`; ε = 64 anche a 5 seed in `rq1-eps64` |
| `experiment_rq1_C{0.25,0.5}_eps{16,64,256}.yaml` | ε e `experiment.max_grad_norm` (C) | punto operativo client-level; un seed, sweep-dir `_op_*` |
| `experiment_rq1_recorddp_nm{0,0.5,1,2,5}.yaml` | `record_dp` attivo con quel `noise_multiplier`; `nm0` = solo clipping per esempio, senza rumore (braccio B1) | E-B, **sempre con `--no-dp`** |
| `experiment_rq1_nodp_groupnorm.yaml` | solo `ml.norm: group` | E-B, riferimento no-DP con la normalizzazione delle celle record-DP; con `--no-dp`, cella "no-DP baseline, norm=group" |
| `experiment_rq2_per_site.yaml`, `experiment_rq2_iid.yaml` | solo `partition.strategy` | E-D, RQ2 |
| `experiment_rq2_{per_site,iid}_eps64.yaml` | `partition.strategy` ed ε = 64 (C = 1) | E-D, RQ2, braccio con DP al punto operativo |
| `experiment_rq3_mu0.yaml` | solo `ml.proximal_mu: 0.0` (FedAvg) | E-E, RQ3, con `--no-dp` |
| `experiment_rq3_mu0_eps64.yaml` | `ml.proximal_mu: 0.0` ed ε = 64 (C = 1) | E-E, RQ3, braccio con DP al punto operativo; il braccio FedProx appaiato è `experiment_rq1_eps64.yaml` |
| `experiment_rq3_mu0_eps64_C{2,4,8}.yaml` | come `experiment_rq3_mu0_eps64.yaml` con `max_grad_norm` = 2, 4, 8 | E-E, RQ3, screening del limite di clipping per FedAvg con DP (seed 42): C = 1 era stato scelto sulla griglia di FedProx |
| `experiment_rq3_mu0.001.yaml`, `experiment_rq3_mu0.1.yaml` | solo `ml.proximal_mu` (0.001, 0.1) | E-E, prove su mu: dose-risposta a 10 round, seed 42, `--no-dp`, sweep-dir `_rq3_*` |
| `experiment_rq3_mu0.01_r30.yaml` | solo `experiment.fl_rounds: 30` | E-E, prove su mu: FedProx a 30 round, seed 42, `--rounds 30 --no-dp`, sweep-dir `_rq3_*` |
| `experiment_ctrl_common_init.yaml` | solo `ml.common_init: true` | controllo del protocollo, segnalazione 48; no-DP, seed 42 |
| `experiment_robustness_entity_split.yaml` | `split.strategy: entity_aware` | robustezza dello split, già eseguito a 5 seed |
| `experiment_chargeplace_scotland.yaml` | secondo dataset, 3 council area | futuro; una sola run di prova |
| `experiment_sc3.yaml`, `experiment_sc3_mu0.yaml` | Scozia ridotta: Glasgow City, Dundee City, Shetland Islands, settembre-novembre 2023, quattro feature; FedProx e FedAvg | RQ3, replica ridotta (2026-10-14), con `--no-dp`, sola utilita' |
| `experiment_sc3_eps64.yaml`, `experiment_sc3_mu0_eps64.yaml` | come sopra, eps 64, C 1 | RQ3, DP completa e, con `--dp-ablation`, solo taglio e solo rumore |
| `experiment_sc3_mu0.001_eps64.yaml`, `experiment_sc3_mu0.1_eps64.yaml` | come sopra, mu 0.001 e 0.1 | RQ3, curva in mu sotto DP |
| `experiment_sc3_groupnorm.yaml` | come `experiment_sc3.yaml` con `ml.norm: group` | RQ1, riferimento della DP per record e per soggetto, `--no-dp` |
| `experiment_sc3_recorddp.yaml` | `ml.norm: group`, `ml.record_dp` C 1, sigma 1 | RQ1, DP per record sulla Scozia, `--no-dp` |
| `experiment_sc3_subjectdp.yaml` | `ml.norm: group`, `ml.subject_dp` (colonnina) C 1, sigma 5 | RQ1, DP per soggetto sulla Scozia, `--no-dp` |
| `experiment_cap32_16.yaml`, `experiment_cap64_32.yaml`, `experiment_cap128_64.yaml` | solo `ml.hidden_dims` e `ml.latent_dim` (1 870, 6 294, 22 822 parametri contro 618) | RQ1, curva di sensibilita' in capacita' (2026-10-14), `--no-dp`, LiRA e RMIA |
| `experiment_canary_multisite_gn.yaml`, `_subject_s0.yaml`, `_subject_s2.yaml` | da `experiment_canary_multisite.yaml` (generati da `scripts/genera_config_subject_clip.py`): `ml.norm: group`; nei due bracci per soggetto `ml.subject_dp` (userID, C 1, 32 soggetti attesi per passo, al piu' 64 sessioni) con sigma 0 (solo taglio) e 2 | DP per soggetto, controllo solo taglio (equivalente misurato di A2, 2026-10-15), con `scripts/canary_persona.py`, regime indotto |
| `experiment_acn4f.yaml`, `experiment_acn4f_mu0.yaml` | solo `ml.input_dim: 4` e `ml.feature_names` (le quattro feature comuni con la Scozia) rispetto a `experiment.yaml` e `experiment_rq3_mu0.yaml` | RQ3, controllo ACN a quattro feature (2026-10-14), con `--no-dp`, sola utilita' |
| `experiment_acn4f_eps64.yaml`, `experiment_acn4f_mu0_eps64.yaml` | come sopra rispetto a `experiment_rq1_eps64.yaml` e `experiment_rq3_mu0_eps64.yaml` | RQ3, controllo ACN a quattro feature: DP completa e, con `--dp-ablation`, solo taglio e solo rumore; sola utilita' |
| `experiment_scotland32_recorddp.yaml` | da `experiment_scotland_32client.yaml`: `ml.norm: group` e `ml.record_dp` (C = 1, sigma = 5) | RQ1, Scozia completa (2026-10-14), con `--no-dp`; circa 4-6 giorni a run; rinviato alla tesi (Scozia a 32 council) |
| `experiment_scotland32_mu0_subjectdp.yaml` | da `experiment_scotland_32client_subjectdp.yaml`: solo `ml.proximal_mu: 0.0` (FedAvg) | RQ1 e RQ3, Scozia completa (2026-10-14), con `--no-dp`; rinviato alla tesi (Scozia a 32 council) |

## Diagnostiche LiRA: non usare finché lo scorer è congelato

| file | cosa cambia |
|---|---|
| `experiment_floor_independent.yaml` | `lira.floor_mode: independent` |
| `experiment_matched_formula_scoring.yaml` | `lira.member_scoring: matched_formula` |
| `experiment_shadow_cold.yaml` | `lira.shadow_init: cold` |
| `experiment_hour_circular_encoding.yaml` | ora codificata in seno e coseno, Office 1, 3 round |

## Canary bilanciato: il protocollo che regge (`docs/CanaryPositiveControl.md`)

Tutti su Office 1, 3 round, 7 feature, capacità (32, 16)/8, k = 20 template membro e
20 non membro, 30 duplicati, `paired_split`, n_shadow 8, shadow cold. Lanciare
sempre la coppia base e swap con lo stesso seed.

| file | cosa cambia |
|---|---|
| `experiment_canary_balanced.yaml`, `_swap.yaml` | il riferimento, 1000 epoche, BatchNorm |
| `experiment_canary_e1000_group.yaml`, `_swap.yaml` | come sopra con GroupNorm, confrontabile con le celle record-DP |
| `experiment_canary_balanced_e100_group.yaml`, `experiment_canary_e100_group_swap.yaml` | 100 epoche, GroupNorm |
| `experiment_canary_e10_group.yaml` | 10 epoche, GroupNorm |
| `experiment_canary_balanced_nshadow32.yaml`, `_swap.yaml` | n_shadow 32, ablation |
| `experiment_canary_balanced_indepfloor.yaml` | floor indipendente, ablation |

## Canary su più siti (`docs/ESPERIMENTI.md`)

Stesso protocollo del canary bilanciato, con i tre siti reali; canary iniettati solo in
Office 1. LiRA legge l'update di Office 1, Yeom e Shadow il modello globale.

| file | cosa cambia |
|---|---|
| `experiment_canary_multisite.yaml`, `_swap.yaml` | solo `sites` (caltech, jpl, office1, come i config RQ) e `experiment.name`; prima run: seed 42, `--no-dp`, per verifica e tempi |
| `experiment_canary_balanced_caltech.yaml`, `_swap.yaml` | canary bilanciato su un secondo sito: solo Caltech, `canary.site: caltech`, `n_duplicates: 561` (amplificazione per record di Office 1); prima run: seed 42, `--no-dp`, per verifica e tempi |

## Canary con DP record-level (regime canary, `--no-dp` per il client-level)

| file | `noise_multiplier` | epoche | nota |
|---|---|---|---|
| `experiment_rdp_e1000_s5.yaml`, `_swap.yaml` | 5 | 1000 | la cella ε dichiarato 7.15, un seed; ε da ricalcolare (segnalazione 4) |
| `experiment_rdp_e1000_s10.yaml` | 10 | 1000 | |
| `experiment_rdp_auditor_tuned.yaml` | 5 | 1000 | come s5 più `auditor_tuned.yaml` |
| `experiment_canary_balanced_recorddp.yaml` | 1 | 1000 | |
| `experiment_canary_balanced_recorddp_fast.yaml` | 1 | 100 | |
| `experiment_canary_recorddp_nm01.yaml` | 0.1 | 100 | |
| `experiment_canary_recorddp_nm005.yaml`, `_swap.yaml` | 0.05 | 100 | |
| `experiment_rdp_eps163.yaml`, `experiment_rdp_eps474.yaml` | 2, 1 | 10 | |
| `experiment_canary_balanced_recorddp_nm5.yaml` | 5 | 5 | smoke test, non un esperimento |

## Canary superati: solo il braccio base è un controllo positivo, lo swap non è un controllo negativo

Gruppi sbilanciati, 5 template membro e 20 non membro: con questa forma lo scambio
dei ruoli estrae insiemi disgiunti e non è uno scambio. Restano come storia.

| file | sito | duplicati | nota |
|---|---|---|---|
| `experiment_canary_positive_control.yaml` | Office 1 | 30 | 6 feature, capacità storica |
| `experiment_canary_positive_control_caltech.yaml` | Caltech | 30 | segnale non riprodotto |
| `experiment_canary_positive_control_caltech_highdensity.yaml` | Caltech | 30 su 84 template | densità aggregata pari a Office 1, amplificazione per record no |
| `experiment_canary_positive_control_caltech_amplification_matched.yaml` | Caltech | 561 | amplificazione per record pari a Office 1 |
| `experiment_canary_positive_control_jpl.yaml` | JPL | 30 | segnale non riprodotto |
| `experiment_canary_positive_control_chargeplace_scotland.yaml` | Glasgow City | 4650, 50 epoche | mai eseguito |
| `experiment_canary_coldstart.yaml`, `_swap.yaml` | Office 1 | 30 | 6 feature, shadow cold |
| `experiment_canary_maxmemo.yaml`, `_swap.yaml` | Office 1 | 30 | 7 feature, capacità (32, 16)/8 |
| `experiment_canary_swap_control.yaml` | Office 1 | 30 | swap sbilanciato |

## Calibrazione della memorizzazione naturale: chiusa, cinque assi tutti nulli

`experiment_overfit_calibration.yaml` (Office 1, 50 epoche), `_bigcap` (capacità
3.3x, 1000 epoche), `_richfeat` (7 feature), `_combined` (capacità e feature),
`_altsite` (Caltech 2021). Servivano a stabilire se il modello può memorizzare
senza canary: no.

## Auditor

`auditor.yaml`: parametri del Privacy Auditor con `total_rounds_budget: 1000`, per
cui le soglie di budget non scattano mai in 10 round. `auditor_tuned.yaml`: budget
su 3 round, selezionabile con `CHARGESHIELD_AUDITOR_CONFIG`. Il campo `attacks` è
storico e non seleziona nulla.

## Infrastruttura e legacy

`flare.yaml` resta in questa cartella solo perché lo scaffolding non collegato
`src/flare/flare_connector.py` e il suo test lo cercano qui: i suoi valori non sono
quelli degli esperimenti. `_legacy_unused/` contiene i config del progetto iniziale a
12 nodi e 4 cluster mai costruito (`clusters.yaml`, `nodes.yaml`, `nodes/`,
`protocols.yaml`, `framework.yaml`, `datasets.yaml`) e il vecchio
`experiment_overfit_control.yaml`, che è solo commenti. Non usarli.
