#!/usr/bin/env python3
"""
Fase A della guida scientifica (ChargeShield_FL_spina_dorsale_consolidata.md,
sezione 7, righe 148-154): genera i TRE output previsti.

    risultati/matrice_run_completati.xlsx  -> registro run
    risultati/matrice_confronti.xlsx       -> matrice di confrontabilita'
    risultati/Matrice_sintesi.xlsx         -> lacune effettive + minimo rerun

REGOLA NON NEGOZIABILE della guida (riga 150):
    "Nessun valore mancante va riempito per analogia."
Quindi: commit e versione/hash degli split NON sono registrati nei JSON di
questo progetto e restano VUOTI, con la dicitura "non registrato". Non si
deducono dalla data, dal nome della cartella o da altre run.

Gli stati di validita' ammessi sono i cinque della guida:
    verificata | completata da verificare | invalidata | incompleta | pianificata

Uso:
    python3 scripts/genera_matrici_faseA.py
"""
from __future__ import annotations
import glob
import json
import os
import math
import re
from collections import defaultdict
from datetime import datetime

import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side

RADICE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys  # noqa: E402
sys.path.insert(0, os.path.join(RADICE, "scripts"))
from check_significance import discover_groups  # noqa: E402
from etichetta_cella import etichetta_cella  # noqa: E402


def composizione_celle() -> dict[str, list[str]]:
    """Celle e run che le compongono, con la STESSA regola di check_significance.py
    (segnalazioni 44, 45, 53, corrette il 2026-09-24): etichetta dai campi
    registrati nel JSON (etichetta_cella.py), cartelle `_*`, `entity-split*`,
    `nvflare-*` e `fedmia-gradient-*` escluse, una run per (cella, seed), la piu'
    recente. Prima i fogli di questo script raggruppavano per (dp_mode, epsilon)
    TUTTI i JSON, senza esclusioni ne' deduplica: la cella no-DP del foglio
    Utility_privacy_limite conteneva 50 run invece delle 5 di nodp-sweep2.
    Restituisce {etichetta: [percorsi assoluti]}."""
    return dict(discover_groups(os.path.join(RADICE, "experiments", "*", "experiment_*.json")))
USCITA = os.path.join(RADICE, "risultati")
# Run di altre macchine analizzate a parte (segnalazione 45): oggi E-D, RQ2.
ALTRE_MACCHINE = os.path.join(RADICE, "experiments_altre_macchine")
FONT = "Arial"

# Commit del fix che rende la loss grezza dei canary indipendente dal filtro
# LiRA (Sprint 10zz+119). I JSON canary precedenti hanno canary_raw_mse_auc_roc
# calcolata su un pool filtrato dalla calibrazione shadow: la metrica c'e' ma
# non e' confrontabile con i baseline, che girano senza shadow.
STACCO_FIX_RAW = datetime(2026, 9, 16, 21, 0)

VERDE = PatternFill("solid", fgColor="E2EFDA")
GIALLO = PatternFill("solid", fgColor="FFF2CC")
ROSSO = PatternFill("solid", fgColor="FCE4E4")
GRIGIO = PatternFill("solid", fgColor="EDEDED")
BORDO = Border(*[Side(style="thin", color="BFBFBF")] * 4)


# ─────────────────────────────────────────────────────────────────────────────
# Lettura delle evidenze
# ─────────────────────────────────────────────────────────────────────────────
def mappa_log() -> dict[str, list[str]]:
    """Associa il nome di uno sweep ai file di log che lo nominano."""
    m = defaultdict(list)
    for p in glob.glob(os.path.join(RADICE, "logs", "*.log")):
        nome = os.path.basename(p)
        chiave = re.sub(r"\.log$", "", nome)
        m[chiave].append(os.path.join("logs", nome))
    return m


def trova_log(sweep: str, log_disponibili: dict[str, list[str]]) -> str:
    """Log plausibile per uno sweep. Solo corrispondenze di nome, mai inferenze."""
    s = sweep.lstrip("_")
    esatti = [v for k, vs in log_disponibili.items() if k == s for v in vs]
    if esatti:
        return "; ".join(sorted(esatti))
    parziali = [v for k, vs in log_disponibili.items()
                if s and (s in k or k in s) for v in vs]
    return "; ".join(sorted(set(parziali))) if parziali else ""


def parametri_canary_dai_log() -> dict[str, dict]:
    """k, duplicati e swap letti dai LOG, non dedotti.

    Risolve le run canary anteriori allo Sprint 10zz+113, che non salvano il
    blocco 'canary' nel JSON. La riga di log
        "[CANARY] Iniettati K template x D duplicati ... + N gemelli non-membro"
    contiene entrambe le cardinalita' in chiaro, e "[SWAP ATTIVO]" dice se il
    braccio e' quello scambiato. E' lettura di un'evidenza, non un'analogia:
    il log e' l'output della run stessa.
    """
    pat = re.compile(
        r"Iniettati (\d+) template . (\d+) duplicati.*?\+ (\d+) gemelli non-membro"
    )
    fuori = {}
    for p in glob.glob(os.path.join(RADICE, "logs", "*.log")):
        txt = open(p, errors="replace").read()
        m = pat.search(txt)
        if not m:
            continue
        k, dup, nnm = int(m.group(1)), int(m.group(2)), int(m.group(3))
        fuori[os.path.basename(p)] = {
            "k": k, "dup": dup, "nnm": nnm,
            "swap": "[SWAP ATTIVO]" in txt,
            "bilanciato": k == nnm,
        }
    return fuori


def canary_dal_log(sweep: str, par: dict[str, dict]) -> dict | None:
    """Parametri canary per uno sweep, se un log con quel nome esiste."""
    s = sweep.lstrip("_")
    for nome, v in par.items():
        if re.sub(r"\.log$", "", nome) == s:
            return v
    return None


def superficie(cfg: dict) -> str:
    """Punto di osservazione dell'avversario (sezione 5 della guida)."""
    rd = cfg.get("record_dp") or {}
    if rd.get("enabled"):
        # Segnalazione 56: una run record-DP ha no_dp=True (spegne il client-level) e
        # finiva in A0, "nessun clipping, nessun rumore", mentre l'update viene da DP-SGD.
        return (f"record-level — update da DP-SGD (clipping per esempio C = "
                f"{rd.get('max_grad_norm')}, sigma = {rd.get('noise_multiplier')}), nessun "
                f"meccanismo client-level")
    if cfg.get("no_dp"):
        return "A0 — update non privatizzato (nessun clipping, nessun rumore)"
    return {
        "dp-fedavg": "A1 — update grezzo g (limite superiore, oltre lo Scenario 1)",
        "central": "A2 — update clippato g_bar (Scenario 1)",
        "local": "A3 — update clippato e rumorizzato g_tilde (Scenario 1)",
    }.get(cfg.get("dp_mode"), "non determinabile dal config")


def regime(cfg: dict, sweep: str) -> str:
    can = cfg.get("canary") or {}
    if can.get("enabled") or "canary" in sweep.lower():
        return "canary (indotto)"
    if (cfg.get("record_dp") or {}).get("enabled"):
        # Fino al 2026-09-26 "(diagnostico)": le run record-DP erano solo prove. Con la
        # campagna E-B sono celle di RQ1 (segnalazione 56).
        return "naturale — record-DP"
    return "naturale"


def rq_di(cfg: dict, sweep: str, reg: str) -> str:
    """Mappa sulle RQ della guida, sezione 4."""
    if reg.startswith("canary"):
        return "CTRL — validazione dello strumento, non una RQ"
    if cfg.get("no_dp") is not None or cfg.get("dp_mode"):
        return "RQ1"
    return "non classificabile"


def stato_validita(cfg: dict, sweep: str, reg: str, ts: datetime,
                   mia: dict, dal_log: dict | None = None) -> tuple[str, str]:
    """Stato + motivo. Solo criteri oggettivi e verificabili nel file stesso.

    Il criterio "post-fix" NON usa il timestamp, che sarebbe una congettura:
    usa la presenza del campo canary_raw_n_member_distinct, introdotto dal fix
    stesso (Sprint 10zz+119). Se il campo c'e', la loss grezza e' stata
    raccolta prima del filtro di calibrazione LiRA; se non c'e', no.
    """
    can = cfg.get("canary") or {}
    e_canary = reg.startswith("canary")
    # LiRA v2 (2026-10-16) calcola la loss grezza dei canary su tutti i canary,
    # senza filtro di calibrazione: e' pulita per costruzione.
    raw_pulita = (mia.get("canary_raw_n_member_distinct") is not None
                  or mia.get("lira_version") == 2)

    if e_canary and not can:
        if dal_log:
            k, nnm, sw = dal_log["k"], dal_log["nnm"], dal_log["swap"]
            if sw and not dal_log["bilanciato"]:
                return ("invalidata",
                        f"RISOLTO DAL LOG (2026-09-21): braccio di scambio con gruppi "
                        f"sbilanciati, k_membro={k} contro k_nonmembro={nnm}. Con "
                        f"cardinalita' diverse i due bracci pescano insiemi di membri "
                        f"DISGIUNTI, quindi non e' un controllo negativo (guida sez. 6; "
                        f"docs/CanaryPositiveControl.md 4.1). Il config non era nel JSON, "
                        f"ma la riga '[CANARY] Iniettati ...' del log lo dichiara")
            return ("completata da verificare",
                    f"il blocco 'canary' non e' nel JSON, ma il log risolve il disegno: "
                    f"k_membro={k}, duplicati={dal_log['dup']}, k_nonmembro={nnm}, "
                    f"swap={'si' if sw else 'no'}, "
                    f"{'bilanciato' if dal_log['bilanciato'] else 'SBILANCIATO'}. "
                    f"Il braccio base resta un positive control leggibile; la coppia con "
                    f"lo swap no, se sbilanciata")
        return ("incompleta",
                "il blocco 'canary' non e' registrato nel config del JSON (i run "
                "anteriori allo Sprint 10zz+113 non lo salvavano) e NON esiste un log "
                "con il nome di questo sweep: k, duplicati e swap_assignment non sono "
                "ricavabili da nessuna evidenza. Dedurli dal nome della cartella "
                "sarebbe l'analogia che la guida vieta")

    if can.get("enabled"):
        nt, nnm = can.get("n_templates"), can.get("n_nonmember_templates")
        if can.get("swap_assignment") and nt is not None and nt != nnm:
            return ("invalidata",
                    f"braccio di scambio con gruppi sbilanciati ({nt} vs {nnm}): i due "
                    "bracci pescano insiemi di membri DISGIUNTI, quindi non e' un "
                    "controllo negativo (guida sez. 6; docs/CanaryPositiveControl.md 4.1)")
        if not raw_pulita:
            return ("completata da verificare",
                    "canary anteriore al fix Sprint 10zz+119 (manca il campo "
                    "canary_raw_n_member_distinct): canary_raw_mse_auc_roc e' calcolata "
                    "su un pool filtrato da insufficient_calibration di LiRA, quindi non "
                    "e' appaiata con la baseline a init casuale, che gira senza shadow")
        if re.fullmatch(r"_canary_balanced(_swap)?_s\d+", sweep):
            return ("verificata",
                    "campagna bilanciata post-fix: numeri ricontrollati dai JSON il "
                    "2026-09-21 (30/30 round sopra 0.5, min 0.6125, t(4)=9.90 appaiato "
                    "sui 5 seed, 400 coppie distinte in tutte le celle)")
        return ("completata da verificare",
                "cella di ablation post-fix: il dato e' pulito ma non e' stato "
                "sottoposto alla stessa verifica numerica della campagna principale")

    if not mia:
        return ("incompleta", "nessuna metrica MIA nel JSON")
    gc = cfg.get("git_commit")
    if gc and gc.endswith("-dirty"):
        # Segnalazione 57: il marcatore dice che al salvataggio l'albero aveva modifiche
        # non committate, non quali. Si chiarisce con git diff fra quel commit e il successivo.
        return ("completata da verificare",
                f"commit {gc[:7]} con modifiche non committate al salvataggio (-dirty): "
                f"verificare con git diff che non toccassero il codice eseguito; hash degli "
                f"split non registrato")
    if gc:
        return ("completata da verificare",
                f"nessun difetto noto; commit {gc[:7]} registrato, hash degli split non "
                f"registrato")
    return ("completata da verificare",
            "nessun difetto noto, ma commit e hash degli split non sono registrati: "
            "la tracciabilita' completa richiesta dalla Fase A non e' ricostruibile")


def leggi_run() -> list[dict]:
    log_disp = mappa_log()
    par_canary = parametri_canary_dai_log()
    righe = []
    # Segnalazione 56: fino al 2026-09-26 il registro leggeva solo experiments/ e le run
    # di altre macchine (E-D, la prova E-B), usate nei fogli, non comparivano.
    fs = [("experiments", f) for f in sorted(glob.glob(
        os.path.join(RADICE, "experiments", "*", "experiment_*.json")))]
    fs += [("experiments_altre_macchine", f) for f in sorted(glob.glob(
        os.path.join(ALTRE_MACCHINE, "*", "experiment_*.json")))]
    for radice_run, f in fs:
        pref = "" if radice_run == "experiments" else "altre_macchine: "
        sweep = os.path.basename(os.path.dirname(f))
        base = os.path.basename(f)
        try:
            d = json.load(open(f))
        except Exception as e:
            righe.append({"id": f"{pref}{sweep}/{base}", "errore": type(e).__name__})
            continue
        cfg = d.get("config") or {}
        summ = d.get("summary") or {}
        pr = d.get("per_round") or {}
        ultimo = max(pr, key=int) if pr else None
        mia = (pr.get(ultimo) or {}).get("mia", {}) if ultimo else {}
        try:
            ts = datetime.strptime(d.get("timestamp", ""), "%Y%m%d_%H%M%S")
        except Exception:
            ts = None
        reg = regime(cfg, sweep)
        stato, motivo = stato_validita(cfg, sweep, reg, ts, mia,
                                       canary_dal_log(sweep, par_canary))
        m = re.search(r"seed(\d+)", sweep)
        seed_nome = m.group(1) if m else None
        discorde = seed_nome is not None and str(cfg.get("seed")) != seed_nome
        can = cfg.get("canary") or {}
        righe.append({
            "id": f"{pref}{sweep}/{base}",
            "sweep": sweep,
            "rq": rq_di(cfg, sweep, reg),
            "commit": "",
            "configurazione": (
                f"dp_mode={cfg.get('dp_mode')}; no_dp={cfg.get('no_dp')}; "
                f"eps={cfg.get('epsilon')}; delta={cfg.get('delta')}; "
                f"round={cfg.get('fl_rounds')}; epoche={cfg.get('epochs')}; "
                f"batch={cfg.get('batch_size')}; feature={len(cfg.get('feature_names') or []) or 'n.d.'}; "
                f"hidden={cfg.get('hidden_dims')}; latent={cfg.get('latent_dim')}"
                + (f"; canary k={can.get('n_templates')}/{can.get('n_nonmember_templates')}, "
                   f"dup={can.get('n_duplicates')}, swap={can.get('swap_assignment')}"
                   if can.get("enabled") else "")
                # Fix 2026-09-22 (segnalazione 5/37): una run record-DP ha
                # no_dp=True (serve a disattivare il client-level) e senza
                # questo campo la stringa "configurazione" la mostrava
                # indistinguibile da una vera baseline no-DP, anche se la
                # colonna "regime" già la marcava "naturale — record-DP
                # (diagnostico)" (funzione regime() sopra). Reso esplicito
                # qui per chi legge solo questa colonna.
                + (f"; record_dp: enabled={ (cfg.get('record_dp') or {}).get('enabled') }, "
                   f"nm={ (cfg.get('record_dp') or {}).get('noise_multiplier') }, "
                   f"eps_record_dp={cfg.get('epsilon_record_dp')}"
                   if cfg.get("record_dp") else "")
            ),
            "hash_split": "",
            "seed": cfg.get("seed"),
            "seed_nome": seed_nome,
            "seed_discorde": discorde,
            # Segnalazione 56: con mu = 0 (E-E) il test di verita' dava "non registrato".
            "algoritmo": ("non registrato" if cfg.get("proximal_mu") is None else
                          f"FedAvg, mu={cfg.get('proximal_mu')}" if cfg.get("proximal_mu") == 0
                          else f"FedProx, mu={cfg.get('proximal_mu')}"),
            "regime": reg,
            "superficie": superficie(cfg),
            "dp_accounting": (
                (f"record-DP, accountant RDP (dp-accounting): eps per record di Poisson, massimo "
                 f"sui client = {cfg.get('epsilon_record_dp')}; per client "
                 f"{cfg.get('epsilon_record_dp_per_client')}; limite con lo shuffle, "
                 f"adiacenza per sostituzione = {cfg.get('epsilon_record_dp_shuffle_bound')}; "
                 f"delta={cfg.get('delta')}; nessun enforcement")
                if (cfg.get("record_dp") or {}).get("enabled") else
                (f"eps_naive={cfg.get('epsilon_cumulative_naive')}; "
                 f"eps_avanzata={cfg.get('epsilon_cumulative_advanced')}; "
                 f"eps_migliore={cfg.get('epsilon_cumulative_best_known')}; "
                 "nessun accountant RDP, nessun enforcement")
            ),
            "checkpoint": f"{len(pr)} round valutati",
            "metriche": (
                f"LiRA medio={summ.get('mean_lira_auc_roc')}; "
                f"Yeom medio={summ.get('mean_auc_roc')}; "
                f"Shadow medio={summ.get('mean_shadow_auc_roc')}; "
                f"LiRA composto={mia.get('composed_lira_auc_roc')}"
                + (f"; canary raw={mia.get('canary_raw_mse_auc_roc')}"
                   if mia.get("canary_raw_mse_auc_roc") is not None else "")
            ),
            "log": trova_log(sweep, log_disp),
            "artefatti": f"{radice_run}/{sweep}/ (JSON + xlsx + history/)",
            "stato": stato,
            "motivo": motivo,
            "ts": ts,
        })
    return righe


# ─────────────────────────────────────────────────────────────────────────────
# Scrittura
# ─────────────────────────────────────────────────────────────────────────────
def stile(ws, larghezze, altezza=100):
    for r in range(1, ws.max_row + 1):
        for c in range(1, ws.max_column + 1):
            cel = ws.cell(r, c)
            cel.font = Font(name=FONT, size=9, bold=(r == 1))
            cel.alignment = Alignment(wrap_text=True, vertical="top")
            cel.border = BORDO
            if r == 1:
                cel.fill = GRIGIO
    for col, w in zip("ABCDEFGHIJKLMNO", larghezze):
        ws.column_dimensions[col].width = w
    ws.freeze_panes = "A2"
    for r in range(2, ws.max_row + 1):
        ws.row_dimensions[r].height = altezza


def colora_stato(ws, col):
    for r in range(2, ws.max_row + 1):
        v = str(ws.cell(r, col).value or "")
        if v.startswith("verificata"):
            ws.cell(r, col).fill = VERDE
        elif v.startswith("invalidata") or v.startswith("incompleta"):
            ws.cell(r, col).fill = ROSSO
        elif v:
            ws.cell(r, col).fill = GIALLO


def scrivi_registro(righe):
    p = os.path.join(USCITA, "matrice_run_completati.xlsx")
    wb = openpyxl.load_workbook(p)
    ws = wb["Matrice_run"]
    # Colonne "commit" e "versione/hash degli split" RIMOSSE il 2026-09-21 su
    # decisione dell'utente: erano vuote su tutte le 235 righe perche' non sono
    # mai state registrate. Da oggi run_experiments.py salva config.git_commit,
    # quindi le run FUTURE saranno tracciabili e la colonna potra' tornare.
    for c, v in enumerate(["ID run", "RQ", "configurazione", "seed", "algoritmo/mu",
                           "regime naturale o canary", "superficie dell'attacco",
                           "DP/accounting", "checkpoint", "metriche", "log",
                           "artefatti", "stato di validita'",
                           "fase (spina dorsale)", "motivo dello stato"], start=1):
        ws.cell(1, c).value = v
    for c in range(16, 18):
        ws.cell(1, c).value = None
    for i, r in enumerate(righe, start=2):
        note_seed = (f"{r['seed']} (cartella dice {r['seed_nome']}: DISCORDE, "
                     "usare il nome cartella)") if r.get("seed_discorde") else r.get("seed")
        for c, v in enumerate([
            r["id"], r["rq"], r["configurazione"],
            note_seed, r["algoritmo"], r["regime"],
            r["superficie"], r["dp_accounting"], r["checkpoint"], r["metriche"],
            r["log"] or "nessun log associabile per nome", r["artefatti"],
            r["stato"], "Fase A — inventario", r["motivo"],
        ], start=1):
            ws.cell(i, c).value = v
    stile(ws, [34, 12, 52, 20, 18, 18, 30, 34, 14, 40, 30, 30, 22, 16, 50], 90)
    colora_stato(ws, 13)

    lg = wb["Legenda_stati"]
    for i, (s, d) in enumerate([
        ("verificata", "Numeri ricontrollati dai JSON da una seconda persona/passata, e nessun difetto noto sulla pipeline che li ha prodotti."),
        ("completata da verificare", "La run e' terminata e le metriche esistono, ma la tracciabilita' richiesta dalla Fase A non e' completa: commit e hash degli split non sono registrati in nessun JSON di questo progetto."),
        ("invalidata", "Esiste un motivo documentato per cui la run non risponde alla domanda per cui e' stata eseguita. Conservata, non cancellata (guida riga 152)."),
        ("incompleta", "La run manca di metriche o si e' interrotta."),
        ("pianificata", "Non ancora eseguita. Nessuna run in questo registro ha questo stato: le lacune stanno in Matrice_sintesi."),
        ("", ""),
        ("NOTA su 'commit' e 'hash split'", "Colonne RIMOSSE il 2026-09-21. La guida (riga 150) le elenca, ma erano vuote su tutte le 235 righe: non sono mai state salvate nei JSON, e dedurle dalla data sarebbe l'analogia che la guida vieta. Due colonne vuote non informano. Da oggi run_experiments.py registra config.git_commit (con marcatore -dirty se l'albero aveva modifiche non committate), quindi le run future saranno tracciabili e la colonna potra' tornare con dati veri."),
    ], start=2):
        lg.cell(i, 1).value = s
        lg.cell(i, 2).value = d
    stile(lg, [30, 110], 60)
    wb.save(p)
    return p, ws.max_row - 1


def scrivi_confronti(righe):
    p = os.path.join(USCITA, "matrice_confronti.xlsx")
    wb = openpyxl.load_workbook(p)
    ws = wb["Matrice_confronti"]
    ws.cell(1, 12).value = "fase (spina dorsale)"

    # Segnalazioni 44, 45, 53 (2026-09-24): celle con l'etichetta e la
    # composizione di check_significance.py, non piu' ricavate dalla stringa
    # "configurazione" su tutte le run del registro.
    per_id = {r["id"]: r for r in righe if "errore" not in r}
    celle = defaultdict(list)
    for et, files in composizione_celle().items():
        for f in files:
            r = per_id.get(f"{os.path.basename(os.path.dirname(f))}/{os.path.basename(f)}")
            # 2026-09-27: anche le celle record-DP di E-B (regime "naturale — record-DP"),
            # escluse finche' erano prove a un seed.
            if r and r["regime"] in ("naturale", "naturale — record-DP") and r["rq"] == "RQ1":
                celle[et].append(r)

    conf = []
    nodp = celle.get("no-DP baseline", [])
    n = 0
    for chiave, rs in sorted(celle.items(), key=str):
        if chiave == "no-DP baseline":
            continue
        n += 1
        seed_dp = {str(r["seed_nome"] or r["seed"]) for r in rs}
        seed_nodp = {str(r["seed_nome"] or r["seed"]) for r in nodp}
        appaiati = sorted(seed_dp & seed_nodp)
        rec = chiave.startswith("record-DP")
        norm_ref = chiave.startswith("no-DP") and "norm=group" in chiave
        conf.append([
            f"C{n:02d}", "RQ1", "no-DP (regime naturale)", chiave,
            "si, se appaiati per seed" if appaiati else "no: nessun seed in comune",
            ("dati, split, algoritmo, round, epoche, batch: identici; l'architettura no "
             "(GroupNorm al posto di BatchNorm, segnalazione 49)" if (rec or norm_ref) else
             "dati, split, architettura, algoritmo, round, epoche, batch: identici"),
            ("normalizzazione GroupNorm e clipping per esempio: il rapporto sul no-DP "
             "mescola rumore e architettura finche' manca una cella no-DP con GroupNorm; "
             "unita' protetta record, da confrontare col client-level a costo comparabile, "
             "non allo stesso eps (ESPERIMENTI.md, E-B)" if rec and appaiati else
             "nessuna differenza non controllata individuata oltre al fattore DP"
             if appaiati else "manca il riferimento appaiato"),
            f"{len(rs)} run nella cella DP, {len(nodp)} nella cella no-DP; "
            f"seed appaiabili: {appaiati or 'nessuno'}",
            "una run per seed, la piu' recente, con la regola di check_significance.py "
            "(segnalazioni 44, 45, 53); la cella puo' unire sweep diversi con lo stesso "
            "trattamento registrato",
            "confronto disponibile" if appaiati else "confronto non costruibile",
            ("Nessuna DP: e' il riferimento con la normalizzazione delle celle record-DP "
             "(E-B), non un contrasto della guida." if norm_ref else
             "Il contrasto B0/B2 della guida (sez. 8) e' questo. Manca invece il "
             "braccio B1 'clipping senza rumore', mai eseguito."),
            "Fase A — confrontabilita' (il confronto si esegue in Fase B)",
        ])

    _rq2 = leggi_rq2()
    if _rq2:
        _s = sorted(set(_rq2["rq2-per_site"]) & set(_rq2["rq2-iid"]))
        _c = sorted({j["config"].get("git_commit", "")[:7] for b in _rq2.values() for j in b.values()})
        conf.append([
            "C-RQ2", "RQ2", "partizione IID", "partizione per sito (non-IID)",
            "si, appaiati per seed",
            "dati, split globale, numerosita' per client, architettura, algoritmo, round, "
            "epoche, macchina e commit identici: cambia solo partition.strategy",
            "la partizione IID rompe corrispondenza client-sito, coerenza temporale e "
            "separazione fra utenti: e' un riferimento sperimentale, non un deployment",
            f"{len(_s)} seed per braccio, senza DP, seconda macchina "
            f"(experiments_altre_macchine/, commit {', '.join(_c)})",
            "una run per seed, la piu' recente",
            "confronto disponibile",
            "Numeri nel foglio RQ2_partizione di Matrice_sintesi.xlsx. " +
            ("Braccio con DP (eps = 64, C = 1) eseguito sulla stessa macchina."
             if leggi_rq2(con_dp=True) else
             "Manca il braccio con DP: stessa coppia al punto operativo client-level."),
            "Fase E — " + ("bracci senza DP e con DP eseguiti" if leggi_rq2(con_dp=True)
                           else "braccio senza DP eseguito, braccio con DP da eseguire"),
        ])
    else:
        conf.append([
            "C-RQ2", "RQ2", "partizione IID", "partizione non-IID (per sito)",
            "no", "—",
            "il braccio IID non e' in questo checkout (experiments_altre_macchine/ assente)",
            "—", "—", "confronto non costruibile",
            "Copiare E-D in experiments_altre_macchine/ (ESPERIMENTI.md, E-D).",
            "Fase E",
        ])
    # 2026-09-27: conteggi dal registro, non piu' il testo fisso del 2026-09-21
    # ("0.01 su 235 run su 235"), rimasto invariato mentre le run crescevano.
    _mu = defaultdict(int)
    for r in righe:
        if "errore" not in r:
            _mu[r["algoritmo"]] += 1
    _n_mu0 = sum(v for k, v in _mu.items() if k.startswith("FedAvg"))
    _rq3 = leggi_rq3()
    if _rq3:
        _s3, _st3 = statistiche_rq3(_rq3)
        _c3 = sorted({j["config"].get("git_commit", "")[:7] for b in _rq3.values() for j in b.values()})
        _h3 = _st3["loss holdout del modello rilasciato"]
        conf.append([
            "C-RQ3", "RQ3", "FedAvg (mu=0)", "FedProx (mu=0.01)",
            "si, appaiati per seed",
            "dati, split, partizione, architettura, round, epoche, macchina e commit identici: "
            "cambia solo ml.proximal_mu",
            "i due bracci senza DP hanno girato in parallelo sulla stessa macchina; la loss di "
            "addestramento locale di FedProx include il termine prossimale (segnalazione 59)",
            f"{len(_s3)} seed per braccio, senza DP, terza macchina (experiments_altre_macchine/, "
            f"commit {', '.join(_c3)})",
            "una run per seed, la piu' recente",
            "confronto disponibile",
            f"Numeri nel foglio RQ3_algoritmo di Matrice_sintesi.xlsx. Loss sull'holdout FedAvg "
            f"contro FedProx {sum(_h3['fedavg']) / len(_s3):.6f} contro "
            f"{sum(_h3['fedprox']) / len(_s3):.6f} (t = {_h3['t']:.2f}). " +
            ("Braccio con DP eseguito." if leggi_rq3(con_dp=True) else
             "Braccio con DP (eps = 64, C = 1) in corso o da eseguire."),
            "Fase D — " + ("bracci senza DP e con DP eseguiti" if leggi_rq3(con_dp=True)
                           else "braccio senza DP eseguito"),
        ])
    else:
        conf.append([
            "C-RQ3", "RQ3", "FedAvg (mu=0)", "FedProx (mu=0.01)",
            "no", "—",
            ("il braccio mu=0 non e' in questo checkout (E-E gira su un'altra macchina e "
             "andra' in experiments_altre_macchine/)" if not _n_mu0 else
             f"{_n_mu0} run con mu = 0 nel registro: confronto da costruire"),
            "algoritmo registrato nel registro run: " +
            "; ".join(f"{k}: {v}" for k, v in sorted(_mu.items())),
            "—", "confronto non costruibile" if not _n_mu0 else "confronto da costruire",
            "Da eseguire da zero riusando split, candidati, checkpoint e condizioni DP "
            "della Fase B (guida Fase D).",
            "Fase A — lacuna rilevata (si esegue in Fase D)",
        ])

    for i, riga in enumerate(conf, start=2):
        for c, v in enumerate(riga, start=1):
            ws.cell(i, c).value = v
    stile(ws, [12, 10, 26, 30, 22, 34, 42, 42, 34, 24, 46, 30], 110)
    colora_stato(ws, 10)
    for r in range(2, ws.max_row + 1):
        if "non costruibile" in str(ws.cell(r, 10).value):
            ws.cell(r, 10).fill = ROSSO

    lg = wb["Legenda_stati"]
    for i, (s, d) in enumerate([
        ("confronto disponibile", "Esistono run appaiabili per seed su entrambi i bracci. Non significa che il confronto sia gia' stato fatto ne' che sia valido: significa che i dati ci sono."),
        ("confronto non costruibile", "Manca del tutto uno dei due bracci. Nessuna analisi sui dati esistenti puo' produrlo."),
        ("", ""),
        ("NOTA", "La guida (riga 152) chiede di segnalare le configurazioni duplicate. Dal 2026-09-24 le celle hanno una run per seed, la piu' recente, con la regola di check_significance.py; i duplicati restano visibili nel registro run (segnalazioni 9 e 44)."),
    ], start=2):
        lg.cell(i, 1).value = s
        lg.cell(i, 2).value = d
    stile(lg, [30, 110], 60)
    wb.save(p)
    return p, len(conf)


def _testo_rq2_dp() -> str:
    """Una frase sul braccio con DP di RQ2, dai dati se ci sono."""
    d = leggi_rq2(con_dp=True)
    if not d:
        return "Manca il braccio con DP."
    s, st = statistiche_rq2(d)
    h = st["loss holdout del modello rilasciato"]; n = len(s)
    return (f"Con DP (eps = {RQ2_EPS_DP:g}, C = 1): loss sull'holdout IID contro per sito "
            f"{sum(h['iid']) / n:.6f} contro {sum(h['per_site']) / n:.6f} (t = {h['t']:.2f}); "
            f"attacchi al caso in entrambi.")


def riga_rq2_sintesi(n_tot: int) -> list:
    """Riga RQ2 della Matrice_sintesi, dai dati di E-D se ci sono."""
    rq2 = leggi_rq2()
    if not rq2:
        return [f"{n_tot} run totali", "RQ2", "lacuna strutturale", "C-RQ2", "no",
                "Il braccio IID di E-D non e' in questo checkout.",
                "IID/non-IID x senza-DP/con-DP: 4 celle x 5 seed = 20 run.",
                "media", "Partizione IID = riferimento sperimentale, non un deployment.",
                "Fase A -> E"]
    s, st = statistiche_rq2(rq2)
    h = st["loss holdout del modello rilasciato"]; y = st["Yeom, AUC all'ultimo round"]
    n = len(s)
    _dp = leggi_rq2(con_dp=True)
    _ndp = 2 * len(statistiche_rq2(_dp)[0]) if _dp else 0
    return [(f"{2 * n + _ndp} run E-D, senza DP e con DP (seconda macchina)" if _ndp else
             f"{2 * n} run E-D senza DP (seconda macchina)"), "RQ2", "completata da verificare",
            "C-RQ2", "si, appaiando per seed",
            ("Bracci senza DP e con DP eseguiti. " if _ndp else "Braccio senza DP eseguito. ") +
            f"Senza DP, loss sull'holdout del modello rilasciato, IID "
            f"contro per sito: {sum(h['iid']) / n:.6f} contro {sum(h['per_site']) / n:.6f} "
            f"(t = {h['t']:.2f}, {n - 1} gdl); Yeom {sum(y['iid']) / n:.4f} contro "
            f"{sum(y['per_site']) / n:.4f}. " + _testo_rq2_dp(),
            ("Nessuna run mancante per il confronto minimo della guida." if leggi_rq2(con_dp=True) else
             "IID e per sito al punto operativo client-level: 2 celle x 5 seed = 10 run; la "
             "cella per sito coincide con quella di RQ1 al punto operativo."),
            "media", "Partizione IID = riferimento sperimentale, non un deployment: va "
            "dichiarato. Macchina diversa da E-A: variabile da dichiarare.",
            "Fase E"]


def riga_rq3_sintesi(n_tot: int) -> list:
    """Riga RQ3 della Matrice_sintesi, dai dati di E-E."""
    d = leggi_rq3()
    s, st = statistiche_rq3(d)
    n = len(s)
    h = st["loss holdout del modello rilasciato"]; y = st["Yeom, AUC all'ultimo round"]
    ddp = leggi_rq3(con_dp=True)
    return [f"{2 * n} run E-E senza DP (terza macchina)" +
            (f", {2 * len(statistiche_rq3(ddp)[0])} con DP" if ddp else ""),
            "RQ3", "completata da verificare", "C-RQ3", "si, appaiando per seed",
            f"Senza DP, loss sull'holdout del modello rilasciato, FedAvg contro FedProx: "
            f"{sum(h['fedavg']) / n:.6f} contro {sum(h['fedprox']) / n:.6f} (t = {h['t']:.2f}, "
            f"{n - 1} gdl); Yeom {sum(y['fedavg']) / n:.4f} contro {sum(y['fedprox']) / n:.4f}. " +
            ("Braccio con DP eseguito (foglio RQ3_algoritmo)." if ddp else
             "Braccio con DP (eps = 64, C = 1) in corso."),
            ("Nessuna run mancante per il confronto minimo." if ddp else
             "Braccio con DP: 2 algoritmi x 5 seed = 10 run sulla stessa macchina. Prove su mu "
             "a un seed (ESPERIMENTI.md, E-E)."),
            "alta", "Tutta la campagna usa FedProx mu = 0.01 come base: se cambiarla e' una "
            "decisione del supervisore. La loss locale di FedProx include il termine prossimale "
            "(segnalazione 59).",
            "Fase A -> D"]


def scrivi_sintesi(righe):
    p = os.path.join(USCITA, "Matrice_sintesi.xlsx")
    wb = openpyxl.load_workbook(p)
    ws = wb["Matrice_sintesi"]
    ws.cell(1, 10).value = "fase (spina dorsale)"

    n_tot = len(righe)
    n_nat = sum(1 for r in righe if r["regime"] == "naturale")
    n_can = sum(1 for r in righe if r["regime"].startswith("canary"))
    n_inv = sum(1 for r in righe if r["stato"] == "invalidata")
    n_ver = sum(1 for r in righe if r["stato"] == "verificata")
    n_incompl = sum(1 for r in righe if r["stato"] == "incompleta")

    # 2026-09-27: righe RQ1 e RQ3 dai dati. Prima il testo era fisso al 2026-09-21
    # ("C01..C11", "0.01 su 235 run su 235") e non diceva nulla di E-A, della griglia
    # e di E-B.
    comp = composizione_celle()
    _ho = {et: [loss_holdout_modello_rilasciato(json.load(open(f))) for f in fs]
           for et, fs in comp.items()}
    _ho = {et: [x for x in v if x is not None] for et, v in _ho.items()}
    _base = (sum(_ho["no-DP baseline"]) / len(_ho["no-DP baseline"])
             if _ho.get("no-DP baseline") else None)
    _rap = {et: sum(v) / len(v) / _base for et, v in _ho.items()
            if v and _base and et != "no-DP baseline"}
    _cl = {et: x for et, x in _rap.items() if not et.startswith("record-DP") and len(_ho[et]) >= 5}
    _rec = {et: x for et, x in _rap.items() if et.startswith("record-DP")}
    _best = min(_cl, key=_cl.get) if _cl else None
    n_mu0 = sum(1 for r in righe if str(r.get("algoritmo", "")).startswith("FedAvg"))
    S = [
        [f"{n_nat} run in regime naturale, {sum(len(comp[e]) for e in _rec)} record-DP",
         "RQ1", "completata da verificare",
         f"C01..C{len([e for e in comp if e != 'no-DP baseline']):02d}", "si, appaiando per seed",
         (f"Client-level: nessuna cella a 5 seed sotto 3 volte il no-DP sull'holdout del "
          f"modello rilasciato; la migliore e' {_best}, {_cl[_best]:.1f} volte. " if _best and
          _cl[_best] > 3 else
          f"Client-level: la migliore cella a 5 seed e' {_best}, {_cl[_best]:.1f} volte. "
          if _best else "") +
         ("Record-level: " + "; ".join(f"{e}, {len(_ho[e])} seed, {x:.1f} volte" for e, x in
                                       sorted(_rec.items())) + ". " if _rec else
          "Record-level: nessuna cella. ") +
         "Mancano il braccio B1 'clipping senza rumore' (sigma=0, segnalazione 38) e la "
         "cella no-DP con GroupNorm, che separa rumore e architettura nelle celle record-DP.",
         "B1: 1 cella x 5 seed, dopo il flag --clip-only. Cella no-DP con GroupNorm: 5 seed. "
         "Altri livelli di sigma per E-B (ESPERIMENTI.md).",
         "alta", "Da decidere in Fase B: quale superficie e' primaria. "
         "Raccomandazione motivata: A1/dp-fedavg, perche' un nullo li' limita "
         "anche A2 e A3.",
         "Fase A -> B"],
        riga_rq2_sintesi(n_tot),
        riga_rq3_sintesi(n_tot) if leggi_rq3() else
        [f"{n_tot} run totali", "RQ3",
         "lacuna da colmare con E-E" if not n_mu0 else "completata da verificare",
         "C-RQ3", "no" if not n_mu0 else "da costruire",
         (f"Nessuna run con mu = 0 in questo checkout ({n_tot} run nel registro). E-E gira "
          f"su un'altra macchina (ESPERIMENTI.md); il config del braccio con DP, "
          f"experiment_rq3_mu0_eps64.yaml, e' pronto." if not n_mu0 else
          f"{n_mu0} run con mu = 0 nel registro."),
         "E-E: mu = 0 e mu = 0.01, senza DP e a eps = 64 con C = 1, 5 seed, tutte sulla "
         "stessa macchina (segnalazione 55): 20 run.",
         "alta", "Attenzione: proximal_mu vive in cfg['ml'], non in "
         "cfg['experiment']. Impostarlo nel posto sbagliato verrebbe ignorato in "
         "silenzio.",
         "Fase A -> D"],
        ["—", "RQ4", "esclusa per scelta", "—", "—",
         "Threat model passivo (Scenario 1, aggregatore honest-but-curious). "
         "ByzantineDetector e' implementato e testato a unita' ma mai esercitato "
         "end-to-end contro un client Byzantine attivo.",
         "Nessun rerun. La guida (sez. 4) chiede di registrare l'esclusione e non "
         "rivendicare una risposta.",
         "nessuna", "Da dichiarare come RQ rinviata in Fase F, con ragione e "
         "conseguenza sul claim: non deve sparire dalla narrativa.",
         "Fase A -> F"],
        [f"{n_can} run canary ({n_ver} verificate, {n_incompl} incompleta)", "CTRL",
         "verificata per la campagna bilanciata post-fix", "—", "non applicabile",
         f"Il controllo positivo e' superato e il negativo bilanciato regge "
         f"(30/30 round sopra 0.5, min 0.6125, t(4)=9.90 appaiato sui 5 seed). "
         f"Ma {n_incompl} run canary hanno il blocco 'canary' NON registrato nel "
         f"config: k, duplicati e swap_assignment non sono leggibili dal file, "
         f"quindi non si puo' stabilire dal JSON se quei bracci di scambio fossero "
         f"bilanciati. Nessuna e' marcata 'invalidata' proprio per questo: "
         f"invalidare richiederebbe un'analogia, che la guida vieta.",
         "Nessun rerun. Serve pero' una verifica documentale: associare a ciascuna "
         "delle run incomplete il YAML di config corrispondente e registrare k e "
         "swap_assignment nel registro. E' lettura di file, non nuovo calcolo.",
         "bassa", "La guida (sez. 6) e' esplicita: 'simmetrico = membership' non "
         "e' un teorema generale, e il canary non certifica il null naturale. Tenere "
         "il risultato separato dal regime naturale (guida Fase C).",
         "Fase A (strumento) — non e' una RQ"],
    ]
    for i, riga in enumerate(S, start=2):
        for c, v in enumerate(riga, start=1):
            ws.cell(i, c).value = v
    stile(ws, [30, 10, 26, 14, 20, 56, 52, 12, 52, 22], 160)
    for r in range(2, ws.max_row + 1):
        v = str(ws.cell(r, 3).value or "")
        ws.cell(r, 3).fill = (VERDE if v.startswith("verificata")
                              else ROSSO if "lacuna" in v else GIALLO)

    rg = wb["Regole_intersezione"]
    for i, (k, d) in enumerate([
        ("Una run puo' servire piu' RQ", "Una run in regime naturale serve RQ1; la stessa run servirebbe RQ3 solo se esistesse la gemella con mu=0. Non esiste, quindi non la si conta per RQ3."),
        ("Regime naturale e canary non si mescolano", "La guida (Fase C) vieta di usare il canary per certificare il null naturale. Nel registro sono due popolazioni separate e non vanno aggregate."),
        ("Deduplica", "check_significance.py e questo script compongono le celle con la stessa regola (segnalazioni 44, 45, 53, 2026-09-24): etichetta dai campi registrati nel JSON (scripts/etichetta_cella.py), cartelle _*, entity-split*, nvflare-* e fedmia-gradient-* escluse, una run per (cella, seed del config), la piu' recente. La scelta del seed del config invece del nome della cartella va dichiarata (segnalazione 9)."),
        ("Priorita'", "alta = blocca una RQ dichiarata; media = blocca una RQ ma con costo maggiore; nessuna = non serve rerun."),
        ("Cosa NON e' in queste matrici", "Qualunque decisione sulle fasi B-F. La Fase A si ferma a: cosa esiste, cosa e' confrontabile, cosa manca. Le raccomandazioni sono marcate come tali e rinviate alla fase competente."),
    ], start=2):
        rg.cell(i, 1).value = k
        rg.cell(i, 2).value = d
    stile(rg, [34, 110], 60)
    wb.save(p)
    return p, len(S)



def scrivi_costo_per_sito():
    """Costo per sito: NON e' nei JSON, solo nei log.

    Errore trovato e corretto il 2026-09-21: la decision matrix affermava che
    il dato per client fosse "nella stessa struttura" dei JSON. Non e' vero.
    per_round[*].fl contiene SOLO mean_loss globale; nessuna chiave nomina un
    client o un sito. La loss per client esiste unicamente nella riga di log
    "[<sito>-NN] Round R — loss=..., n=...". Questa tabella la estrae da li'.

    Segnalazione 46 (2026-09-24): quella loss e' la loss di ADDESTRAMENTO LOCALE
    del client nell'ultimo round (media sulle epoche locali, prima del rumore
    DP), non la qualita' del modello globale rilasciato. Con DP client-level le
    due cose divergono di uno o due ordini di grandezza.
    """
    p = os.path.join(USCITA, "Matrice_sintesi.xlsx")
    wb = openpyxl.load_workbook(p)
    # Fix 2026-09-23 (segnalazione 43): senza questa rimozione ogni nuova
    # esecuzione aggiungeva un secondo foglio "Costo_per_sito1" accanto al vecchio,
    # che restava con i numeri precedenti.
    if "Costo_per_sito" in wb.sheetnames:
        del wb["Costo_per_sito"]
    ws = wb.create_sheet("Costo_per_sito")
    ws.append(["log", "condizione DP", "sito", "round finale",
               "loss addestramento locale (ultimo round)", "n sessioni del client"])
    pat = re.compile(r"\[(\w+)-\d+\] Round (\d+) — loss=([\d.eE+-]+), n=(\d+)")
    righe = []
    for f in sorted(glob.glob(os.path.join(RADICE, "logs", "*.log"))):
        txt = open(f, errors="replace").read()
        if "[NO-DP BASELINE]" in txt:
            cond = "no-DP"
        else:
            m = re.search(r"GradientManager — .=([\d.]+)", txt)
            cond = f"DP (eps={m.group(1)})" if m else "non determinabile dal log"
        ultimo = {}
        for m in pat.finditer(txt):
            sito, rnd, loss, n = m.group(1), int(m.group(2)), float(m.group(3)), int(m.group(4))
            if sito not in ultimo or rnd > ultimo[sito][0]:
                ultimo[sito] = (rnd, loss, n)
        for sito, (rnd, loss, n) in sorted(ultimo.items()):
            righe.append([os.path.basename(f), cond, sito, rnd, loss, n])
    for r in righe:
        ws.append(r)
    stile(ws, [42, 26, 12, 14, 18, 22], 18)
    wb.save(p)
    return len(righe)



def loss_holdout_modello_rilasciato(j: dict) -> float | None:
    """Loss di ricostruzione del modello globale RILASCIATO all'ultimo round,
    sull'holdout naturale. Segnalazione 46 (2026-09-24).

    E' l'MSE medio per record del modello globale finale (quello che viene
    distribuito ai client e che gli attacchi osservano) calcolato sui non-membri
    dell'attacco Yeom, cioe' su tutto l'holdout: `-non_member_score_mean`
    dell'ultimo round. Vale solo con superficie Yeom "global", il default: nessun
    config la cambia, ma se un JSON la registra diversa si restituisce None.
    None anche se Yeom e' fallito in quella run (segnalazione 42).
    """
    c = j.get("config") or {}
    if ((c.get("yeom") or {}).get("observation_surface") or "global") != "global":
        return None
    pr = j.get("per_round") or {}
    if not pr:
        return None
    m = (pr[max(pr, key=int)] or {}).get("mia") or {}
    v = m.get("non_member_score_mean")
    if v is None or v != v:  # None o NaN
        return None
    return -float(v)


def loss_addestramento_locale(j: dict) -> float | None:
    """Media pesata delle loss di addestramento LOCALE dei client nell'ultimo
    round (`per_round[T].fl.mean_loss`), mediata sulle epoche locali e calcolata
    PRIMA del rumore DP di quel round. Non e' la qualita' del modello rilasciato:
    senza DP coincide quasi con la loss sull'holdout, con DP client-level no
    (segnalazione 46). Resta utile come diagnostica: quanto bene un client
    riadatta localmente il modello globale rumoroso che riceve."""
    pr = j.get("per_round") or {}
    loss = [(pr[r] or {}).get("fl", {}).get("mean_loss") for r in sorted(pr, key=int)]
    loss = [x for x in loss if x is not None]
    return loss[-1] if loss else None


def scrivi_utility_privacy():
    """Griglia utility x privacy x limite teorico (2026-09-21).

    Tre cose in un foglio solo, perche' vanno lette insieme:
      - il COSTO: loss sull'holdout del modello globale RILASCIATO all'ultimo
        round, rapporto rispetto al no-DP (segnalazione 46: fino al 2026-09-24
        questa colonna usava la loss di addestramento locale, che con DP
        client-level sottostima il costo di uno o due ordini di grandezza; ora
        quella e' una colonna di diagnostica a parte)
      - la PRIVACY misurata (Yeom, Shadow, LiRA, LiRA composto, TPR@1%FPR)
      - il LIMITE TEORICO che la garanzia (eps,delta) permetterebbe
    Il limite usa l'epsilon CUMULATIVO su T round (composizione base,
    eps_tot = T*eps), non l'epsilon per round: e' quello che copre il
    transcript che l'avversario osserva davvero.
    """
    import math
    p = os.path.join(USCITA, "Matrice_sintesi.xlsx")
    wb = openpyxl.load_workbook(p)
    # Fix 2026-09-23 (segnalazione 43): senza questa rimozione ogni nuova
    # esecuzione aggiungeva un secondo foglio "Utility_privacy_limite1" accanto al vecchio,
    # che restava con i numeri precedenti.
    if "Utility_privacy_limite" in wb.sheetnames:
        del wb["Utility_privacy_limite"]
    ws = wb.create_sheet("Utility_privacy_limite")
    ws.append(["cella", "n run", "loss holdout modello rilasciato",
               "x rispetto a no-DP (holdout)",
               "loss addestramento locale (diagnostica)", "x rispetto a no-DP (locale)",
               "Yeom", "Shadow", "LiRA", "LiRA composto", "TPR@1%FPR",
               "eps per round", "eps_tot (T=10, base)",
               "Adv max teorica", "AUC max teorica", "lettura"])
    # Segnalazioni 44, 45, 53 (2026-09-24): stessa composizione di
    # check_significance.py (composizione_celle). Chiave = (etichetta, epsilon).
    celle = defaultdict(list)
    for et, files in sorted(composizione_celle().items()):
        # Fino al 2026-09-27 le celle record-DP erano escluse: c'era solo la prova a un
        # seed. Con E-B entrano, con l'eps per record al posto di quello per round.
        for f in sorted(files):
            sw = os.path.basename(os.path.dirname(f))
            j = json.load(open(f)); c = j.get("config") or {}
            if (c.get("canary") or {}).get("enabled") or "canary" in sw.lower():
                continue
            pr = j.get("per_round") or {}; su = j.get("summary") or {}
            if not pr:
                continue
            ult = max(pr, key=int)
            k = (et, None if et.startswith(("no-DP", "record-DP")) else c.get("epsilon"))
            celle[k].append((loss_holdout_modello_rilasciato(j), su.get("mean_auc_roc"),
                             su.get("mean_shadow_auc_roc"), su.get("mean_lira_auc_roc"),
                             pr[ult].get("mia", {}).get("composed_lira_auc_roc"),
                             pr[ult].get("mia", {}).get("composed_tpr_at_fpr_0.01"),
                             loss_addestramento_locale(j),
                             c.get("epsilon_record_dp"), c.get("epsilon_record_dp_shuffle_bound")))

    def med(v, i):
        x = [t[i] for t in v if t[i] is not None]
        return sum(x) / len(x) if x else None

    base = med(celle.get(("no-DP baseline", None), []), 0)
    base_loc = med(celle.get(("no-DP baseline", None), []), 6)
    # 2026-09-28: riferimento con la stessa normalizzazione delle celle record-DP
    # (GroupNorm, experiment_rq1_nodp_groupnorm.yaml). Il rapporto di colonna resta sul
    # no-DP della campagna; quello sul riferimento GroupNorm va nella lettura.
    base_gn = med(celle.get(("no-DP baseline, norm=group", None), []), 0)
    for k in sorted(celle, key=lambda t: (str(t[0]), -(t[1] or 0))):
        v = celle[k]
        lf = med(v, 0)
        ll = med(v, 6)
        eps = k[1]
        if k[0].startswith("record-DP"):
            er, esh = med(v, 7), med(v, 8)
            adv = (math.exp(er) - 1) / (math.exp(er) + 1) if er is not None else None
            teo = [None, round(er, 2) if er is not None else None,
                   round(adv, 6) if adv is not None else None,
                   round(0.5 + adv / 2, 6) if adv is not None else None]
            lettura = ((f"{lf / base_gn:.2f} volte il no-DP con GroupNorm, il riferimento con la "
                        f"stessa normalizzazione. " if (lf and base_gn) else "") +
                       f"UNITA' PROTETTA RECORD: al posto di eps_tot c'e' l'eps per record "
                       f"di Poisson, massimo sui client e gia' composto sui round, in media "
                       f"{er:.1f}; con lo shuffle il limite valido e' {esh:.0f}. Non e' "
                       f"confrontabile con l'eps per round del client-level. "
                       + ("Bound VACUO come nelle celle client-level." if base_gn else
                          "Il rapporto di costo mescola rumore e GroupNorm (manca la cella no-DP con "
                          "GroupNorm). Bound VACUO come nelle celle client-level.")
                       if er is not None and esh is not None else
                       "UNITA' PROTETTA RECORD: eps per record non registrato nel JSON")
        elif eps:
            et = 10 * eps
            adv = (math.exp(et) - 1) / (math.exp(et) + 1)
            teo = [eps, et, round(adv, 6), round(0.5 + adv / 2, 6)]
            lettura = ("bound VACUO: a questo eps cumulativo la garanzia permette "
                       "un attacco quasi perfetto, quindi 'siamo sotto il limite' "
                       "non e' informativo")
        else:
            teo = [None, None, None, None]
            lettura = "riferimento senza DP: nessuna garanzia da confrontare"
        rap = round(lf / base, 1) if (lf and base) else None
        rap_loc = round(ll / base_loc, 1) if (ll and base_loc) else None
        if rap and rap > 50:
            lettura = ("UTILITY DISTRUTTA (loss sull'holdout del modello rilasciato "
                       "oltre 50x il riferimento): un nullo di privacy qui non "
                       "distingue 'DP protegge' da 'il modello non impara, quindi non "
                       "memorizza'. " + lettura)
        elif rap and rap > 3:
            lettura = ("COSTO OLTRE LA SOGLIA DI 3x (sezione 0 di ESPERIMENTI.md). "
                       + lettura)
        ws.append([k[0], len(v),
                   round(lf, 6) if lf else None, rap,
                   round(ll, 6) if ll else None, rap_loc,
                   round(med(v, 1) or 0, 4), round(med(v, 2) or 0, 4),
                   round(med(v, 3) or 0, 4), round(med(v, 4) or 0, 4),
                   round(med(v, 5) or 0, 4), *teo, lettura])
    stile(ws, [22, 7, 16, 16, 18, 16, 9, 9, 9, 13, 11, 13, 18, 15, 16, 70], 80)
    for r in range(2, ws.max_row + 1):
        rap = ws.cell(r, 4).value
        if rap and rap > 50:
            ws.cell(r, 4).fill = ROSSO
        elif rap and rap > 3:
            ws.cell(r, 4).fill = GIALLO
        elif rap:
            ws.cell(r, 4).fill = VERDE
    wb.save(p)
    return ws.max_row - 1


def scrivi_worst_case():
    """Vulnerabilita' per record: osservato contro livello di caso, con il
    controllo sui non-membri e il test appaiato (segnalazione 47, 2026-09-24)."""
    fjson = os.path.join(USCITA, "worst_case", "livello_di_caso.json")
    if not os.path.exists(fjson):
        return 0
    d = json.load(open(fjson))
    p = os.path.join(USCITA, "Matrice_sintesi.xlsx")
    wb = openpyxl.load_workbook(p)
    # Fix 2026-09-23 (segnalazione 43): senza questa rimozione ogni nuova
    # esecuzione aggiungeva un secondo foglio "Worst_case_per_record1" accanto al vecchio,
    # che restava con i numeri precedenti.
    if "Worst_case_per_record" in wb.sheetnames:
        del wb["Worst_case_per_record"]
    ws = wb.create_sheet("Worst_case_per_record")
    ws.append(["gruppo", "n seed", "sessioni multi-seed", "record segnalati (membri)",
               "attesi per caso", "sd", "z membri (stabilita' del ranking)",
               "z non-membri (controllo)", "z membri meno non-membri",
               "test appaiato: diff. media di percentile (membro - non membro)",
               "errore standard", "z appaiato", "% osservata", "% attesa", "lettura"])
    for et, r in d["gruppi"].items():
        z = r.get("z")
        c = r.get("controllo_non_membri") or {}
        a = r.get("appaiato") or {}
        za = a.get("z")
        if not a:
            let = ("CONTROLLO MANCANTE: livello_di_caso.json prodotto prima della "
                   "segnalazione 47, senza test appaiato. Il solo z sui membri misura "
                   "la stabilita' del ranking, non l'appartenenza: rigenerare.")
        elif za is not None and za > 3:
            let = ("SEGNALE DI APPARTENENZA PER RECORD: la stessa sessione ha "
                   "percentile piu' alto quando e' membro che quando non lo e'.")
        else:
            let = ("NESSUN SEGNALE DI APPARTENENZA PER RECORD (test appaiato). "
                   + ("L'eccesso sui membri c'e' anche fra i non-membri: e' "
                      "stabilita' del ranking dovuta al record. "
                      if (z and z > 3 and c.get("z") and c["z"] > 3) else "")
                   + "Se l'utility e' distrutta, l'assenza di segnale non prova "
                   "che la DP protegga.")
        ws.append([et, r["n_seed"], r["sessioni_multi_seed"], r["osservati"],
                   r["attesi_per_caso"], r["sd_nulla"], z, c.get("z"),
                   r.get("z_eccesso_membri_meno_non_membri"),
                   a.get("differenza_media"), a.get("errore_standard"), za,
                   r["percentuale_osservata"], r["percentuale_attesa"], let])
    stile(ws, [22, 8, 20, 16, 16, 8, 16, 16, 16, 22, 12, 11, 13, 12, 80], 76)
    for r in range(2, ws.max_row + 1):
        z = ws.cell(r, 12).value
        ws.cell(r, 12).fill = ROSSO if (z and z > 3) else VERDE
    par = d.get("parametri", {})
    ws.append([])
    ws.append([f"criterio: membro in >= {par.get('min_seed')} seed, percentile medio >= "
               f"{par.get('soglia')}, percentile minimo >= {par.get('pavimento')}; "
               f"livello di caso da {par.get('permutazioni')} permutazioni dei percentili "
               f"DENTRO ogni seed (conserva la distribuzione marginale, distrugge solo "
               f"la corrispondenza fra seed). Lettura primaria: test appaiato "
               f"(segnalazione 47); il conteggio sui membri da solo misura la "
               f"stabilita' del ranking."])
    wb.save(p)
    return ws.max_row - 1



# ─────────────────────────────────────────────────────────────────────────────
# RQ2 (E-D): partizione IID contro per sito, dai JSON di experiments_altre_macchine/
# ─────────────────────────────────────────────────────────────────────────────
# Segnalazione 45: il braccio per sito di E-D ha il config della base, quindi in
# experiments/ sostituirebbe i seed di nodp-sweep2 (una run per cella e seed, la piu'
# recente). I due bracci, prodotti su una seconda macchina, stanno percio' in
# experiments_altre_macchine/ (non versionato) e si confrontano fra loro, appaiati
# per seed: stessa macchina, stesso commit, cambia solo partition.strategy.
RQ2_BRACCI = {"rq2-per_site": "per_site", "rq2-iid": "iid"}
# Braccio con DP (2026-09-26): la stessa coppia al punto operativo scelto prima di
# vederla, dp-fedavg con eps = 64 e C = 1 (config experiment_rq2_*_eps64.yaml).
RQ2_BRACCI_DP = {"rq2-per_site-eps64": "per_site", "rq2-iid-eps64": "iid"}
RQ2_EPS_DP = 64.0
T_CRITICO_4GDL = 2.776  # t di Student, 4 gradi di liberta', alpha = 0.05 bilaterale


def leggi_rq2(con_dp: bool = False) -> dict | None:
    """{braccio: {seed: json}} per i due bracci di E-D; None se manca un braccio.
    Le chiavi sono sempre "rq2-per_site" e "rq2-iid"; con_dp=True legge la coppia
    con DP (RQ2_BRACCI_DP). A parita' di seed vince il file piu' recente (nome col
    timestamp), come in check_significance.py. Un JSON con partizione o regime
    diversi dal braccio e' un errore di dati, non un caso da gestire in silenzio."""
    out = {}
    for braccio, strategia in (RQ2_BRACCI_DP if con_dp else RQ2_BRACCI).items():
        fs = sorted(glob.glob(os.path.join(ALTRE_MACCHINE, braccio, "experiment_*.json")))
        if not fs:
            return None
        per_seed = {}
        for f in fs:
            j = json.load(open(f))
            c = j.get("config") or {}
            regime_ok = ((not c.get("no_dp") and c.get("epsilon") == RQ2_EPS_DP
                          and c.get("max_grad_norm") == 1.0 and c.get("dp_mode") == "dp-fedavg")
                         if con_dp else bool(c.get("no_dp")))
            if c.get("partition_strategy") != strategia or not regime_ok:
                raise ValueError(f"{f}: atteso partition_strategy={strategia} e "
                                 f"{'dp-fedavg eps=' + str(RQ2_EPS_DP) + ' C=1' if con_dp else 'no_dp'}, "
                                 f"trovato {c.get('partition_strategy')}, no_dp={c.get('no_dp')}, "
                                 f"eps={c.get('epsilon')}, C={c.get('max_grad_norm')}")
            per_seed[c.get("seed")] = j
        out[braccio.replace("-eps64", "")] = per_seed
    return out


def _metriche_rq2(j: dict) -> dict:
    pr = j.get("per_round") or {}
    ult = str(max(int(r) for r in pr))
    m = pr[ult].get("mia", {})
    su = j.get("summary") or {}
    ho = loss_holdout_modello_rilasciato(j)
    mem = -m["member_score_mean"] if m.get("member_score_mean") is not None else None
    return {
        "loss holdout del modello rilasciato": ho,
        "loss sui membri del modello rilasciato": mem,
        "divario holdout - membri": (ho - mem) if (ho is not None and mem is not None) else None,
        "loss addestramento locale (diagnostica)": loss_addestramento_locale(j),
        "Yeom, AUC all'ultimo round": m.get("auc_roc"),
        "Yeom, AUC media sui round": su.get("mean_auc_roc"),
        "Shadow, AUC media sui round": su.get("mean_shadow_auc_roc"),
        "LiRA composto, AUC": m.get("composed_lira_auc_roc"),
        "LiRA composto, TPR a FPR 1%": m.get("composed_tpr_at_fpr_0.01"),
    }


def appaiato(a: list[float], b: list[float]) -> dict:
    """Differenze b - a appaiate per posizione: media, deviazione standard, t con
    n-1 gradi di liberta', quante differenze sono negative."""
    d = [y - x for x, y in zip(a, b)]
    n = len(d)
    media = sum(d) / n
    sd = (sum((x - media) ** 2 for x in d) / (n - 1)) ** 0.5 if n > 1 else None
    t = media / (sd / n ** 0.5) if sd else None
    return {"n": n, "media": media, "sd": sd, "t": t, "negative": sum(1 for x in d if x < 0)}


def statistiche_rq2(rq2: dict) -> tuple[list[int], dict]:
    seeds = sorted(set(rq2["rq2-per_site"]) & set(rq2["rq2-iid"]))
    ps = {s: _metriche_rq2(rq2["rq2-per_site"][s]) for s in seeds}
    ii = {s: _metriche_rq2(rq2["rq2-iid"][s]) for s in seeds}
    stat = {}
    for k in ps[seeds[0]]:
        a = [ps[s][k] for s in seeds]; b = [ii[s][k] for s in seeds]
        if any(v is None for v in a + b):
            continue
        stat[k] = {"per_site": a, "iid": b, **appaiato(a, b)}
    return seeds, stat


def _lettura_rq2(k: str, s: dict) -> str:
    """Lettura di una riga del foglio RQ2_partizione, uguale per i due regimi."""
    sig = s["t"] is not None and abs(s["t"]) > T_CRITICO_4GDL
    if "AUC" in k:
        return ("entrambi i bracci al caso; " +
                ("differenza significativa a 5 seed" if sig else "nessuna differenza significativa a 5 seed"))
    if "TPR" in k:
        return ("il caso vale 0.01; " +
                ("differenza significativa" if sig else "nessuna differenza significativa"))
    return ("differenza significativa a 5 seed" if sig else
            f"nessuna differenza significativa a 5 seed (|t| < {T_CRITICO_4GDL})")


def scrivi_rq2():
    rq2 = leggi_rq2()
    if not rq2:
        return 0
    seeds, stat = statistiche_rq2(rq2)
    nodp = {}
    for f in composizione_celle().get("no-DP baseline", []):
        j = json.load(open(f)); nodp[j["config"]["seed"]] = _metriche_rq2(j)
    p = os.path.join(USCITA, "Matrice_sintesi.xlsx")
    wb = openpyxl.load_workbook(p)
    if "RQ2_partizione" in wb.sheetnames:
        del wb["RQ2_partizione"]
    ws = wb.create_sheet("RQ2_partizione")
    ws.append(["metrica", f"per sito, media ({len(seeds)} seed)", "IID, media",
               "differenza appaiata IID - per sito", "deviazione standard",
               f"t ({len(seeds) - 1} gdl)", "seed con IID sotto per sito",
               "nodp-sweep2, media (prima macchina, stesso config di per sito)", "lettura"])
    for k, s in stat.items():
        mps = sum(s["per_site"]) / len(seeds); mii = sum(s["iid"]) / len(seeds)
        vnodp = [nodp[x][k] for x in seeds if x in nodp and nodp[x].get(k) is not None]
        let = _lettura_rq2(k, s)
        ws.append([k, round(mps, 6), round(mii, 6), round(s["media"], 6),
                   round(s["sd"], 6) if s["sd"] is not None else None,
                   round(s["t"], 2) if s["t"] is not None else None,
                   f"{s['negative']} su {s['n']}",
                   round(sum(vnodp) / len(vnodp), 6) if len(vnodp) == len(seeds) else None, let])
    ws.append([])
    ws.append(["seed", "braccio"] + list(stat))
    for x in seeds:
        for b, lab in (("per_site", "per sito"), ("iid", "IID")):
            ws.append([x, lab] + [round(stat[k][b][seeds.index(x)], 6) for k in stat])
    commit = {j["config"].get("git_commit") for b in rq2.values() for j in b.values()}
    ho = "loss holdout del modello rilasciato"
    scarti = [abs(stat[ho]["per_site"][i] / nodp[x][ho] - 1) for i, x in enumerate(seeds)
              if x in nodp and nodp[x].get(ho)]
    ws.append([])
    ws.append([f"Fonte: experiments_altre_macchine/rq2-per_site e rq2-iid (E-D, senza DP, seconda "
               f"macchina, commit {', '.join(sorted(c[:7] for c in commit if c))}). I due bracci "
               f"differiscono solo per partition.strategy e sono appaiati per seed. La partizione IID "
               f"rimescola le sessioni fra i siti conservando le numerosita' per client: e' un "
               f"riferimento sperimentale, non un deployment. nodp-sweep2 e' la stessa configurazione "
               f"del braccio per sito su un'altra macchina e un altro commit: la loss sull'holdout "
               f"differisce per seed fino al {100 * max(scarti):.0f}% (segnalazione 45), per questo "
               f"il confronto di RQ2 non la usa." if scarti else
               "Fonte: experiments_altre_macchine/ (E-D, seconda macchina)."])
    rq2dp = leggi_rq2(con_dp=True)
    n_dp = 0
    if rq2dp:
        sd_, st_dp = statistiche_rq2(rq2dp)
        ws.append([])
        ws.append([f"CON DP: dp-fedavg, eps = {RQ2_EPS_DP:g} per round, C = 1 (stesso punto per "
                   f"le due partizioni, scelto prima di vederle), {len(sd_)} seed per braccio"])
        ws.append(["metrica", "per sito, media", "IID, media", "differenza appaiata IID - per sito",
                   "deviazione standard", f"t ({len(sd_) - 1} gdl)", "seed con IID sotto per sito",
                   "", "lettura"])
        for k, s in st_dp.items():
            ws.append([k, round(sum(s["per_site"]) / len(sd_), 6), round(sum(s["iid"]) / len(sd_), 6),
                       round(s["media"], 6), round(s["sd"], 6) if s["sd"] is not None else None,
                       round(s["t"], 2) if s["t"] is not None else None,
                       f"{s['negative']} su {s['n']}", None, _lettura_rq2(k, s)])
        n_dp = len(st_dp)
        ws.append([])
        ws.append(["seed", "braccio (con DP)"] + list(st_dp))
        for x in sd_:
            for b, lab in (("per_site", "per sito"), ("iid", "IID")):
                ws.append([x, lab] + [round(st_dp[k][b][sd_.index(x)], 6) for k in st_dp])
        commit_dp = {j["config"].get("git_commit") for b in rq2dp.values() for j in b.values()}
        n_sig = sum(1 for st in (stat, st_dp) for v in st.values()
                    if v["t"] is not None and abs(v["t"]) > T_CRITICO_4GDL)
        ws.append([f"Fonte: experiments_altre_macchine/rq2-per_site-eps64 e rq2-iid-eps64 (seconda "
                   f"macchina, commit {', '.join(sorted(c[:7] for c in commit_dp if c))}). "
                   f"Molteplicita': {len(stat) + len(st_dp)} test appaiati in questo foglio, "
                   f"{n_sig} con |t| > {T_CRITICO_4GDL}; a alpha = 0.05 senza correzione ce ne si "
                   f"attende circa {0.05 * (len(stat) + len(st_dp)):.1f} per caso. Metriche primarie "
                   f"(ESPERIMENTI.md, sezione 0): loss sull'holdout per il costo, Yeom e test per "
                   f"record (foglio Worst_case_per_record) per l'attacco; LiRA e Shadow sono "
                   f"secondarie e la calibrazione di LiRA e' degenere (STATO.md 3.4)."])
        # costo della DP per partizione: rapporto con DP / senza DP allo stesso seed
        comuni = [x for x in sd_ if x in seeds]
        if comuni:
            rap = {b: [st_dp[ho][b][sd_.index(x)] / stat[ho][b][seeds.index(x)] for x in comuni]
                   for b in ("per_site", "iid")}
            geo = {b: math.exp(sum(math.log(v) for v in rap[b]) / len(comuni)) for b in rap}
            inter = appaiato([math.log(v) for v in rap["per_site"]], [math.log(v) for v in rap["iid"]])
            ws.append([])
            ws.append(["COSTO DELLA DP PER PARTIZIONE: loss sull'holdout con DP / senza DP, stesso seed"])
            ws.append(["seed"] + [str(x) for x in comuni] + ["media geometrica"])
            for b, lab in (("per_site", "per sito"), ("iid", "IID")):
                ws.append([lab] + [round(v, 2) for v in rap[b]] + [round(geo[b], 2)])
            sig = inter["t"] is not None and abs(inter["t"]) > T_CRITICO_4GDL
            ws.append([f"Interazione partizione x DP, sul logaritmo del rapporto (IID - per sito): "
                       f"media {inter['media']:+.3f}, sd {inter['sd']:.3f}, t({inter['n'] - 1}) = "
                       f"{inter['t']:.2f}; " + ("significativa" if sig else
                       f"non significativa a {inter['n']} seed (|t| < {T_CRITICO_4GDL})") +
                       f". Rapporto IID/per sito delle medie geometriche: {geo['iid'] / geo['per_site']:.2f}."])
    stile(ws, [40, 16, 14, 18, 14, 10, 14, 22, 60], 40)
    wb.save(p)
    return len(stat) + n_dp


# ─────────────────────────────────────────────────────────────────────────────
# RQ3 (E-E): FedAvg (mu = 0) contro FedProx (mu = 0.01), 2026-09-27
# ─────────────────────────────────────────────────────────────────────────────
# Come RQ2: i bracci girano su un'altra macchina (Windows) e stanno in
# experiments_altre_macchine/; il braccio FedProx ha il config della base e in
# experiments/ sostituirebbe i seed di nodp-sweep2 (segnalazione 45). Il confronto e'
# fra i due bracci della stessa macchina, appaiati per seed.
RQ3_BRACCI = {"rq3-mu0.01": 0.01, "rq3-mu0": 0.0}
RQ3_BRACCI_DP = {"rq3-mu0.01-eps64": 0.01, "rq3-mu0-eps64": 0.0}
RQ3_NOMI = {"rq3-mu0.01": "FedProx mu = 0.01", "rq3-mu0": "FedAvg mu = 0"}


def leggi_rq3(con_dp: bool = False) -> dict | None:
    """{braccio: {seed: json}} per i due bracci di E-E; None se manca un braccio. Chiavi
    sempre "rq3-mu0.01" e "rq3-mu0". Un JSON con mu, regime o partizione diversi dal
    braccio e' un errore di dati."""
    out = {}
    for braccio, mu in (RQ3_BRACCI_DP if con_dp else RQ3_BRACCI).items():
        fs = sorted(glob.glob(os.path.join(ALTRE_MACCHINE, braccio, "experiment_*.json")))
        if not fs:
            return None
        per_seed = {}
        for f in fs:
            j = json.load(open(f))
            c = j.get("config") or {}
            regime_ok = ((not c.get("no_dp") and c.get("epsilon") == RQ2_EPS_DP
                          and c.get("max_grad_norm") == 1.0 and c.get("dp_mode") == "dp-fedavg")
                         if con_dp else bool(c.get("no_dp")))
            mu_ok = c.get("proximal_mu") is not None and float(c["proximal_mu"]) == mu
            if not (mu_ok and regime_ok and c.get("partition_strategy", "per_site") == "per_site"):
                raise ValueError(f"{f}: atteso proximal_mu={mu}, "
                                 f"{'dp-fedavg eps=64 C=1' if con_dp else 'no_dp'}, per_site; "
                                 f"trovato mu={c.get('proximal_mu')}, no_dp={c.get('no_dp')}, "
                                 f"eps={c.get('epsilon')}, C={c.get('max_grad_norm')}, "
                                 f"partizione={c.get('partition_strategy')}")
            per_seed[c.get("seed")] = j
        out[braccio.replace("-eps64", "")] = per_seed
    return out


def statistiche_rq3(d: dict) -> tuple[list[int], dict]:
    """Differenze FedAvg - FedProx appaiate per seed, per ogni metrica di _metriche_rq2."""
    seeds = sorted(set(d["rq3-mu0.01"]) & set(d["rq3-mu0"]))
    fp = {s: _metriche_rq2(d["rq3-mu0.01"][s]) for s in seeds}
    fa = {s: _metriche_rq2(d["rq3-mu0"][s]) for s in seeds}
    stat = {}
    for k in fp[seeds[0]]:
        a = [fp[s][k] for s in seeds]; b = [fa[s][k] for s in seeds]
        if any(v is None for v in a + b):
            continue
        stat[k] = {"fedprox": a, "fedavg": b, **appaiato(a, b)}
    return seeds, stat


def _curva_holdout(js: dict, seeds: list[int]) -> list[float | None]:
    """Loss sull'holdout del modello globale a ogni round, media sui seed."""
    n_r = min(len(js[s].get("per_round") or {}) for s in seeds)
    out = []
    for r in range(1, n_r + 1):
        v = [-(js[s]["per_round"][str(r)].get("mia") or {}).get("non_member_score_mean", float("nan"))
             for s in seeds]
        v = [x for x in v if x == x]
        out.append(sum(v) / len(v) if v else None)
    return out


def _lettura_rq3(k: str, s: dict) -> str:
    if k.startswith("loss addestramento locale"):
        return ("non confrontabile fra i bracci: con FedProx include il termine prossimale "
                "(segnalazione 59)")
    return _lettura_rq2(k, s)


def scrivi_rq3():
    d = leggi_rq3()
    if not d:
        return 0
    seeds, stat = statistiche_rq3(d)
    n = len(seeds)
    p = os.path.join(USCITA, "Matrice_sintesi.xlsx")
    wb = openpyxl.load_workbook(p)
    if "RQ3_algoritmo" in wb.sheetnames:
        del wb["RQ3_algoritmo"]
    ws = wb.create_sheet("RQ3_algoritmo")
    ws.append(["metrica", f"FedProx mu = 0.01, media ({n} seed)", "FedAvg mu = 0, media",
               "differenza appaiata FedAvg - FedProx", "deviazione standard", f"t ({n - 1} gdl)",
               "seed con FedAvg sotto FedProx", "lettura"])
    for k, st in stat.items():
        ws.append([k, round(sum(st["fedprox"]) / n, 6), round(sum(st["fedavg"]) / n, 6),
                   round(st["media"], 6), round(st["sd"], 6) if st["sd"] is not None else None,
                   round(st["t"], 2) if st["t"] is not None else None,
                   f"{st['negative']} su {st['n']}", _lettura_rq3(k, st)])
    ho = "loss holdout del modello rilasciato"
    rap = [stat[ho]["fedprox"][i] / stat[ho]["fedavg"][i] for i in range(n)]
    geo = math.exp(sum(math.log(x) for x in rap) / n)
    ws.append([f"Rapporto FedProx / FedAvg sulla loss sull'holdout per seed: "
               f"{', '.join(f'{x:.2f}' for x in rap)}; media geometrica {geo:.2f}."])
    ws.append([])
    ws.append(["seed", "braccio"] + list(stat))
    for x in seeds:
        for b, lab in (("fedprox", "FedProx"), ("fedavg", "FedAvg")):
            ws.append([x, lab] + [round(stat[k][b][seeds.index(x)], 6) for k in stat])
    ws.append([])
    ws.append(["round", "loss sull'holdout del modello globale, FedProx (media sui seed)",
               "idem, FedAvg"])
    cfp = _curva_holdout(d["rq3-mu0.01"], seeds); cfa = _curva_holdout(d["rq3-mu0"], seeds)
    for r, (a, b) in enumerate(zip(cfp, cfa), start=1):
        ws.append([r, round(a, 6) if a is not None else None, round(b, 6) if b is not None else None])
    commit = sorted({j["config"].get("git_commit", "")[:7] for b in d.values() for j in b.values()})
    ws.append([])
    ws.append([f"Fonte: experiments_altre_macchine/rq3-mu0.01 e rq3-mu0 (E-E, senza DP, terza "
               f"macchina, commit {', '.join(commit)}; i due bracci hanno girato in parallelo). "
               f"Differiscono solo per ml.proximal_mu e sono appaiati per seed; il round 1 e' "
               f"identico nei due bracci, perche' il termine prossimale entra dal round 2. Il "
               f"braccio FedProx ha il config di nodp-sweep2 ma gira su un'altra macchina "
               f"(segnalazione 55): il confronto usa i due bracci della stessa macchina."])
    n_dp = 0
    ddp = leggi_rq3(con_dp=True)
    if ddp:
        sd_, st_dp = statistiche_rq3(ddp)
        ws.append([])
        ws.append([f"CON DP: dp-fedavg, eps = {RQ2_EPS_DP:g} per round, C = 1 (punto scelto prima "
                   f"di vedere RQ3 con DP), {len(sd_)} seed per braccio"])
        ws.append(["metrica", "FedProx, media", "FedAvg, media", "differenza appaiata FedAvg - FedProx",
                   "deviazione standard", f"t ({len(sd_) - 1} gdl)", "seed con FedAvg sotto FedProx",
                   "lettura"])
        for k, st in st_dp.items():
            ws.append([k, round(sum(st["fedprox"]) / len(sd_), 6), round(sum(st["fedavg"]) / len(sd_), 6),
                       round(st["media"], 6), round(st["sd"], 6) if st["sd"] is not None else None,
                       round(st["t"], 2) if st["t"] is not None else None,
                       f"{st['negative']} su {st['n']}", _lettura_rq3(k, st)])
        n_dp = len(st_dp)
        comuni = [x for x in sd_ if x in seeds]
        if comuni:
            rr = {b: [st_dp[ho][b][sd_.index(x)] / stat[ho][b][seeds.index(x)] for x in comuni]
                  for b in ("fedprox", "fedavg")}
            gg = {b: math.exp(sum(math.log(v) for v in rr[b]) / len(comuni)) for b in rr}
            inter = appaiato([math.log(v) for v in rr["fedprox"]], [math.log(v) for v in rr["fedavg"]])
            ws.append([])
            ws.append(["COSTO DELLA DP PER ALGORITMO: loss sull'holdout con DP / senza DP, stesso seed"])
            ws.append(["seed"] + [str(x) for x in comuni] + ["media geometrica"])
            for b, lab in (("fedprox", "FedProx"), ("fedavg", "FedAvg")):
                ws.append([lab] + [round(v, 2) for v in rr[b]] + [round(gg[b], 2)])
            if inter["t"] is not None:
                sig = abs(inter["t"]) > T_CRITICO_4GDL
                ws.append([f"Interazione algoritmo x DP, sul logaritmo del rapporto (FedAvg - FedProx): "
                           f"media {inter['media']:+.3f}, sd {inter['sd']:.3f}, t({inter['n'] - 1}) = "
                           f"{inter['t']:.2f}; " + ("significativa" if sig else
                           f"non significativa a {inter['n']} seed (|t| < {T_CRITICO_4GDL})") + "."])
    stile(ws, [40, 18, 16, 20, 14, 10, 16, 60], 40)
    wb.save(p)
    return len(stat) + n_dp


def scrivi_glossario():
    """Glossario delle metriche che compaiono nelle matrici.

    Esiste perche' due numeri di queste tabelle NON si leggono da soli:
    lo z della vulnerabilita' per record, e il limite teorico della DP.
    Chi apre il foglio senza il contesto rischia di leggerli al contrario.
    """
    p = os.path.join(USCITA, "Matrice_sintesi.xlsx")
    wb = openpyxl.load_workbook(p)
    # Fix 2026-09-23 (segnalazione 43): senza questa rimozione ogni nuova
    # esecuzione aggiungeva un secondo foglio "Glossario_metriche1" accanto al vecchio,
    # che restava con i numeri precedenti.
    if "Glossario_metriche" in wb.sheetnames:
        del wb["Glossario_metriche"]
    ws = wb.create_sheet("Glossario_metriche")
    ws.append(["termine", "definizione", "come si legge", "dove compare",
               "documento di riferimento"])
    voci = [
        ("z (vulnerabilita' per record)",
         "Quante deviazioni standard il numero di record segnalati dista dalla media "
         "dei conteggi ottenuti permutando i percentili DENTRO ogni seed. La "
         "permutazione conserva la distribuzione dei punteggi di ogni seed e "
         "distrugge solo la corrispondenza FRA seed, che e' cio' che il criterio "
         "misura. Formula: z = (osservati - media permutata) / dev.std. permutata.",
         "z ~ 0 -> il conteggio e' quello che il caso produce, nessuna vulnerabilita' "
         "rilevabile con questo criterio. z > 3 -> eccesso reale: esiste un "
         "sottoinsieme di record sistematicamente nel decile alto su seed "
         "indipendenti. Lo z dice quanto e' improbabile il caso, NON quanto grave "
         "sia l'esposizione: per quella si guardano l'eccesso assoluto e la %.",
         "foglio Worst_case_per_record", "docs/VulnerabilitaPerRecord.md sez. 3"),
        ("record segnalati",
         "Sessioni reali che sono membro in almeno 2 seed, con percentile medio del "
         "punteggio d'attacco >= 90 e percentile minimo >= 75. Le soglie sono "
         "dichiarate in anticipo e non ottimizzate sui risultati.",
         "Da confrontare SEMPRE con 'attesi per caso'. Il conteggio da solo non e' "
         "un risultato: con migliaia di sessioni e soglia al 90 percentile un certo "
         "numero di coincidenze e' garantito.",
         "foglio Worst_case_per_record", "docs/VulnerabilitaPerRecord.md sez. 2"),
        ("AUC max teorica",
         "Tetto che la garanzia (eps,delta)-DP pone sull'AUC di QUALUNQUE attacco: "
         "0.5 + Adv/2 con Adv = (e^eps - 1 + 2delta)/(e^eps + 1) (Humphries et al.). "
         "Calcolata sull'eps CUMULATIVO T*eps, non su quello per round, perche' deve "
         "coprire il transcript che l'avversario osserva davvero.",
         "A eps_tot = 10 vale 0.99996: il bound e' VACUO, non esclude quasi nulla. "
         "Dire 'siamo sotto il limite teorico' e' vero e privo di contenuto. Va usato "
         "al contrario: la distanza fra permesso e misurato quantifica quanto la "
         "garanzia formale sia lasca rispetto al comportamento reale.",
         "foglio Utility_privacy_limite", "docs/LimiteTeoricoDP.md"),
        ("x rispetto a no-DP (holdout)",
         "Rapporto fra la loss di ricostruzione sull'holdout naturale del modello "
         "globale RILASCIATO all'ultimo round e quella della cella senza DP. E' la "
         "misura di costo (soglia di 3x, sezione 0 di ESPERIMENTI.md). Fino al "
         "2026-09-24 la colonna usava la loss di addestramento locale "
         "(segnalazione 46).",
         "Oltre 50x significa utility distrutta. In quelle celle un nullo di privacy "
         "NON distingue 'la DP protegge' da 'il modello non impara, quindi non "
         "memorizza e non espone'. Le due spiegazioni non sono separabili.",
         "foglio Utility_privacy_limite", "docs/LimiteTeoricoDP.md sez. 3"),
        ("loss addestramento locale (diagnostica)",
         "Media pesata delle loss di addestramento locale dei client nell'ultimo "
         "round, mediata sulle 50 epoche e calcolata prima del rumore DP di quel "
         "round (per_round[T].fl.mean_loss; nel log: 'Round T — loss globale').",
         "NON e' la qualita' del modello rilasciato. Senza DP coincide quasi con la "
         "loss sull'holdout; con DP client-level misura quanto bene un client "
         "riadatta localmente il modello rumoroso che riceve, e la distanza dalla "
         "loss sull'holdout misura il danno del rumore aggiunto all'aggregazione. Con "
         "FedProx (mu > 0) include il termine prossimale (mu/2)||w - w_globale||^2: fra "
         "celle con mu diversi non si confronta (segnalazione 59).",
         "fogli Utility_privacy_limite e Costo_per_sito", "segnalazione 46"),
        ("A0 / A1 / A2 / A3",
         "Punti di osservazione dell'avversario sulla stessa pipeline di "
         "privatizzazione. A0 = nessuna DP. A1 = update grezzo g (dp-fedavg). "
         "A2 = update clippato (central). A3 = update clippato e rumorizzato (local).",
         "Sono ORDINATI per informazione: A2 e A3 si ottengono da A1 con "
         "trasformazioni che distruggono informazione. Chi osserva A1 puo' calcolarsi "
         "A2 e A3; il contrario e' impossibile. Quindi un nullo ad A1 limita anche "
         "gli altri due: non c'era segnale da sopprimere.",
         "registro run, colonna superficie dell'attacco",
         "docs/paper/latex_dsn2027/sections/threat_model.tex"),
        ("stato di validita'",
         "I cinque stati previsti dalla guida: verificata, completata da verificare, "
         "invalidata, incompleta, pianificata.",
         "'completata da verificare' e' lo stato di default e NON significa "
         "sospetta: significa che la tracciabilita' completa richiesta dalla Fase A "
         "non e' ricostruibile, perche' commit e hash degli split non erano "
         "registrati prima del 2026-09-21.",
         "registro run", "ChargeShield_FL_spina_dorsale_consolidata.md riga 150"),
    ]
    for v in voci:
        ws.append(list(v))
    stile(ws, [26, 76, 76, 30, 34], 150)
    wb.save(p)
    return len(voci)


def main():
    os.makedirs(USCITA, exist_ok=True)
    righe = leggi_run()
    p1, n1 = scrivi_registro(righe)
    p2, n2 = scrivi_confronti(righe)
    p3, n3 = scrivi_sintesi(righe)
    n4 = scrivi_costo_per_sito()
    n5 = scrivi_utility_privacy()
    n6 = scrivi_worst_case()
    n8 = scrivi_rq2()
    n9 = scrivi_rq3()
    n7 = scrivi_glossario()
    print(f"registro run   : {n1} righe -> {p1}")
    print(f"confrontabilita: {n2} righe -> {p2}")
    print(f"sintesi        : {n3} righe -> {p3}")
    print(f"costo per sito : {n4} righe (foglio in Matrice_sintesi)")
    print(f"utility/privacy: {n5} righe (foglio in Matrice_sintesi)")
    print(f"worst-case     : {n6} righe (foglio in Matrice_sintesi)")
    print(f"RQ2 partizione : {n8} metriche (foglio in Matrice_sintesi)")
    print(f"RQ3 algoritmo  : {n9} metriche (foglio in Matrice_sintesi)")
    print(f"glossario      : {n7} voci (foglio in Matrice_sintesi)")
    stati = defaultdict(int)
    for r in righe:
        stati[r["stato"]] += 1
    print("\nstati di validita':")
    for k, v in sorted(stati.items()):
        print(f"   {k:28s} {v}")


if __name__ == "__main__":
    main()
