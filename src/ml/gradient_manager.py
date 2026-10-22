# src/ml/gradient_manager.py
# ChargeShield-FL — ML Plane: Gradient Manager
#
# Responsabilità (Purdue L1→L2):
#   - riceve GradientUpdate da AutoencoderTrainer
#   - applica gradient clipping (norma L2)
#   - applica Gaussian noise (Differential Privacy)
#   - emette GradientUpdate privatizzato verso FedAvgAggregator
#   - non conosce NVFLARE, né Auditor, né IDS

from __future__ import annotations

import logging
import math
from typing import Any

import torch

from ml.base_ml import (
    AbstractMLModel,
    AggregatedUpdate,
    GradientUpdate,
    MLPlaneEvent,
    MLPlaneListener,
)

logger = logging.getLogger(__name__)


class GradientManager(AbstractMLModel):
    """
    Applica Differential Privacy (Gaussian Mechanism) ai pesi locali
    prima che escano dal nodo verso l'aggregatore.

    Purdue Level: L1 (Control) → L2 (Supervisory)
    ML Plane role: privatizza GradientUpdate, emette evento DP.

    Parametri da config (nessun hardcoded):
        epsilon      : budget privacy
        delta        : probabilità fallimento DP
        max_grad_norm: soglia clipping norma L2
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Args:
            config: dizionario con epsilon, delta, max_grad_norm
        """
        epsilon       = config.get("epsilon")
        delta         = config.get("delta")
        max_grad_norm = config.get("max_grad_norm")

        if epsilon is None:
            raise ValueError("config['epsilon'] è obbligatorio")
        if delta is None:
            raise ValueError("config['delta'] è obbligatorio")
        if max_grad_norm is None:
            raise ValueError("config['max_grad_norm'] è obbligatorio")

        self.epsilon       = epsilon
        self.delta         = delta
        self.max_grad_norm = max_grad_norm
        # Calibrazione (2026-10-16, Segnalazioni R voce 78): "analitica" (default,
        # Balle e Wang 2018, adiacenze dichiarate in src/ml/client_dp_accounting.py)
        # o "classica" (formula precedente, solo per riprodurre le run vecchie).
        self.sigma_calibration = str(config.get("sigma_calibration", "analitica"))
        if self.sigma_calibration not in ("analitica", "classica"):
            raise ValueError(f"sigma_calibration non valida: {self.sigma_calibration!r}")

        # sigma per client (dp-fedavg/local): sensibilita' 2C con l'analitica
        # (sostituzione), C con la classica (comportamento precedente)
        self.sigma = self._compute_sigma()

        self._listeners: list[MLPlaneListener] = []

        logger.info(
            f"GradientManager — ε={epsilon}, δ={delta}, "
            f"max_norm={max_grad_norm}, σ={self.sigma:.4f}"
        )
        # NOTA: questo modulo implementa weight perturbation (rumore sui pesi
        # aggregati post-FedAvg), NON DP-SGD (rumore sui gradienti per-campione
        # durante il training). I bound di privacy sono diversi:
        # - DP-SGD: garanzia per ogni campione nel training set
        # - Weight perturbation: garanzia sul modello globale aggregato
        # Documentare questa distinzione nella sezione Privacy del paper.
        # NOTA composizione: sigma è calcolato per il Gaussian Mechanism a singolo round.
        # Con T round, la garanzia degrada. Usare RDP/zCDP per composizione formale.

    # ── AbstractMLModel ────────────────────────────────────────────────────────

    def get_weights(self) -> list[Any]:
        raise NotImplementedError("GradientManager non possiede un modello diretto.")

    def set_weights(self, weights: list[Any]) -> None:
        raise NotImplementedError("GradientManager non possiede un modello diretto.")

    def train_step(self, data: Any) -> float:
        raise NotImplementedError("GradientManager non esegue training.")

    def emit_event(self, event: MLPlaneEvent) -> None:
        for listener in self._listeners:
            listener.on_ml_event(event)

    def subscribe(self, listener: MLPlaneListener) -> None:
        self._listeners.append(listener)

    # ── DP API ────────────────────────────────────────────────────────────────

    def privatize(
        self,
        update: GradientUpdate,
        weight_keys: list[str] | None = None,
        reference_weights: list[Any] | None = None,
    ) -> GradientUpdate:
        """
        Applica gradient clipping + Gaussian noise ai pesi del GradientUpdate.

        Args:
            update:      GradientUpdate grezzo da AutoencoderTrainer
            weight_keys: chiavi dello state_dict corrispondenti a update.weights.
                         Se fornite, i buffer BatchNorm (running_mean, running_var,
                         num_batches_tracked) vengono ESCLUSI dal rumore DP:
                         aggiungere rumore Gaussiano a running_var la renderebbe
                         negativa (σ >> var tipica), causando NaN in sqrt durante
                         la forward pass in eval mode.
            reference_weights: pesi del modello globale ricevuto dal client
                         all'INIZIO di questo round (prima del training locale),
                         stesso ordine/formato di update.weights. Se forniti,
                         il clipping (Step 1) limita la norma del DELTA
                         (update.weights - reference_weights), non del vettore
                         pesi assoluto — vedi `_clip_weights()` per il perché
                         (fix 2026-07-22, review indipendente). Se None,
                         fallback al clipping assoluto (comportamento storico,
                         retro-compatibile con chiamate che non hanno un
                         riferimento disponibile).

        Returns:
            GradientUpdate con pesi privatizzati (DP garantita)
        """
        if not update.weights:
            logger.warning(f"[{update.node_id}] Pesi vuoti — skip DP")
            # Fix 2026-07-22 (review post-wiring ML Plane, commit 1ca713d):
            # senza questa emit_event(), un update con pesi vuoti sarebbe
            # silenziosamente assente da collector.privatized_updates(round_num)
            # — non solo "non privatizzato", ma proprio mancante dalla lista
            # "updates" che FedMIA/run_ids() consumano, cambiando silenziosamente
            # `n` per tutti gli altri client quel round (stessa classe di bug
            # trovata in AutoencoderTrainer.train_local(), vedi commento lì).
            # Riemettiamo `update` invariato come purdue_level=2: è la scelta
            # corretta perché non c'è nulla da privatizzare (pesi vuoti), quindi
            # "non privatizzato" È il dato reale osservato in questo round.
            self.emit_event(MLPlaneEvent(
                event_type="gradient_upload",
                purdue_level=2,
                payload=update,
                round_num=update.round_num,
                metadata={"noise_perturbation_applied": False, "empty_weights_skip": True},
            ))
            return update

        # Step 1: clip norma L2 del delta rispetto al modello ricevuto a inizio
        # round (o, se reference_weights non è disponibile, del vettore pesi
        # assoluto — comportamento storico, vedi _clip_weights()).
        clipped = self._clip_weights(update.weights, reference=reference_weights, weight_keys=weight_keys)

        # Step 2: aggiungi rumore Gaussiano (escludi buffer BN se keys note)
        noised = self._add_noise(clipped, weight_keys=weight_keys)

        privatized = GradientUpdate(
            node_id=update.node_id,
            cluster_id=update.cluster_id,
            round_num=update.round_num,
            weights=noised,
            gradients=None,
            loss=update.loss,
            n_samples=update.n_samples,
            metadata={
                **update.metadata,
                # "noise_perturbation_applied" (non "dp_applied") riflette che
                # il meccanismo è weight perturbation (Geyer 2017, Wei 2020), non DP-SGD.
                # Per claims DP formali vedere le note in _compute_sigma().
                "noise_perturbation_applied": True,
                "epsilon": self.epsilon,
                "delta": self.delta,
                "sigma": self.sigma,
                "max_grad_norm": self.max_grad_norm,
            },
        )

        # Emetti evento ML Plane
        self.emit_event(MLPlaneEvent(
            event_type="gradient_upload",
            purdue_level=2,
            payload=privatized,
            round_num=update.round_num,
            metadata={"noise_perturbation_applied": True},
        ))

        logger.debug(
            f"[{update.node_id}] DP applicato — "
            f"σ={self.sigma:.4f}, norm_clip={self.max_grad_norm}"
        )

        return privatized

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _compute_sigma(self) -> float:
        """
        Calcola σ per il Gaussian Mechanism.
        σ = max_grad_norm * sqrt(2 * ln(1.25 / δ)) / ε

        ATTENZIONE — weight perturbation, non DP-SGD:
        Questo implementa rumore sull'intero vettore pesi dopo il training locale,
        non per-sample gradient clipping (DP-SGD). Con epochs > 1 per round,
        la sensitività del vettore pesi a un singolo campione NON è formalmente
        bounded da max_grad_norm. La garanzia (ε,δ)-DP formale vale solo per
        epochs=1. Per il paper: descrivere ε come parametro di rumore sperimentale,
        non come garanzia DP formale. Il budget totale su T round sotto composizione
        naive è ε_tot ≈ T × ε_per_round — riportarlo nelle tabelle dei risultati.
        """
        if self.epsilon <= 0:
            raise ValueError(f"epsilon deve essere > 0, ricevuto: {self.epsilon}")
        if not (0 < self.delta < 1):
            raise ValueError(f"delta deve essere in (0, 1), ricevuto: {self.delta}")
        if self.delta > 1e-2:
            logger.warning(
                f"delta={self.delta} è insolitamente alto per DP — "
                "valori tipici sono 1e-5 o inferiori. Verificare la configurazione."
            )
        from ml.client_dp_accounting import sigma_classico, sigma_gaussiano_analitico
        if self.sigma_calibration == "classica":
            self._sigma_unit = sigma_classico(self.epsilon, self.delta, 1.0)
            return self.max_grad_norm * self._sigma_unit
        self._sigma_unit = sigma_gaussiano_analitico(self.epsilon, self.delta, 1.0)
        return 2.0 * self.max_grad_norm * self._sigma_unit

    def clip_only(
        self,
        update: GradientUpdate,
        weight_keys: list[str] | None = None,
        reference_weights: list[Any] | None = None,
    ) -> GradientUpdate:
        """
        Applica SOLO il clipping L2 (NESSUN rumore) a un GradientUpdate.

        Usato dalla modalità **central DP** (2026-07-22): a differenza di
        DP-FedAvg (`privatize()`, rumore per-client prima dell'aggregazione),
        sotto central DP ogni client clippa il proprio update per limitare la
        sensitività, ma il rumore viene aggiunto UNA SOLA VOLTA sull'aggregato
        FedAvg (vedi `privatize_aggregate()`), non per client. Il clipping resta
        necessario anche qui: senza un bound sulla norma del singolo update, la
        sensitività della media aggregata (e quindi il σ da usare sull'aggregato)
        non sarebbe definita.

        Args:
            update:       GradientUpdate grezzo da AutoencoderTrainer
            weight_keys:  chiavi state_dict — esclude i buffer BatchNorm dalla
                          norma clippata (vedi `_clip_weights()`).
            reference_weights: pesi del modello globale ricevuto a inizio round
                          — se forniti, clippa il DELTA invece del vettore
                          assoluto (fix 2026-07-22, vedi `_clip_weights()`).

        Returns:
            GradientUpdate con pesi clippati, NON rumorizzati.
        """
        if not update.weights:
            logger.warning(f"[{update.node_id}] Pesi vuoti — skip clip")
            # Fix 2026-07-22 — stesso motivo dell'analogo early-return in
            # privatize() qui sopra: senza emit_event() questo nodo sarebbe
            # silenziosamente assente da collector.privatized_updates(round_num)
            # in modalità central DP.
            self.emit_event(MLPlaneEvent(
                event_type="gradient_upload",
                purdue_level=2,
                payload=update,
                round_num=update.round_num,
                metadata={"clipped_only_central_dp": True, "empty_weights_skip": True},
            ))
            return update

        clipped = self._clip_weights(update.weights, reference=reference_weights, weight_keys=weight_keys)

        clipped_update = GradientUpdate(
            node_id=update.node_id,
            cluster_id=update.cluster_id,
            round_num=update.round_num,
            weights=clipped,
            gradients=None,
            loss=update.loss,
            n_samples=update.n_samples,
            metadata={
                **update.metadata,
                # Distingue da "noise_perturbation_applied" (DP-FedAvg/local):
                # qui il client NON ha aggiunto rumore, solo clipping.
                "clipped_only_central_dp": True,
                "max_grad_norm": self.max_grad_norm,
            },
        )

        # FIX 2026-07-22 (ML Plane realmente usato): a differenza di
        # privatize(), clip_only() non emetteva MAI un evento ML Plane —
        # in modalità "central" DP (l'unica che chiama questo metodo,
        # vedi run_fl_rounds()) il FLArtifactCollector non aveva quindi
        # nessuna visibilità sull'update clippato-ma-pulito, solo su quello
        # raw pre-clip (da AutoencoderTrainer). Aggiunto per parità con
        # privatize(): stesso event_type/purdue_level, il consumatore
        # distingue le due modalità dal metadata.
        self.emit_event(MLPlaneEvent(
            event_type="gradient_upload",
            purdue_level=2,
            payload=clipped_update,
            round_num=update.round_num,
            metadata={"clipped_only_central_dp": True},
        ))

        return clipped_update

    # ── Ablazioni del meccanismo per client (2026-10-05, segnalazione 38) ──────
    # Per separare il costo del taglio da quello del rumore nel meccanismo per
    # client (dp-fedavg/local), servono i due pezzi di privatize() da soli.
    # Nessuna delle due ablazioni da' una garanzia DP: clip_no_noise() non
    # aggiunge rumore, noise_no_clip() non limita la sensibilita'. Il rumore di
    # noise_no_clip() resta tarato sul max_grad_norm configurato (stesso sigma
    # di privatize()), cosi' "solo rumore" ha esattamente il rumore della DP
    # completa. Gli eventi ML Plane sono emessi a purdue_level=2 come per
    # privatize(), quindi FLArtifactCollector li tratta come update inviati.

    def clip_no_noise(
        self,
        update: GradientUpdate,
        weight_keys: list[str] | None = None,
        reference_weights: list[Any] | None = None,
    ) -> GradientUpdate:
        """Ablazione "solo taglio": clipping del delta come in privatize(), nessun rumore."""
        if not update.weights:
            logger.warning(f"[{update.node_id}] Pesi vuoti — skip clip (ablazione)")
            return update
        clipped = self._clip_weights(update.weights, reference=reference_weights, weight_keys=weight_keys)
        out = GradientUpdate(
            node_id=update.node_id,
            cluster_id=update.cluster_id,
            round_num=update.round_num,
            weights=clipped,
            gradients=None,
            loss=update.loss,
            n_samples=update.n_samples,
            metadata={
                **update.metadata,
                "dp_ablation": "clip-only",
                "noise_perturbation_applied": False,
                "max_grad_norm": self.max_grad_norm,
            },
        )
        self.emit_event(MLPlaneEvent(
            event_type="gradient_upload",
            purdue_level=2,
            payload=out,
            round_num=update.round_num,
            metadata={"dp_ablation": "clip-only", "noise_perturbation_applied": False},
        ))
        return out

    def noise_no_clip(
        self,
        update: GradientUpdate,
        weight_keys: list[str] | None = None,
    ) -> GradientUpdate:
        """Ablazione "solo rumore": stesso rumore di privatize() (sigma tarato su
        max_grad_norm), nessun clipping. Buffer BatchNorm esclusi come in _add_noise()."""
        if not update.weights:
            logger.warning(f"[{update.node_id}] Pesi vuoti — skip rumore (ablazione)")
            return update
        tensors = [
            (w if isinstance(w, torch.Tensor) else torch.tensor(w)).detach().clone()
            for w in update.weights
        ]
        noised = self._add_noise(tensors, weight_keys=weight_keys)
        out = GradientUpdate(
            node_id=update.node_id,
            cluster_id=update.cluster_id,
            round_num=update.round_num,
            weights=noised,
            gradients=None,
            loss=update.loss,
            n_samples=update.n_samples,
            metadata={
                **update.metadata,
                "dp_ablation": "noise-only",
                "noise_perturbation_applied": True,
                "epsilon": self.epsilon,
                "delta": self.delta,
                "sigma": self.sigma,
                "max_grad_norm": self.max_grad_norm,
            },
        )
        self.emit_event(MLPlaneEvent(
            event_type="gradient_upload",
            purdue_level=2,
            payload=out,
            round_num=update.round_num,
            metadata={"dp_ablation": "noise-only", "noise_perturbation_applied": True},
        ))
        return out

    def privatize_aggregate(
        self,
        global_weights: list[Any],
        weight_keys: list[str] | None = None,
        n_participants: int = 1,
        participant_n_samples: list[int] | None = None,
    ) -> list[torch.Tensor]:
        """
        Aggiunge UN SOLO rumore Gaussiano all'aggregato FedAvg (central DP).

        Central DP [McMahan et al. 2018, "Learning Differentially Private
        Recurrent Language Models"]: il server (trusted) riceve gli update
        raw-ma-clippati (vedi `clip_only()`) di tutti i client, calcola la
        media pesata pulita, poi aggiunge UN SOLO draw di rumore Gaussiano
        all'aggregato — non uno per client.

        FIX 2026-08-31 (Fase 8 — chiude il "LIMITE NOTO" segnalato 2026-07-24
        e mai risolto): la sensitività della media NON è max_grad_norm/n
        (quello vale solo per una media UNIFORME, pesi 1/n ciascuno).
        `FedAvgAggregator.aggregate()` fa una media pesata per n_samples
        (McMahan et al. 2017); con siti reali ACN-Data di dimensione molto
        diversa (Office1 un ordine di grandezza più piccolo di Caltech/JPL —
        vedi README "Dataset"), il "replace-one" worst case è il client con
        il peso maggiore, non 1/n: la vera sensitività è
        max_grad_norm × max_i(n_i / N_totale), dove n_i è il conteggio
        campioni del client i nel round e N_totale la somma di tutti.
        `participant_n_samples` (lista di n_samples per client di QUESTO
        round — reperibile da `AggregatedUpdate.metadata["participant_n_samples"]`,
        popolato da `FedAvgAggregator.aggregate()`) rende questo calcolabile
        realmente, non assunto. Se non fornito (retro-compatibilità con
        chiamate esistenti/test), ricade sul comportamento storico
        (sigma_central = sigma/n_participants, corretto SOLO se i client
        pesano davvero 1/n ciascuno) con un warning esplicito — mai un
        fallback silenzioso.

        Impatto sui risultati già pubblicati: questo fix cambia la MAGNITUDO
        del rumore aggiunto sotto `dp_mode="central"` (più rumore quando i
        pesi sono sbilanciati, mai meno) — non tocca in alcun modo la
        misurazione di leakage (LiRA/Yeom/Shadow operano sull'AUC empirico,
        indipendente da questa formula). I risultati Central DP della
        campagna 5-seed×8-config (Sprint 10tt) sono stati prodotti con la
        formula PRECEDENTE (sotto-stimava lievemente il rumore necessario per
        l'ε nominale dichiarato) — l'AUC composito misurato (≈0.5, nessun
        leakage) non è messo in discussione da questo fix (più rumore avrebbe
        semmai spinto l'AUC ancora più vicino a 0.5, mai sopra la soglia), ma
        l'ε formale dichiarato per quei run specifici era leggermente
        ottimistico rispetto al rumore realmente applicato — da annotare
        esplicitamente nel paper (vedi README Sprint-log per la data di
        questo fix) se si vuole citare l'ε nominale come garanzia esatta per
        quei run storici, oppure rilanciare Central DP con questo fix per
        numeri pienamente coerenti.

        Args:
            global_weights:  pesi aggregati (puliti) da FedAvgAggregator
            weight_keys:     chiavi state_dict, per escludere i buffer BatchNorm
            n_participants:  numero di client aggregati questo round — usato
                             SOLO come fallback storico se participant_n_samples
                             non è fornito.
            participant_n_samples: n_samples di ciascun client partecipante a
                             QUESTO round — se fornito, la sensitività usa
                             max_i(n_i/N) invece di 1/n_participants.

        Returns:
            Lista di tensori con rumore Gaussiano aggiunto (buffer BN esclusi).
        """
        if participant_n_samples:
            total = sum(participant_n_samples) or 1
            max_weight_fraction = max(participant_n_samples) / total
        else:
            logger.warning(
                "privatize_aggregate() chiamato senza participant_n_samples — "
                "ricado sul fallback storico sigma/n_participants, corretto "
                "SOLO se i client pesano davvero 1/n_participants ciascuno "
                "(non il caso dei 3 siti reali ACN-Data, dimensioni sbilanciate)."
            )
            max_weight_fraction = 1.0 / max(1, n_participants)

        # analitica: sostituzione dei dati di un client (pesi n_i pubblici e fissi,
        # tutti i client partecipano), D = 2 w_max C (revisione 2026-10-16)
        sigma_central = (self.sigma * max_weight_fraction if self.sigma_calibration == "classica"
                         else 2.0 * self._sigma_unit * self.max_grad_norm * max_weight_fraction)
        tensors = [
            w if isinstance(w, torch.Tensor) else torch.tensor(w)
            for w in global_weights
        ]
        noised = self._add_noise(tensors, weight_keys=weight_keys, sigma=sigma_central)
        logger.debug(
            f"Central DP — rumore sull'aggregato: σ_central={sigma_central:.4f} "
            f"(σ_singolo={self.sigma:.4f} × max_weight_fraction={max_weight_fraction:.4f})"
        )
        return noised

    def _clip_weights(
        self,
        weights: list[Any],
        reference: list[Any] | None = None,
        weight_keys: list[str] | None = None,
    ) -> list[torch.Tensor]:
        """
        Clippa la norma L2 a max_grad_norm.

        FIX 2026-07-22 (review indipendente, punto B1 di docs/CaseStudies.md
        §2.4.3) — due problemi distinti nella versione precedente, entrambi
        corretti qui:

        1) **Clippava il vettore pesi ASSOLUTO invece del DELTA**. La
           costruzione canonica (McMahan et al., citata da `privatize_aggregate()`)
           bound la sensitività del contributo del client, cioè quanto il suo
           update si allontana dal modello globale ricevuto — non la norma
           del modello stesso. Clippare l'assoluto, con max_grad_norm=1.0 su
           un modello di ~650 parametri, fa scattare quasi sempre factor<1,
           comprimendo l'INTERO modello verso una palla di norma 1 a ogni
           round — un'operazione molto più aggressiva di un bound sull'
           incremento per round, che può da sola spiegare parte dell'effetto
           "la DP sopprime la MIA" osservato, confondendolo con l'effetto del
           rumore Gaussiano.
           Fix: se `reference` è fornito, si clippa `weights - reference`
           (il vero "contributo" del client) e si ricostruisce
           `reference + delta_clippato` — il modello resta vicino al globale
           ricevuto, non viene tirato verso zero. Se `reference` è None
           (nessun riferimento disponibile), fallback al comportamento
           storico (clip assoluto) per retro-compatibilità.

        2) **I buffer BatchNorm entravano nella norma clippata**, mentre
           `_add_noise()` li esclude già dal rumore (vedi il motivo lì:
           running_var negativa → NaN). Includerli nel calcolo della norma
           "diluisce" il budget di clipping disponibile per i parametri
           veri, e — nel vecchio clipping assoluto — li faceva restringere
           verso zero ad ogni round (running_var sempre più piccola,
           normalizzazione BatchNorm sempre più aggressiva in eval mode).
           Fix: se `weight_keys` è fornito, i buffer BatchNorm sono esclusi
           dal calcolo della norma E restituiti invariati (valore locale,
           non quello di riferimento) — stessa filosofia di `_add_noise()`.
           Se `weight_keys` è None, fallback: tutti i tensori float
           partecipano alla norma (comportamento storico).

        I buffer int64 (num_batches_tracked) sono sempre esclusi e restituiti
        invariati — non sono float, non attraversano né clip né rumore.
        """
        tensors = [w if isinstance(w, torch.Tensor) else torch.tensor(w)
                   for w in weights]
        _BN_BUFFERS = {"running_mean", "running_var", "num_batches_tracked"}

        def _is_bn_buffer(idx: int) -> bool:
            return (
                weight_keys is not None
                and idx < len(weight_keys)
                and weight_keys[idx].split(".")[-1] in _BN_BUFFERS
            )

        # Indici dei tensori float "clippabili" (parametri veri, non buffer BN)
        clip_idx = [
            i for i, t in enumerate(tensors)
            if t.is_floating_point() and not _is_bn_buffer(i)
        ]
        if not clip_idx:
            return tensors

        if reference is not None:
            # ── Modalità corretta: clip del DELTA rispetto al modello ricevuto ──
            ref_tensors = [
                r if isinstance(r, torch.Tensor) else torch.tensor(r)
                for r in reference
            ]
            deltas = [tensors[i].float() - ref_tensors[i].float() for i in clip_idx]
            flat   = torch.cat([d.flatten() for d in deltas])
            norm   = torch.norm(flat, p=2)
            factor = min(1.0, self.max_grad_norm / (float(norm) + 1e-8))

            result = list(tensors)  # copia; sovrascriviamo solo gli indici clippati
            for i, d in zip(clip_idx, deltas):
                result[i] = ref_tensors[i].float() + d * factor
            return result

        # ── Fallback storico: clip del vettore assoluto (nessun reference) ──
        # Usato quando il chiamante non ha un modello di riferimento disponibile
        # (retro-compatibilità con codice/test esistenti che chiamano
        # _clip_weights()/privatize()/clip_only() senza reference_weights).
        float_tensors = [tensors[i] for i in clip_idx]
        flat   = torch.cat([t.flatten().float() for t in float_tensors])
        norm   = torch.norm(flat, p=2)
        factor = min(1.0, self.max_grad_norm / (float(norm) + 1e-8))

        result = list(tensors)
        for i in clip_idx:
            result[i] = tensors[i] * factor
        return result

    def _add_noise(
        self,
        weights: list[torch.Tensor],
        weight_keys: list[str] | None = None,
        sigma: float | None = None,
    ) -> list[torch.Tensor]:
        """
        Aggiunge rumore Gaussiano N(0, σ²) solo ai tensori floating-point,
        escludendo i buffer BatchNorm quando le chiavi dello state_dict sono note.

        Args:
            sigma: override esplicito di σ (usato da `privatize_aggregate()` per
                   il central DP, dove σ è scalato 1/n_participants rispetto a
                   `self.sigma`). Se None, usa `self.sigma` (comportamento
                   originale, DP-FedAvg/local per-client).

        Motivazione dell'esclusione BatchNorm:
            running_var è una varianza (tipicamente ~0.1–1.0 per dati normalizzati).
            Aggiungere rumore con σ >> running_var (es. σ=48 per ε=0.1) rende
            running_var negativa. BatchNorm in eval mode calcola:
                y = (x - running_mean) / sqrt(running_var + eps)
            Se running_var + eps < 0, sqrt() → NaN, invalidando tutte le score MIA.
            I buffer BN (running_mean, running_var, num_batches_tracked) sono
            statistiche ausiliarie, non parametri DP-perturbabili.

        I buffer int64 (num_batches_tracked) vengono già esclusi perché
        torch.randn_like() su int64 solleva RuntimeError in PyTorch ≥2.0.
        """
        _sigma = sigma if sigma is not None else self.sigma
        _BN_BUFFERS = {"running_mean", "running_var", "num_batches_tracked"}
        result: list[torch.Tensor] = []
        for i, w in enumerate(weights):
            if not w.is_floating_point():
                result.append(w)
            elif (
                weight_keys is not None
                and i < len(weight_keys)
                and weight_keys[i].split(".")[-1] in _BN_BUFFERS
            ):
                # Buffer BatchNorm: restituiti invariati, non sono parametri DP
                result.append(w)
            else:
                result.append(w + torch.randn_like(w) * _sigma)
        return result
