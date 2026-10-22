# src/ml/autoencoder_trainer.py
# ChargeShield-FL — ML Plane: Autoencoder Trainer
#
# Responsabilità (Purdue L0→L1):
#   - riceve sessioni EV grezze dal layer Dataset
#   - esegue il training loop locale dell'Autoencoder
#   - supporta FedAvg (proximal_mu=0) e FedProx (proximal_mu>0)
#   - emette GradientUpdate verso il ML Plane
#   - non conosce NVFLARE, né Auditor, né IDS

from __future__ import annotations

import logging
import math
import zlib
from typing import Any

import torch
from torch.utils.data import DataLoader, TensorDataset

from core.autoencoder import Autoencoder
from ml.subject_dp import record_dp_on, row_valid, sampling_schedule, subject_dp_config, subject_ids
from ml.base_ml import (
    AbstractMLModel,
    AggregatedUpdate,
    GradientUpdate,
    MLPlaneEvent,
    MLPlaneListener,
)

logger = logging.getLogger(__name__)


class AutoencoderTrainer(AbstractMLModel):
    """
    Training locale dell'Autoencoder su sessioni EV (ACN-Data).

    Purdue Level: L0 (Field) → L1 (Control)
    ML Plane role: produce GradientUpdate dopo ogni training locale.

    Parametri da config (nessun hardcoded):
        input_dim     : dimensione input
        lr            : learning rate
        epochs        : epoche di training locale per round FL
        batch_size    : dimensione batch
        proximal_mu   : coefficiente FedProx (0.0 = FedAvg puro)
    """

    CONTINUOUS_FEATURES = [
        "total_energy_kwh",   # kWh erogati (ACN: kWhDelivered)
        "max_power_kw",       # potenza media stimata
        "kwh_requested",      # energia richiesta dall'utente
        "minutes_available",  # minuti disponibili dichiarati
        "hour_of_day",        # ora connessione (0–23) — pattern comportamentale
        "duration_hours",     # durata sessione in ore
    ]

    def __init__(self, config: dict[str, Any], node_id: str, cluster_id: str) -> None:
        self.node_id    = node_id
        self.cluster_id = cluster_id

        input_dim  = config.get("input_dim")
        lr         = config.get("lr")
        epochs     = config.get("epochs")
        batch_size = config.get("batch_size")

        if input_dim is None:
            raise ValueError("config['input_dim'] è obbligatorio")
        if lr is None:
            raise ValueError("config['lr'] è obbligatorio")
        if epochs is None:
            raise ValueError("config['epochs'] è obbligatorio")
        if batch_size is None:
            raise ValueError("config['batch_size'] è obbligatorio")

        self.epochs     = epochs
        self.batch_size = batch_size

        # FedProx: 0.0 = FedAvg puro. Usare .get() con default — "or" ignorerebbe 0.0 intenzionale.
        self.proximal_mu: float = config.get("proximal_mu", 0.0)

        # ── DP a livello di RECORD (2026-09-16, Sprint 10zz+116) ──────────────
        # Opt-in via cfg["ml"]["record_dp"]. Assente o enabled=False (ogni
        # config esistente) => comportamento invariato bit-per-bit: train_step()
        # resta quello di prima e questo blocco e' inerte.
        #
        # PERCHE' ESISTE: il meccanismo DP principale di questo progetto clippa
        # e rumorizza UNA VOLTA PER ROUND l'update aggregato del client, cioe'
        # implementa un'adiacenza CLIENT-level (vedi §Preliminaries del paper).
        # Gli attacchi invece sondano l'appartenenza di un RECORD. Quel
        # disallineamento e' una domanda aperta del paper, non un'assunzione, e
        # per rispondervi serve il braccio mancante: DP-SGD vero, con clipping
        # del gradiente PER ESEMPIO a ogni passo locale. Non e' un'analisi piu'
        # fine del meccanismo esistente, e' un meccanismo diverso.
        #
        # PARAMETRI (parametrizzazione standard DP-SGD, Abadi et al. 2016):
        #   max_grad_norm C  : ogni gradiente per-esempio e' clippato a ||g||<=C
        #   noise_multiplier : std del rumore gaussiano aggiunto alla SOMMA dei
        #                      gradienti clippati = noise_multiplier * C
        # Il gradiente del batch e' poi (somma_clippata + rumore) / batch_size.
        #
        # LIMITE DA DICHIARARE: qui NON c'e' un accountant RDP/moments. Questo
        # blocco implementa il MECCANISMO, non la contabilita' del budget: il
        # (eps, delta) risultante andrebbe calcolato con un accountant vero
        # (Opacus/dp-accounting) e non e' derivabile dai due parametri sopra
        # senza amplificazione da sottocampionamento. Vale quindi come braccio
        # sperimentale ("il clipping per-record sopprime il segnale canary?"),
        # non come claim di garanzia formale.
        _rdp = config.get("record_dp") or {}
        self.record_dp_enabled: bool = record_dp_on(config)  # lettura stretta (2026-10-16)
        self.record_dp_max_grad_norm: float = float(_rdp.get("max_grad_norm", 1.0))
        self.record_dp_noise_multiplier: float = float(_rdp.get("noise_multiplier", 0.0))
        if self.record_dp_enabled and self.record_dp_max_grad_norm <= 0.0:
            raise ValueError(
                "config['ml']['record_dp']['max_grad_norm'] deve essere > 0 "
                f"(ricevuto {self.record_dp_max_grad_norm})"
            )
        self._global_weights: list[Any] | None = None

        # ── DP a livello di SOGGETTO (2026-10-13) ─────────────────────────────
        # Opt-in via cfg["ml"]["subject_dp"]; assente o enabled=False => blocco
        # inerte e comportamento invariato. Meccanismo, accounting e definizione
        # di soggetto in src/ml/subject_dp.py. Alternativa alla DP per record,
        # non combinabile con essa: le due unita' protette sono diverse.
        self.subject_dp: dict[str, Any] = subject_dp_config(config)
        self.subject_dp_enabled: bool = self.subject_dp["enabled"]
        if self.subject_dp_enabled and self.record_dp_enabled:
            raise ValueError(
                "config['ml']: record_dp e subject_dp non possono essere attivi "
                "insieme (proteggono unita' diverse; scegliere una cella per volta)"
            )
        if self.subject_dp_enabled:
            if float(self.subject_dp["max_grad_norm"]) <= 0.0:
                raise ValueError(
                    "config['ml']['subject_dp']['max_grad_norm'] deve essere > 0 "
                    f"(ricevuto {self.subject_dp['max_grad_norm']})"
                )
            if int(self.subject_dp["max_records_per_subject"]) < 1:
                raise ValueError(
                    "config['ml']['subject_dp']['max_records_per_subject'] deve essere >= 1"
                )
        # Pesi dei soggetti presenti in piu' client ({soggetto: w}, w < 1),
        # impostati da run_fl_rounds() con set_subject_weights(). Assente => 1.
        self._subject_weights: dict[str, float] = {}
        self.subject_dp_last: dict[str, Any] | None = None
        # Celle con DP (2026-10-16): la loss locale resta nel client, non va
        # nell'update che esce (e nemmeno negli eventi del ML Plane).
        self.trasmetti_loss: bool = bool(config.get("trasmetti_loss", True))
        self.ultima_loss_locale: float | None = None
        self.record_dp_last: dict[str, Any] | None = None

        # Seed per DataLoader shuffle deterministico (riproducibilità DSN 2027)
        self._seed: int = config.get("seed", 42)

        # feature_names: escalation di entropia feature opzionale (Sprint 10kk,
        # 2026-08-28 — vedi docs/TestRoadmap_DSN2027.md #2, passo 2), usata solo
        # dagli esperimenti di calibrazione LiRA che la impostano esplicitamente
        # in YAML. Default None → CONTINUOUS_FEATURES di classe (le 6 feature
        # storiche), invariato per ogni config esistente. Sovrascrive l'attributo
        # SOLO su questa istanza (shadowing), non la classe — _sessions_to_tensor()
        # legge già self.CONTINUOUS_FEATURES, quindi nessun'altra modifica serve.
        feature_names = config.get("feature_names")
        if feature_names is not None:
            self.CONTINUOUS_FEATURES = list(feature_names)

        # hidden_dims/latent_dim: escalation di capacità opzionale (Sprint 10jj,
        # 2026-08-28), usata solo dagli esperimenti di calibrazione LiRA che la
        # impostano esplicitamente in YAML. Default None/4 → architettura
        # storica invariata (16, 8)/4, identica per tutti i run esistenti.
        hidden_dims = config.get("hidden_dims")
        latent_dim  = config.get("latent_dim", 4)
        if hidden_dims is not None:
            hidden_dims = tuple(hidden_dims)

        # norm (Sprint 10zz+113): "group" e' obbligatorio con record_dp
        # (microbatch da 1 + sensibilita' per-record non garantita da BatchNorm,
        # vedi src/core/autoencoder.py). Se record_dp e' attivo e la config non
        # lo specifica, forziamo "group" ed emettiamo un warning: e' preferibile
        # a un crash a meta' del primo round o, peggio, a una garanzia DP
        # silenziosamente invalida.
        # Default "group" dal 2026-10-16 (Segnalazioni R voce 74, decisione della
        # dottoranda: GroupNorm in tutte le celle). Le run precedenti usavano
        # "batch" senza dichiararlo: per riprodurle serve norm: batch nel config.
        _norm = str(config.get("norm", "group"))
        if self.record_dp_enabled and _norm != "group":
            logger.warning(
                "[RECORD-DP] norm=%r incompatibile con il clipping per-esempio: "
                "forzato a 'group'. BatchNorm richiede batch>1 e non limita la "
                "sensibilita' per-record. Dichiarare norm: group in config per "
                "rendere la scelta esplicita.", _norm
            )
            _norm = "group"
        if self.subject_dp_enabled and _norm != "group":
            # Stesso motivo: il gradiente di un soggetto si calcola sulle sole
            # sue sessioni (anche una sola), e BatchNorm non lo permette.
            logger.warning(
                "[SUBJECT-DP] norm=%r incompatibile con il clipping per soggetto: "
                "forzato a 'group'. Dichiarare norm: group in config.", _norm
            )
            _norm = "group"
        self.norm: str = _norm

        self.model     = Autoencoder(input_dim=input_dim, latent_dim=latent_dim,
                                     hidden_dims=hidden_dims, norm=_norm)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = torch.nn.MSELoss()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self._listeners: list[MLPlaneListener] = []
        self._current_round: int = 0

        logger.info(
            f"AutoencoderTrainer [{node_id}] — device={self.device}, "
            f"input_dim={input_dim}, lr={lr}, epochs={epochs}, "
            f"hidden_dims={hidden_dims or '(16, 8) [default]'}, latent_dim={latent_dim}, "
            f"feature_names={feature_names or '6 storiche [default]'}, "
            f"proximal_mu={self.proximal_mu} "
            f"({'FedProx' if self.proximal_mu > 0 else 'FedAvg'})"
        )

    # ── AbstractMLModel ────────────────────────────────────────────────────────

    def get_weights(self) -> list[Any]:
        """Restituisce tutti gli entry dello state_dict (parametri + buffer BN) come lista CPU.
        Usare state_dict invece di parameters() per trasferire anche i buffer BatchNorm."""
        return [v.clone().cpu() for v in self.model.state_dict().values()]

    def get_weight_keys(self) -> list[str]:
        """Restituisce le chiavi dello state_dict nell'ordine usato da get_weights().
        Passare a GradientManager.privatize(weight_keys=...) per escludere i buffer
        BatchNorm (running_mean, running_var) dal rumore DP Gaussiano."""
        return list(self.model.state_dict().keys())

    def set_weights(self, weights: list[Any]) -> None:
        """
        Carica i pesi globali nel modello locale via load_state_dict.
        Trasferisce anche i buffer BatchNorm (running_mean, running_var).
        Salva solo i parametri trainable per il termine prossimale FedProx.
        """
        orig_state = self.model.state_dict()
        keys = list(orig_state.keys())
        new_state: dict[str, torch.Tensor] = {}
        for k, w in zip(keys, weights):
            t = w if isinstance(w, torch.Tensor) else torch.tensor(w)
            new_state[k] = t.to(self.device).to(orig_state[k].dtype)
        self.model.load_state_dict(new_state, strict=True)
        # FedProx proximal term usa solo i parametri trainable
        self._global_weights = [p.data.clone().detach() for p in self.model.parameters()]

    def train_step(self, data: Any) -> float:
        """
        Singolo passo di training su un batch.
        Se proximal_mu > 0, aggiunge termine prossimale FedProx:
            loss += (mu/2) * ||w - w_global||²
        """
        if self.record_dp_enabled:
            return self._train_step_record_dp(data)

        self.model.train()
        batch = data.to(self.device)
        self.optimizer.zero_grad()
        reconstructed = self.model(batch)
        loss = self.criterion(reconstructed, batch)

        if self.proximal_mu > 0.0 and self._global_weights is not None:
            proximal_term = torch.tensor(0.0, device=self.device)
            for param, w_global in zip(self.model.parameters(), self._global_weights):
                proximal_term += torch.norm(param - w_global) ** 2
            loss += (self.proximal_mu / 2.0) * proximal_term

        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def _train_step_record_dp(self, data: Any) -> float:
        """
        Passo DP-SGD con clipping del gradiente PER ESEMPIO (Abadi et al. 2016).

        Differenza sostanziale rispetto a train_step(): li' il gradiente viene
        calcolato sulla media del batch e il clipping/rumore DP avviene altrove,
        una volta per round sull'update aggregato del client (adiacenza
        client-level). Qui ogni esempio contribuisce un gradiente clippato
        individualmente a norma <= C, e il rumore e' aggiunto alla somma dei
        gradienti clippati: l'influenza di un singolo record sull'update e'
        quindi limitata per costruzione a ogni passo, che e' la condizione
        perche' l'adiacenza sia RECORD-level.

        Implementazione: microbatch da 1. Piu' lenta di vmap/functorch di un
        fattore ~batch_size in numero di backward, ma esplicita e verificabile
        riga per riga --- scelta deliberata su un modello da <2k parametri,
        dove il costo assoluto resta modesto e la correttezza conta piu' della
        velocita'.

        FedProx: il termine prossimale (mu/2)||w - w_global||^2 NON entra nei
        gradienti per-esempio ed e' escluso dal clipping. Non dipende dai dati,
        quindi non ha sensibilita' per-record: clipparlo sarebbe concettualmente
        sbagliato, e includerlo in ogni microbatch lo moltiplicherebbe per
        batch_size. Viene percio' aggiunto una sola volta, al gradiente medio
        gia' rumorizzato, dove il suo effetto e' identico a quello che ha in
        train_step().
        """
        self.model.train()
        batch = data.to(self.device)
        n = batch.shape[0]
        if n == 0:
            return 0.0

        params = [p for p in self.model.parameters() if p.requires_grad]
        somma = [torch.zeros_like(p) for p in params]
        loss_totale = 0.0
        C = self.record_dp_max_grad_norm

        for i in range(n):
            self.optimizer.zero_grad(set_to_none=True)
            x = batch[i : i + 1]
            loss_i = self.criterion(self.model(x), x)
            loss_i.backward()
            loss_totale += float(loss_i.item())

            # norma L2 del gradiente di QUESTO esempio, su tutti i parametri
            norma = torch.sqrt(
                sum((p.grad.detach() ** 2).sum() for p in params if p.grad is not None)
            )
            _n = float(norma.item())
            if not math.isfinite(_n):      # 2026-10-16: min(1, nan) = 1 non taglierebbe
                raise RuntimeError("norma del gradiente per esempio non finita")
            scala = min(1.0, C / (_n + 1e-12))
            for acc, p in zip(somma, params):
                if p.grad is not None:
                    acc.add_(p.grad.detach() * scala)

        # Rumore gaussiano sulla SOMMA dei gradienti clippati, std = sigma * C
        # (parametrizzazione standard DP-SGD), poi media sul lotto.
        sigma = self.record_dp_noise_multiplier
        self.optimizer.zero_grad(set_to_none=True)
        # enumerate (Sprint 10zz+114): params.index(p) e' errato su tensori —
        # list.index() confronta con ==, e su un Tensor == e' un confronto
        # ELEMENTO PER ELEMENTO, non un test di identita'. Con parametri di
        # forma diversa il broadcasting fallisce ("The size of tensor a (7)
        # must match the size of tensor b (32)"). Il bug si manifestava solo
        # dal round 2, quando self._global_weights smette di essere None e il
        # termine prossimale FedProx diventa attivo. Era anche O(n) per
        # parametro per microbatch. zip itera params nell'ordine giusto,
        # quindi enumerate da' l'indice corretto a costo zero.
        for idx, (acc, p) in enumerate(zip(somma, params)):
            g = acc
            if sigma > 0.0:
                g = g + torch.normal(mean=0.0, std=sigma * C, size=g.shape, device=g.device)
            g = g / float(n)
            # Termine prossimale FedProx, aggiunto DOPO il rumore: vedi docstring.
            if self.proximal_mu > 0.0 and self._global_weights is not None:
                if idx < len(self._global_weights):
                    g = g + self.proximal_mu * (p.detach() - self._global_weights[idx])
            p.grad = g

        self.optimizer.step()
        return loss_totale / n

    # ── DP per record con campionamento di Poisson (2026-10-16) ────────────────
    #
    # Segnalazioni R voce 77: con lo shuffle a batch fissi l'epsilon di Poisson
    # dell'accountant non e' una garanzia. Da qui ogni passo estrae ogni record
    # con probabilita' q = B/n (lotto di dimensione variabile, anche vuoto),
    # passi per epoca = floor(n/B), esattamente i parametri di
    # src/ml/record_dp_accounting.py. Gradienti per esempio con torch.func.vmap
    # (stesse somme del ciclo a 1e-8, circa 7 volte piu' veloce); il ciclo resta
    # in _record_dp_clipped_sum_loop come riferimento dei test.

    def _record_dp_clipped_sum_loop(self, batch: torch.Tensor) -> tuple[list[torch.Tensor], list[float]]:
        """Somma dei gradienti per esempio tagliati a C, un esempio alla volta."""
        params = [p for p in self.model.parameters() if p.requires_grad]
        somma = [torch.zeros_like(p) for p in params]
        losses: list[float] = []
        C = self.record_dp_max_grad_norm
        for i in range(batch.shape[0]):
            self.optimizer.zero_grad(set_to_none=True)
            x = batch[i : i + 1]
            loss_i = self.criterion(self.model(x), x)
            loss_i.backward()
            losses.append(float(loss_i.item()))
            norma = torch.sqrt(sum((p.grad.detach() ** 2).sum() for p in params if p.grad is not None))
            _n = float(norma.item())
            if not math.isfinite(_n):
                raise RuntimeError("norma del gradiente per esempio non finita")
            scala = min(1.0, C / (_n + 1e-12))
            for acc, p in zip(somma, params):
                if p.grad is not None:
                    acc.add_(p.grad.detach() * scala)
        return somma, losses

    def _record_dp_clipped_sum_vmap(self, batch: torch.Tensor) -> tuple[list[torch.Tensor], list[float]]:
        """Come _record_dp_clipped_sum_loop, con torch.func.vmap."""
        from torch.func import functional_call, grad_and_value, vmap

        nomi = [n for n, p in self.model.named_parameters() if p.requires_grad]
        params = {n: p.detach() for n, p in self.model.named_parameters() if p.requires_grad}
        buffers = {n: b.detach() for n, b in self.model.named_buffers()}
        crit = self.criterion
        modello = self.model

        def _loss(pr: dict[str, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
            x = x.unsqueeze(0)
            return crit(functional_call(modello, (pr, buffers), (x,)), x)

        g, l = vmap(grad_and_value(_loss), in_dims=(None, 0))(params, batch)
        flat = torch.cat([g[n].reshape(batch.shape[0], -1) for n in nomi], dim=1)
        norme = torch.sqrt((flat ** 2).sum(dim=1))
        if not bool(torch.isfinite(norme).all()):
            raise RuntimeError("norma del gradiente per esempio non finita")
        scala = torch.clamp(self.record_dp_max_grad_norm / (norme + 1e-12), max=1.0)
        somma = [(g[n] * scala.view(-1, *([1] * (g[n].dim() - 1)))).sum(dim=0) for n in nomi]
        return somma, [float(v) for v in l.detach()]

    def _train_local_record_dp(self, tensor: torch.Tensor, round_num: int) -> list[float]:
        """DP-SGD per record con campionamento di Poisson. Restituisce la loss
        media di ogni epoca (sugli esempi estratti)."""
        self.model.train()
        n = int(tensor.shape[0])
        B = int(self.batch_size)
        if n < B:
            logger.warning(f"[RECORD-DP] [{self.node_id}] {n} record < batch {B}: nessun passo")
            self.record_dp_last = {"n": n, "q": None, "passi_per_epoca": 0}
            return []
        q = B / n
        per_epoch = n // B
        self.record_dp_last = {"n": n, "q": q, "passi_per_epoca": per_epoch}
        sigma = self.record_dp_noise_multiplier
        C = self.record_dp_max_grad_norm
        gen = torch.Generator().manual_seed(
            self._seed + 104729 * round_num + 7927 + zlib.crc32(self.cluster_id.encode()))
        params = [p for p in self.model.parameters() if p.requires_grad]
        epoch_losses: list[float] = []
        for _epoch in range(self.epochs):
            step_losses: list[float] = []
            for _step in range(per_epoch):
                scelti = (torch.rand(n, generator=gen) < q).nonzero().flatten()
                if scelti.numel() > 0:
                    somma, losses = self._record_dp_clipped_sum_vmap(tensor[scelti].to(self.device))
                    step_losses.extend(losses)
                else:
                    somma = [torch.zeros_like(p) for p in params]
                # il passo rumoroso si fa anche con il lotto vuoto (Poisson)
                self.optimizer.zero_grad(set_to_none=True)
                for idx, (acc, p) in enumerate(zip(somma, params)):
                    g = acc
                    if sigma > 0.0:
                        g = g + torch.normal(mean=0.0, std=sigma * C, size=g.shape, device=g.device)
                    g = g / float(B)                 # denominatore costante = B atteso
                    if self.proximal_mu > 0.0 and self._global_weights is not None:
                        if idx < len(self._global_weights):
                            g = g + self.proximal_mu * (p.detach() - self._global_weights[idx])
                    p.grad = g
                self.optimizer.step()
            if step_losses:
                epoch_losses.append(sum(step_losses) / len(step_losses))
        return epoch_losses

    # ── DP a livello di soggetto ─────────────────────────────────────────────

    def set_subject_weights(self, weights: dict[str, float]) -> None:
        """Pesi dei soggetti presenti anche in altri client (w < 1); gli altri
        soggetti hanno peso 1. Vedi src/ml/subject_dp.py."""
        self._subject_weights = dict(weights or {})

    def _subject_clipped_sum(
        self,
        tensor: torch.Tensor,
        rows_per_subject: list[torch.Tensor],
        weights: list[float],
        seeds: list[int],
    ) -> tuple[list[torch.Tensor], list[float]]:
        """Somma dei gradienti tagliati dei soggetti estratti, SENZA rumore.

        Per ogni soggetto: al piu' max_records_per_subject sue sessioni (scelte
        a caso se sono di piu'), gradiente della loss media su quelle, taglio a
        norma C * w. Il contributo di un soggetto alla somma ha quindi norma
        <= C * w per costruzione, qualunque sia il numero delle sue sessioni:
        e' la condizione perche' l'adiacenza sia a livello di soggetto.
        La scelta dei record usa un generatore per soggetto (`seeds`), cosi'
        il contributo di un soggetto non dipende da quali altri soggetti sono
        nel lotto ne' da quante sessioni hanno.
        Il termine prossimale FedProx non entra qui (non dipende dai dati).
        """
        params = [p for p in self.model.parameters() if p.requires_grad]
        somma = [torch.zeros_like(p) for p in params]
        losses: list[float] = []
        C = float(self.subject_dp["max_grad_norm"])
        m = int(self.subject_dp["max_records_per_subject"])
        for rows, w, sd in zip(rows_per_subject, weights, seeds):
            if rows.numel() > m:
                g = torch.Generator().manual_seed(int(sd))
                rows = rows[torch.randperm(rows.numel(), generator=g)[:m]]
            x = tensor[rows].to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            loss = self.criterion(self.model(x), x)
            loss.backward()
            losses.append(float(loss.item()))
            norma = torch.sqrt(
                sum((p.grad.detach() ** 2).sum() for p in params if p.grad is not None)
            )
            _n = float(norma.item())
            if not math.isfinite(_n):      # 2026-10-16: min(1, nan) = 1 non taglierebbe
                raise RuntimeError("norma del gradiente del soggetto non finita")
            scala = min(1.0, (C * float(w)) / (_n + 1e-12))
            for acc, p in zip(somma, params):
                if p.grad is not None:
                    acc.add_(p.grad.detach() * scala)
        return somma, losses

    def _train_local_subject_dp(
        self,
        tensor: torch.Tensor,
        subjects: list[str],
        round_num: int,
    ) -> list[float]:
        """Training locale con DP-SGD a livello di soggetto (vedi
        src/ml/subject_dp.py). Restituisce la loss media di ogni epoca."""
        self.model.train()
        gruppi: dict[str, list[int]] = {}
        for i, s in enumerate(subjects):
            gruppi.setdefault(s, []).append(i)
        nomi = list(gruppi)
        n_subj = len(nomi)
        rows = [torch.tensor(gruppi[s], dtype=torch.long) for s in nomi]
        pesi = [float(self._subject_weights.get(s, 1.0)) for s in nomi]
        expected = int(self.subject_dp["expected_subjects_per_step"])
        q, per_epoch = sampling_schedule(n_subj, expected)
        # Denominatore COSTANTE (numero atteso di soggetti per passo), anche
        # con q = 1: dividere per N renderebbe la scala del passo, e quindi
        # l'effetto di FedProx e dell'eps di Adam, dipendente dai dati.
        denom = float(max(1, expected))
        self.subject_dp_last = {"n_soggetti": n_subj, "q": q, "passi_per_epoca": per_epoch}
        sigma = float(self.subject_dp["noise_multiplier"])
        C = float(self.subject_dp["max_grad_norm"])
        # Generatore dedicato al campionamento dei soggetti e dei record: non
        # consuma il generatore globale (che governa il rumore, come per la DP
        # per record) e dipende da seed e round come lo shuffle storico.
        # crc32 del client: client diversi hanno campionamenti indipendenti
        # (la somma delle RDP fra client lo presuppone).
        gen = torch.Generator().manual_seed(
            self._seed + 104729 * round_num + 7919 + zlib.crc32(self.cluster_id.encode()))
        logger.info(
            f"[SUBJECT-DP] [{self.node_id}] round {round_num}: {n_subj} soggetti, "
            f"q={q:.4g}, {per_epoch} passi per epoca, sigma={sigma}, C={C}, "
            f"pesi < 1: {sum(1 for w in pesi if w < 1.0)}"
        )
        params = [p for p in self.model.parameters() if p.requires_grad]
        epoch_losses: list[float] = []
        passo = 0
        for _epoch in range(self.epochs):
            step_losses: list[float] = []
            for _step in range(per_epoch):
                passo += 1
                scelti = (torch.rand(n_subj, generator=gen) < q).nonzero().flatten().tolist()
                base = self._seed + 1_000_003 * (round_num * 100_003 + passo)
                somma, losses = self._subject_clipped_sum(
                    tensor, [rows[i] for i in scelti], [pesi[i] for i in scelti],
                    [base + i for i in scelti])
                step_losses.extend(losses)
                # Il passo si fa anche se nessun soggetto e' stato estratto:
                # con Poisson il rilascio rumoroso avviene comunque.
                self.optimizer.zero_grad(set_to_none=True)
                for idx, (acc, p) in enumerate(zip(somma, params)):
                    g = acc
                    if sigma > 0.0:
                        g = g + torch.normal(mean=0.0, std=sigma * C, size=g.shape, device=g.device)
                    g = g / float(denom)
                    if self.proximal_mu > 0.0 and self._global_weights is not None:
                        if idx < len(self._global_weights):
                            g = g + self.proximal_mu * (p.detach() - self._global_weights[idx])
                    p.grad = g
                self.optimizer.step()
            if step_losses:
                epoch_losses.append(sum(step_losses) / len(step_losses))
        return epoch_losses

    def emit_event(self, event: MLPlaneEvent) -> None:
        for listener in self._listeners:
            listener.on_ml_event(event)

    def subscribe(self, listener: MLPlaneListener) -> None:
        """Registra un listener ML Plane (Auditor o IDS)."""
        self._listeners.append(listener)
        logger.debug(f"Listener registrato: {type(listener).__name__}")

    # ── Training API ───────────────────────────────────────────────────────────

    def train_local(
        self,
        sessions: list[dict[str, Any]],
        round_num: int,
    ) -> GradientUpdate:
        """
        Esegue il training locale completo su sessions.
        Restituisce GradientUpdate con pesi, loss, n_samples.
        """
        self._current_round = round_num

        subjects: list[str] | None = None
        if self.subject_dp_enabled:
            tensor, subjects = self._sessions_to_tensor_and_subjects(sessions)
        else:
            tensor = self._sessions_to_tensor(sessions)
        if tensor is None:
            logger.warning(f"[{self.node_id}] Nessuna sessione valida — skip training")
            _skip_update = GradientUpdate(
                node_id=self.node_id,
                cluster_id=self.cluster_id,
                round_num=round_num,
                weights=self.get_weights(),
                gradients=None,
                loss=None,
                n_samples=0,
            )
            # Fix 2026-07-22 (review post-wiring ML Plane, commit 1ca713d):
            # questo early-return NON emetteva alcun evento ML Plane, a
            # differenza del percorso normale sotto (riga ~239) e del percorso
            # "0 batch prodotti" (riga ~215, effective_samples=0 ma comunque
            # emesso). Prima del wiring reale era innocuo — run_fl_rounds()
            # appendeva comunque `update` alla propria lista locale a
            # prescindere da cosa il trainer emettesse internamente. Ora che
            # results[round_num]["raw_updates"]/["updates"] sono popolati
            # leggendo da FLArtifactCollector (src/ml/ml_plane.py), un client
            # che finisce qui (0 sessioni valide questo round) sarebbe stato
            # silenziosamente ASSENTE da entrambe le liste — non solo con
            # pesi stale, ma proprio mancante, cambiando silenziosamente `n`
            # per il Krum score peer-relative di ogni altro client in
            # run_ids() quel round. Emettiamo quindi anche qui, con lo stesso
            # evento/livello del percorso normale.
            self.emit_event(MLPlaneEvent(
                event_type="gradient_upload",
                purdue_level=1,
                payload=_skip_update,
                round_num=round_num,
            ))
            return _skip_update

        epoch_losses: list[float] = []
        if self.subject_dp_enabled:
            # Campionamento di Poisson dei soggetti al posto del DataLoader
            # (vedi _train_local_subject_dp e src/ml/subject_dp.py).
            epoch_losses = self._train_local_subject_dp(tensor, subjects or [], round_num)
        elif self.record_dp_enabled:
            # Poisson sui record dal 2026-10-16 (vedi _train_local_record_dp).
            epoch_losses = self._train_local_record_dp(tensor, round_num)
        else:
            dataset    = TensorDataset(tensor)
            # Generator con seed determinístico per riproducibilità dello shuffle tra run.
            # Il seed dipende da round_num per evitare che tutti i round usino lo stesso shuffle.
            _dl_gen = torch.Generator()
            _dl_gen.manual_seed(self._seed + round_num)
            dataloader = DataLoader(
                dataset, batch_size=self.batch_size, shuffle=True,
                drop_last=True, generator=_dl_gen,
            )

            for epoch in range(self.epochs):
                batch_losses = [self.train_step(batch) for (batch,) in dataloader]
                if not batch_losses:
                    logger.warning(
                        f"[{self.node_id}] Epoch {epoch}: nessun batch "
                        f"(sessioni={len(sessions)}, batch_size={self.batch_size})"
                    )
                    continue
                epoch_losses.append(sum(batch_losses) / len(batch_losses))

        mean_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else None
        loss_str = f"{mean_loss:.6f}" if mean_loss is not None else "N/A"

        # Bug fix: se drop_last=True e il cluster ha < batch_size sessioni,
        # il DataLoader produce 0 batch → epoch_losses è vuota → nessun training.
        # In quel caso n_samples deve essere 0: i pesi sono stale e non devono
        # contribuire al FedAvg (FedAvgAggregator filtra n_samples == 0).
        effective_samples = len(sessions) if epoch_losses else 0
        if effective_samples == 0 and len(sessions) > 0:
            logger.warning(
                f"[{self.node_id}] Round {round_num} — nessun batch prodotto "
                f"(sessioni={len(sessions)} < batch_size={self.batch_size}). "
                f"Pesi stale: n_samples impostato a 0 per escludere da FedAvg."
            )
        logger.info(f"[{self.node_id}] Round {round_num} — loss={loss_str}, n={effective_samples}")

        self.ultima_loss_locale = mean_loss
        update = GradientUpdate(
            node_id=self.node_id,
            cluster_id=self.cluster_id,
            round_num=round_num,
            weights=self.get_weights(),
            gradients=None,
            loss=mean_loss if self.trasmetti_loss else None,
            n_samples=effective_samples,
            metadata={
                "epochs": self.epochs,
                "device": str(self.device),
                "proximal_mu": self.proximal_mu,
            },
        )

        self.emit_event(MLPlaneEvent(
            event_type="gradient_upload",
            purdue_level=1,
            payload=update,
            round_num=round_num,
        ))

        return update

    def apply_global_model(self, aggregated: AggregatedUpdate) -> None:
        """
        Carica il modello globale ricevuto dal server.
        Emette evento weight_download sul ML Plane.
        """
        self.set_weights(aggregated.global_weights)
        self.emit_event(MLPlaneEvent(
            event_type="weight_download",
            purdue_level=1,
            payload=aggregated,
            round_num=aggregated.round_num,
        ))
        logger.info(f"[{self.node_id}] Modello globale applicato — round {aggregated.round_num}")

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _sessions_to_tensor(
        self,
        sessions: list[dict[str, Any]],
    ) -> torch.Tensor | None:
        """
        Converte sessioni EV in tensore float32.
        Scarta sessioni con valori None nelle feature continue.
        """
        rows: list[list[float]] = []
        for s in sessions:
            row: list[float] = []
            valid = True
            for feat in self.CONTINUOUS_FEATURES:
                val = s.get(feat)
                if val is None:
                    valid = False
                    break
                try:
                    x = float(val)
                except (TypeError, ValueError):
                    valid = False
                    break
                if not math.isfinite(x):   # 2026-10-16: come row_valid
                    valid = False
                    break
                row.append(x)
            if valid:
                rows.append(row)

        return torch.tensor(rows, dtype=torch.float32) if rows else None

    def _sessions_to_tensor_and_subjects(
        self,
        sessions: list[dict[str, Any]],
    ) -> tuple[torch.Tensor | None, list[str]]:
        """Come _sessions_to_tensor, ma restituisce anche il soggetto di ogni
        riga valida. I soggetti si calcolano sulla lista COMPLETA (il fallback
        dipende dalla posizione e deve coincidere con cross_client_weights)."""
        ids = subject_ids(sessions, str(self.subject_dp["subject_key"]), self.cluster_id)
        rows: list[list[float]] = []
        subj: list[str] = []
        feats = list(self.CONTINUOUS_FEATURES)
        for s, sid in zip(sessions, ids):
            # Stessa regola di _sessions_to_tensor, condivisa con l'accountant
            # (ml.subject_dp.row_valid): cosi' N coincide nei due punti.
            if row_valid(s, feats):
                rows.append([float(s.get(f)) for f in feats])
                subj.append(sid)
        if not rows:
            return None, []
        return torch.tensor(rows, dtype=torch.float32), subj
