# src/core/autoencoder.py
"""
Autoencoder — Modello FL per Anomaly Detection
===============================================
Implementa un autoencoder PyTorch per il rilevamento di anomalie
nelle sessioni di ricarica EV.

Ruolo nel framework:
- È il modello ML distribuito tramite FL tra i client (i 3 siti reali
  ACN-Data — Caltech/JPL/Office1 — negli esperimenti di privacy; +2 client
  sintetici solo nello sweep IDS/Byzantine separato — aggiornato 2026-07-24,
  era "12 nodi" dal vecchio schema fittizio a 4 cluster ormai superato)
- Ogni nodo addestra l'autoencoder sui propri dati locali
- FedAvg aggrega i pesi di encoder e decoder
- Le anomalie vengono rilevate tramite errore di ricostruzione (MSE)

Perché un autoencoder:
- I dati ACN-Data non hanno label di anomalia (anomaly_label = None)
- L'autoencoder apprende la distribuzione normale dei dati
- Sessioni anomale producono un MSE alto → rilevate come anomalie
- Si presta naturalmente a FL: pesi aggregabili con FedAvg

Architettura:
    Input  (6 feature numeriche normalizzate)
        ↓
    Encoder:  6 → 16 → 8 → 4
        ↓
    Latent space (4 dimensioni)
        ↓
    Decoder:  4 → 8 → 16 → 6
        ↓
    Output (ricostruzione)
        ↓
    MSE → soglia anomalia

Feature numeriche (da AutoencoderTrainer.CONTINUOUS_FEATURES):
    total_energy_kwh, max_power_kw, kwh_requested,
    minutes_available, hour_of_day, duration_hours

Riferimenti:
    - Hinton & Salakhutdinov, "Reducing Dimensionality with NNs", Science 2006
    - Chalapathy & Chawla, "Deep Learning for Anomaly Detection", 2019
"""

import torch
import torch.nn as nn
from torch import Tensor


# Numero di feature numeriche in input all'autoencoder.
# Corrisponde alle feature dell'ACNDataset dopo preprocessing.
INPUT_DIM = 6


class Encoder(nn.Module):
    """
    Encoder dell'autoencoder: comprime l'input in uno spazio latente.

    Architettura di default: 6 → 16 → 8 → 4
    Usa ReLU nei layer intermedi. Il layer finale non ha attivazione:
    ReLU sull'ultimo layer comprime lo spazio latente in [0,+∞),
    dimezzando la capacità espressiva senza benefici architetturali.
    BatchNorm1d stabilizza il training in FL dove i dati locali
    possono avere distribuzioni molto diverse tra i nodi.

    hidden_dims (aggiunto 2026-08-28, Sprint 10jj — escalation calibrazione
    LiRA positive-control, vedi docs/TestRoadmap_DSN2027.md): permette di
    aumentare la capacità del modello SOLO per gli esperimenti che lo
    richiedono esplicitamente via config['ml']['hidden_dims']. Default
    None → (16, 8), identico byte-per-byte all'architettura storica usata
    per tutti i risultati pubblicati finora (570 parametri). Nessun run
    esistente è affetto a meno che non imposti esplicitamente hidden_dims
    in YAML.
    """

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        latent_dim: int = 4,
        hidden_dims: tuple[int, int] | None = None,
        norm: str = "group",  # default dal 2026-10-16
    ):
        """
        Args:
            input_dim:   dimensione dell'input (default 6 feature)
            latent_dim:  dimensione dello spazio latente (default 4)
            hidden_dims: (h1, h2) dimensioni dei due layer nascosti
                         dell'encoder (default None → (16, 8), storico)
        """
        super().__init__()
        h1, h2 = hidden_dims if hidden_dims is not None else (16, 8)

        # norm (Sprint 10zz+113): scelta del layer di normalizzazione.
        #   "batch" — nn.BatchNorm1d, comportamento storico (default fino al
        #       2026-10-15): serve norm="batch" per riprodurre le celle vecchie.
        #   "group" (default dal 2026-10-16) — nn.GroupNorm(1, dim), equivalente
        #       a LayerNorm su 2-D.
        #       OBBLIGATORIO con DP a livello di record, per due ragioni:
        #       (a) Tecnica: il clipping per-esempio richiede microbatch di
        #           dimensione 1, e BatchNorm1d in training solleva
        #           "Expected more than 1 value per channel" perche' non puo'
        #           stimare una varianza su un solo campione.
        #       (b) Di privacy, ed e' quella che conta: BatchNorm calcola
        #           media e varianza SUL BATCH, quindi l'output di un campione
        #           dipende dagli altri campioni. La sensibilita' del singolo
        #           record non e' piu' limitata dal clipping del suo gradiente
        #           e la garanzia DP per-record decade comunque. Opacus rifiuta
        #           per questo i modelli con BatchNorm. GroupNorm normalizza
        #           entro il singolo campione e non crea dipendenza fra campioni.
        # NOTA: cambiare norm cambia i parametri del modello, quindi una cella
        # "group" NON e' confrontabile con una cella "batch": il confronto
        # record-DP richiede un riferimento no-DP anch'esso in modalita' "group".
        if norm not in ("batch", "group"):
            raise ValueError(f"norm non valido: {norm!r} (atteso 'batch' o 'group')")
        _Norm = (lambda d: nn.BatchNorm1d(d)) if norm == "batch" \
                else (lambda d: nn.GroupNorm(1, d))

        self.network = nn.Sequential(
            # Layer 1: input_dim → h1
            nn.Linear(input_dim, h1),
            _Norm(h1),
            nn.ReLU(),
            # Layer 2: h1 → h2
            nn.Linear(h1, h2),
            _Norm(h2),
            nn.ReLU(),
            # Layer 3: h2 → latent_dim (spazio latente — no ReLU: preserva segno)
            nn.Linear(h2, latent_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass dell'encoder.

        Args:
            x: tensore input di shape (batch_size, input_dim)

        Returns:
            rappresentazione latente di shape (batch_size, latent_dim)
        """
        return self.network(x)


class Decoder(nn.Module):
    """
    Decoder dell'autoencoder: ricostruisce l'input dallo spazio latente.

    Architettura di default: 4 → 8 → 16 → 6
    Usa Sigmoid nell'ultimo layer perché le feature sono normalizzate [0,1].

    hidden_dims: mirror simmetrico dell'Encoder — vedi nota lì (Sprint 10jj,
    2026-08-28). Default None → (8, 16), storico.
    """

    def __init__(
        self,
        latent_dim: int = 4,
        output_dim: int = INPUT_DIM,
        hidden_dims: tuple[int, int] | None = None,
    ):
        """
        Args:
            latent_dim:  dimensione dello spazio latente (default 4)
            output_dim:  dimensione dell'output (deve essere = input_dim)
            hidden_dims: (h1, h2) dimensioni dei due layer nascosti del
                         decoder, nell'ordine di attraversamento
                         latent→output (default None → (8, 16), storico)
        """
        super().__init__()
        h1, h2 = hidden_dims if hidden_dims is not None else (8, 16)
        self.network = nn.Sequential(
            # Layer 1: latent_dim → h1
            nn.Linear(latent_dim, h1),
            nn.ReLU(),
            # Layer 2: h1 → h2
            nn.Linear(h1, h2),
            nn.ReLU(),
            # Layer 3: h2 → output_dim (ricostruzione)
            # Sigmoid: output in [0,1] — coerente con feature normalizzate
            nn.Linear(h2, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass del decoder.

        Args:
            x: rappresentazione latente di shape (batch_size, latent_dim)

        Returns:
            ricostruzione di shape (batch_size, output_dim)
        """
        return self.network(x)


class Autoencoder(nn.Module):
    """
    Autoencoder completo per anomaly detection su sessioni EV.

    Combina Encoder e Decoder. Durante il training FL:
    - Ogni nodo addestra l'autoencoder sui propri dati locali
    - FLAREConnector estrae i pesi con state_dict()
    - FedAvg aggrega i pesi di tutti i nodi
    - Il modello globale viene caricato con load_state_dict()

    La soglia di anomalia viene calibrata localmente su ogni nodo
    usando il 95° percentile degli errori di ricostruzione
    sul validation set locale.

    Uso tipico:
        model = Autoencoder()
        model.fit(train_loader, epochs=5)
        error = model.reconstruction_error(sample_tensor)
        if model.is_anomaly(sample_tensor):
            # segnala anomalia
    """

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        latent_dim: int = 4,
        threshold: float = 0.1,
        hidden_dims: tuple[int, int] | None = None,
        norm: str = "group",  # default dal 2026-10-16
    ):
        """
        Args:
            input_dim:   numero di feature in input (default 6)
            latent_dim:  dimensione spazio latente (default 4)
            threshold:   soglia MSE per anomalia (calibrata con fit())
            hidden_dims: (h1, h2) dei due layer nascosti dell'encoder
                         (default None → (16, 8), architettura storica a
                         570 parametri usata per tutti i risultati
                         pubblicati). Il decoder usa lo stesso hidden_dims
                         in ordine speculare (h2, h1) per simmetria.
                         Aggiunto Sprint 10jj (2026-08-28) per l'escalation
                         di capacità del sanity-check LiRA — vedi
                         docs/TestRoadmap_DSN2027.md.
        """
        super().__init__()
        decoder_hidden_dims = (hidden_dims[1], hidden_dims[0]) if hidden_dims is not None else None
        self.encoder = Encoder(input_dim, latent_dim, hidden_dims=hidden_dims, norm=norm)
        self.decoder = Decoder(latent_dim, input_dim, hidden_dims=decoder_hidden_dims)

        # Soglia di anomalia: MSE > threshold → anomalia
        # Viene aggiornata durante fit() con il 95° percentile
        self.threshold = threshold

        # Loss function: MSE tra input e ricostruzione
        self._criterion = nn.MSELoss(reduction="mean")

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass completo: input → latent → ricostruzione.

        Args:
            x: tensore input di shape (batch_size, input_dim)

        Returns:
            ricostruzione di shape (batch_size, input_dim)
        """
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction

    def reconstruction_error(self, x: Tensor) -> float:
        """
        Calcola l'errore di ricostruzione MSE per un campione.

        Un MSE alto indica che il campione è anomalo rispetto
        alla distribuzione appresa durante il training.
        Il training mode originale viene ripristinato al termine.

        Args:
            x: tensore input di shape (1, input_dim) o (input_dim,)

        Returns:
            MSE tra input e ricostruzione (float)
        """
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                if x.dim() == 1:
                    x = x.unsqueeze(0)
                reconstruction = self.forward(x)
                error = self._criterion(reconstruction, x)
            return error.item()
        finally:
            self.train(was_training)

    def is_anomaly(self, x: Tensor) -> bool:
        """
        Decide se un campione è anomalo confrontando MSE con la soglia.

        Args:
            x: tensore input di shape (1, input_dim) o (input_dim,)

        Returns:
            True se il campione è anomalo, False altrimenti
        """
        return self.reconstruction_error(x) > self.threshold

    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        epochs: int = 5,
        learning_rate: float = 0.01,
    ) -> list[float]:
        """
        Addestra l'autoencoder sui dati locali del nodo.

        Calibra automaticamente la soglia di anomalia al termine
        del training usando il 95° percentile degli errori
        sul training set (assunzione: training set = dati normali).

        Args:
            train_loader:  DataLoader con i dati di training locali
            epochs:        numero di epoche locali (da config/flare.yaml)
            learning_rate: learning rate (da config/flare.yaml)

        Returns:
            lista degli errori medi per epoca (per logging)
        """
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        epoch_losses: list[float] = []

        for epoch in range(epochs):
            batch_losses: list[float] = []

            for batch in train_loader:
                # Unpack tuple: DataLoader da TensorDataset restituisce (tensor,)
                # invece di un tensore raw. Supporta entrambi i casi.
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]
                optimizer.zero_grad()
                reconstruction = self.forward(batch)
                loss = self._criterion(reconstruction, batch)
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.item())

            if not batch_losses:
                continue  # DataLoader vuoto — salta l'epoca senza ZeroDivisionError
            epoch_loss = sum(batch_losses) / len(batch_losses)
            epoch_losses.append(epoch_loss)

        # Calibra soglia: 95° percentile degli errori sul training set
        self.threshold = self._calibrate_threshold(train_loader)

        return epoch_losses

    def _calibrate_threshold(
        self,
        data_loader: torch.utils.data.DataLoader,
        percentile: float = 95.0,
    ) -> float:
        """
        Calibra la soglia di anomalia sul training set.

        Usa il percentile specificato degli errori di ricostruzione.
        Il 95° percentile significa che il 5% dei dati normali
        sarà classificato come anomalo (false positive rate = 5%).

        Args:
            data_loader: DataLoader con i dati di calibrazione
            percentile:  percentile da usare come soglia (default 95)

        Returns:
            soglia calibrata (float)
        """
        self.eval()
        errors: list[float] = []

        with torch.no_grad():
            for batch in data_loader:
                # Unpack tuple: DataLoader da TensorDataset restituisce (tensor,)
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]
                reconstruction = self.forward(batch)
                # Errore per campione (non per batch)
                batch_errors = torch.mean(
                    (reconstruction - batch) ** 2, dim=1
                )
                errors.extend(batch_errors.tolist())

        if not errors:
            # data_loader interamente vuoto (0 batch in ogni epoca) — torch.quantile()
            # su un tensore vuoto solleva RuntimeError ("quantile() input tensor must
            # be non-empty"). Riportato nella review v4 (2026-07-09) come residuo del
            # fix ZeroDivisionError in fit(): quel guard evita il crash nel training
            # loop, ma _calibrate_threshold() veniva chiamata comunque subito dopo con
            # lo stesso DataLoader vuoto. Fallback: mantieni la soglia corrente
            # (invariata rispetto a prima della chiamata) invece di far crashare fit().
            return self.threshold

        # Calcola il percentile degli errori
        errors_tensor = torch.tensor(errors)
        threshold = float(torch.quantile(errors_tensor, percentile / 100.0))
        return threshold

    def get_weights(self) -> dict:
        """
        Restituisce i pesi del modello per la trasmissione a FedAvg.

        Returns:
            state_dict con tutti i pesi del modello
        """
        return self.state_dict()

    def set_weights(self, weights: dict) -> None:
        """
        Carica i pesi aggregati dal server FL.

        Args:
            weights: state_dict ricevuto dall'aggregatore dopo FedAvg
        """
        self.load_state_dict(weights)
