"""Small CPU LSTM with direct seasonal-residual forecasts and a rain-event head."""

from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from mosamiat.collect import CollectionError
from mosamiat.modeling.baseline import FORECAST_METRICS
from mosamiat.modeling.lstm_data import (
    HOURS,
    REGRESSION,
    SCALARS,
    HourlyWindows,
    Preprocessing,
    channels,
)


@dataclass(frozen=True)
class Settings:
    hidden_size: int = 32
    batch_size: int = 64
    max_epochs: int = 40
    patience: int = 6
    learning_rate: float = 0.001
    seed: int = 42


class Network(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, rain_rate: float):
        super().__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, HOURS * (len(REGRESSION) + 1))
        with torch.no_grad():
            self.head.weight.zero_()
            self.head.bias.zero_()
            p = np.clip(rain_rate, 1e-4, 1 - 1e-4)
            self.head.bias.view(HOURS, -1)[:, -1].fill_(float(np.log(p / (1 - p))))

    def forward(self, history):
        _, (hidden, _) = self.encoder(history)
        output = self.head(hidden[-1]).reshape(len(history), HOURS, -1)
        # h1 repeats issue-23, and h24 repeats issue-24 (issue hour is unavailable).
        seasonal = torch.roll(history[:, :, : len(REGRESSION)], shifts=-1, dims=1)
        return torch.cat((output[:, :, :-1] + seasonal, output[:, :, -1:]), dim=-1)


def loss_terms(output, targets):
    valid = torch.isfinite(targets)
    target = torch.nan_to_num(targets)
    squared = (output[..., :-1] - target[..., :-1]).square()
    rain = F.binary_cross_entropy_with_logits(output[..., -1:], target[..., -1:], reduction="none")
    weights = valid.to(output.dtype)
    weights[..., len(SCALARS) : -1] *= 0.5  # Two components describe one wind direction.
    return (torch.cat((squared, rain), dim=-1) * weights).sum(), weights.sum()


@dataclass
class Forecaster:
    model: Network
    prep: Preprocessing
    settings: Settings

    def save(self, path: Path) -> None:
        torch.save(
            {
                "schema_version": 1,
                "preprocessing": asdict(self.prep),
                "settings": asdict(self.settings),
                "state_dict": self.model.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: Path):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint["schema_version"] != 1:
            raise CollectionError("Unsupported LSTM checkpoint")
        prep = Preprocessing(**checkpoint["preprocessing"])
        settings = Settings(**checkpoint["settings"])
        model = Network(2 * len(prep.columns) + 2, settings.hidden_size, prep.rain_rate)
        model.load_state_dict(checkpoint["state_dict"])
        return cls(model.eval(), prep, settings)

    @torch.inference_mode()
    def predict(self, values: pd.DataFrame, issues: pd.DatetimeIndex) -> pd.DataFrame:
        if issues.empty or str(issues.tz) != "UTC" or not issues.is_unique:
            raise CollectionError("Expected distinct UTC issue hours")
        past = self.prep.inputs(channels(values))
        ends = values.index.get_indexer(issues - pd.Timedelta(hours=1))
        if (ends < HOURS - 1).any():
            raise CollectionError("Each forecast needs 24 completed hourly inputs")
        windows = past.unfold(0, HOURS, 1).transpose(1, 2)
        batches = []
        for batch in range(0, len(ends), self.settings.batch_size):
            starts = ends[batch : batch + self.settings.batch_size] - HOURS + 1
            batches.append(self.model(windows[starts]))
        output = torch.cat(batches).reshape(-1, len(REGRESSION) + 1)
        scalar = (
            output[:, : len(SCALARS)].numpy() * self.prep.scales[: len(SCALARS)]
            + self.prep.means[: len(SCALARS)]
        )
        index = pd.MultiIndex.from_product(
            [issues, range(1, HOURS + 1)], names=["reference_at_utc", "horizon_hours"]
        )
        frame = pd.DataFrame(scalar, columns=SCALARS, index=index)
        for name in ("aqi_value", "precipitation_mm", "uv_index", "wind_speed_mps"):
            frame[name] = frame[name].clip(lower=0)
        frame["humidity_pct"] = frame.humidity_pct.clip(0, 100)
        sine, cosine = output[:, -3:-1].numpy().T
        frame["wind_direction_deg"] = (
            np.where(
                np.hypot(sine, cosine) > 1e-8,
                np.rad2deg(np.arctan2(sine, cosine)),
                self.prep.wind_fallback_deg,
            )
            % 360
        )
        frame["rain_probability"] = torch.sigmoid(output[:, -1]).numpy()
        frame = frame[list(FORECAST_METRICS)]
        frame.insert(
            0,
            "target_at_utc",
            index.get_level_values(0) + pd.to_timedelta(index.get_level_values(1), unit="h"),
        )
        return frame


def fit(
    values: pd.DataFrame,
    rain: pd.Series,
    validation: pd.Timestamp,
    test: pd.Timestamp,
    timezone: str,
    settings: Settings = Settings(),
) -> tuple[Forecaster, pd.DataFrame, int]:
    if min(settings.max_epochs, settings.patience, settings.batch_size, settings.hidden_size) < 1:
        raise CollectionError("LSTM settings must be positive")
    if not rain.index.equals(values.index) or not rain.dropna().isin([0, 1]).all():
        raise CollectionError("Rain events must be binary and aligned to the hourly index")
    if validation not in values.index or test not in values.index or validation >= test:
        raise CollectionError("Expected ordered validation and test boundaries inside the dataset")
    # Exclude the entire test period before fitting or choosing the best epoch.
    frame = channels(values.loc[values.index < test])
    split = frame.index.get_loc(validation)
    prep = Preprocessing.fit(frame.iloc[:split], rain.iloc[:split], timezone)
    inputs = prep.inputs(frame)
    targets = torch.from_numpy(
        np.column_stack(
            [
                prep.normalized(frame)[:, : len(REGRESSION)],
                rain.loc[frame.index].to_numpy(dtype=np.float32),
            ]
        )
    )
    train = HourlyWindows(inputs, targets, 0, split)
    valid = HourlyWindows(inputs, targets, split, len(frame))
    for fold in (train, valid):
        if not torch.isfinite(targets[fold.start + 1 : fold.stop + HOURS]).any(dim=0).all():
            raise CollectionError(
                "Every forecast channel needs usable labels in each training/validation fold"
            )
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(settings.seed)
            model = Network(inputs.shape[1], settings.hidden_size, prep.rain_rate)
            history, best_epoch = train_epochs(model, train, valid, settings)
    finally:
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        torch.set_num_threads(threads)
    return Forecaster(model.eval(), prep, settings), history, best_epoch


def train_epochs(model, train, valid, settings):
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate)
    train_loader = DataLoader(train, batch_size=settings.batch_size, shuffle=True)
    valid_loader = DataLoader(valid, batch_size=settings.batch_size, shuffle=False)
    best_loss, best_epoch = np.inf, 0
    history = []
    for epoch in range(1, settings.max_epochs + 1):
        model.train()
        train_sum = train_weight = 0.0
        for x, y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            total, count = loss_terms(model(x), y)
            if count.item() == 0:
                continue
            (total / count).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_sum += total.item()
            train_weight += count.item()
        model.eval()
        val_sum = val_weight = 0.0
        with torch.inference_mode():
            for x, y in valid_loader:
                total, count = loss_terms(model(x), y)
                val_sum += total.item()
                val_weight += count.item()
        loss = val_sum / val_weight
        if not np.isfinite(loss):
            raise CollectionError("Non-finite LSTM validation loss")
        history.append(
            {"epoch": epoch, "training_loss": train_sum / train_weight, "validation_loss": loss}
        )
        if loss < best_loss:
            best_loss, best_epoch, weights = loss, epoch, deepcopy(model.state_dict())
        elif epoch - best_epoch >= settings.patience:
            break
    model.load_state_dict(weights)
    return pd.DataFrame(history).set_index("epoch"), best_epoch
