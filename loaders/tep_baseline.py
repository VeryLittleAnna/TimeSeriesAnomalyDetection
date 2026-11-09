from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator
import numpy as np
import pandas as pd

Array = np.ndarray

@dataclass(frozen=True)
class SequenceData:
    X: Array
    y: Array
    sequence_id: str
    timestamps: Array | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

def _to_windows(
    X: Array,
    y: Array,
    window_size: int,
    stride: int,
) -> tuple[Array, Array]:
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")
    if len(X) < window_size:
        return (
            np.empty((0, window_size, X.shape[1]), dtype=np.float32),
            np.empty(0, dtype=bool),
        )

    windows = np.lib.stride_tricks.sliding_window_view(
        X, window_shape=window_size, axis=0
    ).transpose(0, 2, 1)
    windows = windows[::stride]
    # Causal convention used by the coursework: label of the window endpoint.
    labels = y[window_size - 1 :: stride]
    return np.asarray(windows, dtype=np.float32), np.asarray(labels, dtype=bool)

def _window_event_mask(y: Array, window_size: int, stride: int) -> Array:
    """Whether any event occurs inside each causal window."""

    if len(y) < window_size:
        return np.empty(0, dtype=bool)
    windows = np.lib.stride_tricks.sliding_window_view(
        np.asarray(y, dtype=bool), window_shape=window_size
    )
    return np.asarray(windows[::stride].any(axis=1), dtype=bool)

def _window_statistics(windows: Array) -> Array:
    """The same 23 feature groups as CSVDataLoader._extract_window_stats_vectorized."""

    if len(windows) == 0:
        return np.empty((0, windows.shape[-1] * 23), dtype=np.float32)

    means = np.mean(windows, axis=1)
    stds = np.std(windows, axis=1)
    mins = np.min(windows, axis=1)
    maxs = np.max(windows, axis=1)
    medians = np.median(windows, axis=1)
    q25 = np.percentile(windows, 25, axis=1)
    q50 = np.percentile(windows, 50, axis=1)
    q75 = np.percentile(windows, 75, axis=1)

    centered = windows - means[:, None, :]
    std_safe = np.where(stds == 0, 1.0, stds)
    skews = np.mean((centered / std_safe[:, None, :]) ** 3, axis=1)
    kurtoses = np.mean((centered / std_safe[:, None, :]) ** 4, axis=1) - 3

    time = np.arange(windows.shape[1], dtype=np.float64)
    time_centered = time - time.mean()
    slope_denominator = np.sum(time_centered**2)
    slopes = (np.sum(centered * time_centered[None, :, None], axis=1) / slope_denominator
              if slope_denominator else np.zeros_like(means))

    energy = np.sum(windows**2, axis=1)
    autocorr_den = np.sum(centered**2, axis=1)
    autocorr_num = np.sum(centered[:, :-1] * centered[:, 1:], axis=1)
    autocorr = np.divide(
        autocorr_num,
        autocorr_den,
        out=np.zeros_like(autocorr_num),
        where=autocorr_den != 0,
    )
    crossings = np.sum(centered[:, :-1] * centered[:, 1:] < 0, axis=1)
    first_values = windows[:, 0]
    last_values = windows[:, -1]

    fft = np.abs(np.fft.fft(windows, axis=1))
    harmonics = []
    for harmonic in range(5):
        if harmonic < fft.shape[1]:
            harmonics.append(fft[:, harmonic])
        else:
            harmonics.append(np.zeros_like(means))

    coefficient_of_variation = np.divide(
        stds, means, out=np.zeros_like(stds), where=means != 0
    )
    rms = np.sqrt(np.mean(windows**2, axis=1))

    result = np.concatenate(
        [
            means,
            mins,
            stds,
            maxs,
            medians,
            skews,
            kurtoses,
            q25,
            q50,
            q75,
            slopes,
            energy,
            autocorr,
            crossings,
            first_values,
            last_values,
            *harmonics,
            coefficient_of_variation,
            rms,
        ],
        axis=1,
    )
    return np.nan_to_num(result).astype(np.float32, copy=False)

FEATURES = [f'xmeas_{i}' for i in range(1, 23)] + [f'xmv_{i}' for i in range(1, 12)]


class TEP_baseline:
    def __init__(self, data_dir, anomaly_number, *, run_start=1, run_stop=500,
                 stride=1, extract_stats=False):
        if not isinstance(stride, int) or isinstance(stride, bool) or stride <= 0:
            raise ValueError('stride must be a positive integer')
        self.stride = stride
        self.extract_stats = extract_stats
        self.max_samples = None
        if not 1 <= int(anomaly_number) <= 15:
            raise ValueError('anomaly_number must be 1..15')
        if not 1 <= run_start <= run_stop <= 500:
            raise ValueError('Expected 1 <= run_start <= run_stop <= 500')
        self.data_dir = Path(data_dir)
        self.anomaly_number = int(anomaly_number)
        self.run_start, self.run_stop = run_start, run_stop
        self.feature_names = FEATURES.copy()

    def iter_sequences(self):
        for run in range(self.run_start, self.run_stop + 1):
            for fault in (0, self.anomaly_number):
                path = self.data_dir / f'fault_{fault:02d}' / f'run_{run:03d}.parquet'
                frame = pd.read_parquet(path)
                if list(frame.columns) != ['sample', *FEATURES, 'y']:
                    raise ValueError(f'{path}: unexpected schema')
                if not np.array_equal(frame['sample'], np.arange(1, 501)):
                    raise ValueError(f'{path}: expected 500 ordered samples')
                expected = (frame['sample'].to_numpy() > 20) & (fault != 0)
                if not np.array_equal(frame['y'].to_numpy(), expected):
                    raise ValueError(f'{path}: invalid fault onset labels')
                x = frame[FEATURES].to_numpy(dtype=np.float32)
                if not np.isfinite(x).all():
                    raise ValueError(f'{path}: nonfinite values')
                yield SequenceData(x, expected, f'rieth_training:{run}:{fault}',
                                   metadata={'run': run, 'fault': fault})

    def get_split_groups(self, sequence_ids):
        # Same run across normal/fault scenarios stays on the same split side.
        return np.asarray([':'.join(str(s).split(':')[:2]) for s in sequence_ids], dtype=object)

    def iter_window_batches(
        self,
        window: int,
        *,
        stride: int | None = None,
        extract_stats: bool | None = None,
    ) -> Iterator[tuple[Array, Array, Array]]:
        """Yield one already-windowed batch per source sequence."""

        stride = self.stride if stride is None else stride
        extract_stats = self.extract_stats if extract_stats is None else extract_stats
        for sequence in self.iter_sequences():
            X, y = _to_windows(sequence.X, sequence.y, window, stride)
            if extract_stats:
                X = _window_statistics(X)
            elif window == 1:
                X = X[:, 0, :]
            ids = np.full(len(y), sequence.sequence_id, dtype=object)
            yield X, y, ids

    def get_full_data(
        self,
        max_samples: int | None = None,
        window: int | None = None,
        get_simulation_ids: bool = False,
        stride: int | None = None,
        extract_stats: bool | None = None,
    ) -> tuple[Array, Array] | tuple[Array, Array, Array]:
        """Materialize data using the coursework loader's return contract."""

        limit = self.max_samples if max_samples is None else max_samples
        stride = self.stride if stride is None else stride
        extract_stats = self.extract_stats if extract_stats is None else extract_stats

        X_parts: list[Array] = []
        y_parts: list[Array] = []
        id_parts: list[Array] = []
        window_event_parts: list[Array] = []
        total = 0

        for sequence in self.iter_sequences():
            if window is None:
                X_part = sequence.X
                y_part = sequence.y
                window_event_part = np.asarray(sequence.y, dtype=bool)
            else:
                X_part, y_part = _to_windows(sequence.X, sequence.y, window, stride)
                window_event_part = _window_event_mask(sequence.y, window, stride)
                if extract_stats:
                    X_part = _window_statistics(X_part)
                elif window == 1:
                    # Keep the legacy runner's numeric window parameter while
                    # presenting point-wise samples to sklearn estimators.
                    X_part = X_part[:, 0, :]

            if limit is not None:
                remaining = limit - total
                if remaining <= 0:
                    break
                X_part = X_part[:remaining]
                y_part = y_part[:remaining]
                window_event_part = window_event_part[:remaining]

            if len(X_part):
                X_parts.append(np.asarray(X_part, dtype=np.float32))
                y_parts.append(np.asarray(y_part, dtype=bool))
                window_event_parts.append(
                    np.asarray(window_event_part, dtype=bool)
                )
                id_parts.append(
                    np.full(len(y_part), sequence.sequence_id, dtype=object)
                )
                total += len(y_part)

        if not X_parts:
            raise ValueError("No usable observations were found")

        X = np.concatenate(X_parts)
        y = np.concatenate(y_parts)
        self.window_has_event_ = np.concatenate(window_event_parts)
        ids = np.concatenate(id_parts)
        if get_simulation_ids:
            return X, y, ids
        return X, y
