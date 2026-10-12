

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal, Sequence
import re
import warnings

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.preprocessing import StandardScaler


Array = np.ndarray


@dataclass(frozen=True)
class SequenceData:
    """One contiguous sequence which must not be joined across its boundary."""

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
    slopes = np.sum(centered * time_centered[None, :, None], axis=1) / slope_denominator

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


def standardize_train_test(
    X_train: Array,
    X_test: Array,
    scaler: StandardScaler | None = None,
) -> tuple[Array, Array, StandardScaler]:
    """Fit a scaler only on train, preserving 2-D or 3-D array shapes."""

    scaler = scaler or StandardScaler()
    train_shape = X_train.shape
    test_shape = X_test.shape
    train_2d = np.asarray(X_train).reshape(-1, train_shape[-1])
    test_2d = np.asarray(X_test).reshape(-1, test_shape[-1])
    transformed_train = scaler.fit_transform(train_2d).reshape(train_shape)
    transformed_test = scaler.transform(test_2d).reshape(test_shape)
    return (
        transformed_train.astype(np.float32, copy=False),
        transformed_test.astype(np.float32, copy=False),
        scaler,
    )


class _SequenceLoader:
    """Shared compatibility and streaming API."""

    def __init__(
        self,
        *,
        max_samples: int | None = None,
        stride: int = 1,
        extract_stats: bool = False,
    ) -> None:
        self.max_samples = max_samples
        self.stride = stride
        self.extract_stats = extract_stats
        self.feature_names: list[str] = []

    def iter_sequences(self) -> Iterator[SequenceData]:
        raise NotImplementedError

    def get_split_groups(self, sequence_ids: Array) -> Array:
        """Groups which must stay wholly in either train or test."""

        return np.asarray(sequence_ids, dtype=object)

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
        total = 0

        for sequence in self.iter_sequences():
            if window is None:
                X_part = sequence.X
                y_part = sequence.y
            else:
                X_part, y_part = _to_windows(sequence.X, sequence.y, window, stride)
                if extract_stats:
                    X_part = _window_statistics(X_part)

            if limit is not None:
                remaining = limit - total
                if remaining <= 0:
                    break
                X_part = X_part[:remaining]
                y_part = y_part[:remaining]

            if len(X_part):
                X_parts.append(np.asarray(X_part, dtype=np.float32))
                y_parts.append(np.asarray(y_part, dtype=bool))
                id_parts.append(
                    np.full(len(y_part), sequence.sequence_id, dtype=object)
                )
                total += len(y_part)

        if not X_parts:
            raise ValueError("No usable observations were found")

        X = np.concatenate(X_parts)
        y = np.concatenate(y_parts)
        ids = np.concatenate(id_parts)
        if get_simulation_ids:
            return X, y, ids
        return X, y


class TiNADataLoader(_SequenceLoader):
    """Loader for the OD-TiNA daily Parquet partitions.

    ``alarms`` and maintenance codes are operational events, not verified fault
    intervals.  The produced ``y`` is therefore a configurable proxy label.
    """

    _LABEL_COLUMNS = {"alarms", "m_id", "m_subid"}
    _CATEGORICAL_FEATURES = {"FEATURE76", "FEATURE87"}
    _TIME_COLUMNS = {"timestamp", "time", "datetime", "date", "unix"}
    _NORMAL_TOKENS = {"", "0", "false", "nan", "none", "null", "normal"}

    def __init__(
        self,
        data_dir: str | Path,
        *,
        label_source: Literal["alarms", "maintenance", "either"] = "alarms",
        feature_columns: Sequence[str] | None = None,
        max_files: int | None = None,
        rows_per_file: int | None = None,
        max_samples: int | None = None,
        stride: int = 1,
        extract_stats: bool = False,
        allow_full_materialization: bool = False,
        resample_seconds: int | None = 10,
        gap_tolerance_steps: float = 1.5,
    ) -> None:
        super().__init__(
            max_samples=max_samples, stride=stride, extract_stats=extract_stats
        )
        self.data_dir = Path(data_dir)
        self.label_source = label_source
        self.max_files = max_files
        self.rows_per_file = rows_per_file
        self.allow_full_materialization = allow_full_materialization
        if resample_seconds is not None and resample_seconds <= 0:
            raise ValueError("resample_seconds must be positive or None")
        if gap_tolerance_steps <= 1:
            raise ValueError("gap_tolerance_steps must be greater than 1")
        self.resample_seconds = resample_seconds
        self.gap_tolerance_steps = gap_tolerance_steps
        self.files = sorted(self.data_dir.rglob("*.parquet"))
        if max_files is not None:
            self.files = self.files[:max_files]
        if not self.files:
            raise ValueError(f"No Parquet files found under {self.data_dir}")

        schema = pq.read_schema(self.files[0])
        if feature_columns is None:
            self.feature_names = [
                field.name
                for field in schema
                if (
                    pa.types.is_integer(field.type)
                    or pa.types.is_floating(field.type)
                    or pa.types.is_boolean(field.type)
                )
                and field.name not in self._LABEL_COLUMNS
                and field.name not in self._TIME_COLUMNS
                and field.name not in self._CATEGORICAL_FEATURES
            ]
        else:
            self.feature_names = list(feature_columns)
        if not self.feature_names:
            raise ValueError("No numeric TiNA feature columns were found")

    @classmethod
    def _event_mask(cls, values: pd.Series) -> Array:
        normalized = values.astype("string").str.strip().str.lower()
        return (~normalized.isin(cls._NORMAL_TOKENS) & normalized.notna()).to_numpy()

    @staticmethod
    def _timestamps_seconds(values: pd.Index | pd.Series) -> Array:
        if isinstance(values, pd.DatetimeIndex) or pd.api.types.is_datetime64_any_dtype(values):
            return np.asarray(values, dtype="datetime64[ns]").astype(np.int64) / 1e9
        numeric = pd.to_numeric(np.asarray(values), errors="coerce").astype(np.float64)
        finite = numeric[np.isfinite(numeric)]
        if not len(finite):
            raise ValueError("TiNA timestamps contain no finite values")
        magnitude = float(np.nanmedian(np.abs(finite)))
        divisor = (
            1e9
            if magnitude > 1e17
            else 1e6
            if magnitude > 1e14
            else 1e3
            if magnitude > 1e11
            else 1.0
        )
        return numeric / divisor

    def _split_boundaries(self, timestamps: Array) -> Array:
        if len(timestamps) < 2:
            return np.array([0, len(timestamps)], dtype=int)
        deltas = np.diff(timestamps)
        positive = deltas[deltas > 0]
        if not len(positive):
            return np.arange(len(timestamps) + 1, dtype=int)
        nominal_step = float(np.median(positive))
        breaks = np.flatnonzero(
            (deltas <= 0) | (deltas > self.gap_tolerance_steps * nominal_step)
        ) + 1
        return np.r_[0, breaks, len(timestamps)]

    def _aggregate_segment(
        self,
        X_frame: pd.DataFrame,
        y: Array,
        timestamps: Array,
    ) -> tuple[Array, Array, Array]:
        if self.resample_seconds is None:
            X = X_frame.ffill().fillna(0.0).to_numpy(dtype=np.float32)
            return X, np.asarray(y, dtype=bool), timestamps

        bucket = np.floor(timestamps / self.resample_seconds).astype(np.int64)
        working = X_frame.copy()
        working["__bucket"] = bucket
        working["__label"] = np.asarray(y, dtype=bool)
        grouped = working.groupby("__bucket", sort=True)
        # Mean aggregation is causal when the output timestamp is the end of
        # the completed bin. Labels use any(), so short alarms are not averaged away.
        X_aggregated = grouped[self.feature_names].mean().ffill().fillna(0.0)
        y_aggregated = grouped["__label"].any().to_numpy(dtype=bool)
        timestamps_aggregated = (
            X_aggregated.index.to_numpy(dtype=np.float64) + 1
        ) * self.resample_seconds
        return (
            X_aggregated.to_numpy(dtype=np.float32),
            y_aggregated,
            timestamps_aggregated,
        )

    def _read_sequences(self, path: Path) -> Iterator[SequenceData]:
        schema_names = set(pq.read_schema(path).names)
        requested = list(self.feature_names)
        if self.label_source in {"alarms", "either"}:
            requested.append("alarms")
        if self.label_source in {"maintenance", "either"}:
            requested.append("m_id")
        time_column = next((c for c in self._TIME_COLUMNS if c in schema_names), None)
        if time_column:
            requested.append(time_column)

        missing = [column for column in self.feature_names if column not in schema_names]
        if missing:
            raise ValueError(f"{path} misses TiNA features: {missing}")
        frame = pd.read_parquet(path, columns=list(dict.fromkeys(requested)))
        if self.rows_per_file is not None:
            frame = frame.iloc[: self.rows_per_file]

        X_frame = frame[self.feature_names].replace([np.inf, -np.inf], np.nan)
        y = np.zeros(len(frame), dtype=bool)
        if self.label_source in {"alarms", "either"}:
            y |= self._event_mask(frame["alarms"])
        if self.label_source in {"maintenance", "either"}:
            y |= self._event_mask(frame["m_id"])

        timestamp_values = (
            frame[time_column].to_numpy()
            if time_column in frame.columns
            else frame.index.to_numpy()
        )
        timestamps = self._timestamps_seconds(timestamp_values)
        order = np.argsort(timestamps, kind="stable")
        timestamps = timestamps[order]
        X_frame = X_frame.iloc[order].reset_index(drop=True)
        y = y[order]

        relative_path = str(path.relative_to(self.data_dir))
        boundaries = self._split_boundaries(timestamps)
        for segment_index, (start, end) in enumerate(
            zip(boundaries[:-1], boundaries[1:])
        ):
            if end <= start:
                continue
            X_part, y_part, timestamps_part = self._aggregate_segment(
                X_frame.iloc[start:end].reset_index(drop=True),
                y[start:end],
                timestamps[start:end],
            )
            if not len(X_part):
                continue
            yield SequenceData(
                X=X_part,
                y=y_part,
                sequence_id=f"{relative_path}#segment={segment_index}",
                timestamps=timestamps_part,
                metadata={
                    "path": str(path),
                    "label_source": self.label_source,
                    "resample_seconds": self.resample_seconds,
                    "segment_index": segment_index,
                },
            )

    def iter_sequences(self) -> Iterator[SequenceData]:
        for path in self.files:
            yield from self._read_sequences(path)

    def get_full_data(self, *args: Any, **kwargs: Any):
        requested_limit = args[0] if args else kwargs.get("max_samples", None)
        effective_limit = self.max_samples if requested_limit is None else requested_limit
        if effective_limit is None and not self.allow_full_materialization:
            raise ValueError(
                "TiNA has more than 38 million rows. Set max_samples, use "
                "iter_window_batches(), or explicitly pass "
                "allow_full_materialization=True."
            )
        return super().get_full_data(*args, **kwargs)


class Petrobras3WDataLoader(_SequenceLoader):
    """Loader for Petrobras 3W Dataset 2.0.0 episode Parquets."""

    _EXCLUDED_COLUMNS = {"timestamp", "class", "state"}

    def __init__(
        self,
        data_dir: str | Path,
        *,
        sources: Sequence[Literal["real", "simulated", "drawn", "unknown"]] = (
            "real",
        ),
        event_types: Sequence[int] | None = None,
        feature_columns: Sequence[str] | None = None,
        feature_policy: Literal["intersection", "union"] = "intersection",
        drop_unlabeled: bool = True,
        max_files: int | None = None,
        max_samples: int | None = None,
        stride: int = 1,
        extract_stats: bool = False,
    ) -> None:
        super().__init__(
            max_samples=max_samples, stride=stride, extract_stats=extract_stats
        )
        self.data_dir = Path(data_dir)
        self.sources = set(sources)
        self.event_types = set(event_types) if event_types is not None else None
        self.drop_unlabeled = drop_unlabeled
        self.feature_policy = feature_policy

        candidates = sorted(self.data_dir.rglob("*.parquet"))
        self.files = [path for path in candidates if self._accept_file(path)]
        if max_files is not None:
            self.files = self.files[:max_files]
        if not self.files:
            raise ValueError(f"No matching 3W Parquet files found under {self.data_dir}")

        if feature_columns is None:
            feature_sets = [self._numeric_schema_columns(path) for path in self.files]
            if feature_policy == "intersection":
                selected = set.intersection(*feature_sets)
            elif feature_policy == "union":
                selected = set.union(*feature_sets)
            else:
                raise ValueError("feature_policy must be 'intersection' or 'union'")
            self.feature_names = sorted(selected - self._EXCLUDED_COLUMNS)
        else:
            self.feature_names = list(feature_columns)
        if not self.feature_names:
            raise ValueError("No numeric 3W feature columns were found")

    @staticmethod
    def _source(path: Path) -> str:
        stem = path.stem.upper()
        if stem.startswith("WELL"):
            return "real"
        if stem.startswith("SIMULATED"):
            return "simulated"
        if stem.startswith("DRAWN"):
            return "drawn"
        return "unknown"

    @staticmethod
    def _event_type(path: Path) -> int | None:
        for parent in path.parents:
            try:
                value = int(parent.name)
            except ValueError:
                continue
            if 0 <= value <= 9:
                return value
        return None

    def _accept_file(self, path: Path) -> bool:
        if self._source(path) not in self.sources:
            return False
        event_type = self._event_type(path)
        return self.event_types is None or event_type in self.event_types

    @staticmethod
    def _numeric_schema_columns(path: Path) -> set[str]:
        return {
            field.name
            for field in pq.read_schema(path)
            if pa.types.is_integer(field.type)
            or pa.types.is_floating(field.type)
            or pa.types.is_boolean(field.type)
        }

    def _read_sequence(self, path: Path) -> SequenceData:
        schema_names = set(pq.read_schema(path).names)
        if "class" not in schema_names:
            raise ValueError(f"3W label column 'class' is absent in {path}")
        available_features = [c for c in self.feature_names if c in schema_names]
        requested = available_features + ["class"]
        if "timestamp" in schema_names:
            requested.append("timestamp")
        frame = pd.read_parquet(path, columns=requested)

        if self.drop_unlabeled:
            frame = frame.loc[frame["class"].notna()].copy()
        labels = frame["class"].fillna(0)
        y = labels.ne(0).to_numpy(dtype=bool)

        X_frame = frame.reindex(columns=self.feature_names)
        X_frame = X_frame.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
        X = X_frame.to_numpy(dtype=np.float32)
        timestamps = (
            frame["timestamp"].to_numpy()
            if "timestamp" in frame.columns
            else frame.index.to_numpy()
        )
        sequence_id = str(path.relative_to(self.data_dir))
        return SequenceData(
            X=X,
            y=y,
            sequence_id=sequence_id,
            timestamps=timestamps,
            metadata={
                "path": str(path),
                "source": self._source(path),
                "event_type": self._event_type(path),
            },
        )

    def iter_sequences(self) -> Iterator[SequenceData]:
        for path in self.files:
            sequence = self._read_sequence(path)
            if len(sequence.X):
                yield sequence

    def get_split_groups(self, sequence_ids: Array) -> Array:
        """Keep all real episodes of one well on the same side of a split."""

        result: list[str] = []
        for sequence_id in sequence_ids:
            match = re.search(r"WELL[-_]?\d+", str(sequence_id), flags=re.IGNORECASE)
            result.append(match.group(0).upper() if match else str(sequence_id))
        return np.asarray(result, dtype=object)
