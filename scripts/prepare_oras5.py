#!/usr/bin/env python3
"""Download and convert the ICDC ORAS5 1-degree control member for TSC-Fusion.

The ICDC r1x1 archive has a documented interpolation-mask defect. This script
always applies the corrected ICDC masks before writing a standard
TIME/LEVEL/LATITUDE/LONGITUDE NetCDF file; there is no unsafe opt-out.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tarfile
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import netCDF4
import numpy as np


ICDC_BASE_URL = (
    "https://icdc.cen.uni-hamburg.de/thredds/fileServer/"
    "ftpthredds/EASYInit/oras5/r1x1"
)
ORAS5_DOI = "10.24381/cds.67e8eeb7"
MASK_CORRECTION_URL = (
    f"{ICDC_BASE_URL}/Correction_to_ORAS5_r1x1_files.pdf"
)
DEFAULT_DEPTHS_M = (
    0.0, 5.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 125.0,
    150.0, 200.0, 250.0, 300.0, 400.0, 500.0, 600.0, 700.0,
    800.0, 900.0, 1000.0,
)
ICDC_FIRST_YEAR = 1979
ICDC_LAST_YEAR = 2018


@dataclass(frozen=True)
class VariableSpec:
    source_name: str
    output_name: str
    dimensionality: int
    mask_file: str
    mask_variable: str
    units: str
    long_name: str


VARIABLE_SPECS = (
    VariableSpec(
        "votemper", "TEMP", 3, "tmask_r1x1.nc", "tmask", "degC",
        "Sea water potential temperature",
    ),
    VariableSpec(
        "vosaline", "SALT", 3, "tmask_r1x1.nc", "tmask", "PSU",
        "Sea water practical salinity",
    ),
    VariableSpec(
        "vozocrte", "UVEL", 3, "tmask_r1x1.nc", "tmask", "m s-1",
        "Eastward rotated ocean velocity",
    ),
    VariableSpec(
        "vomecrtn", "VVEL", 3, "tmask_r1x1.nc", "tmask", "m s-1",
        "Northward rotated ocean velocity",
    ),
    VariableSpec(
        "sossheig", "SSHA", 2, "tmask2D_r1x1.nc", "tmask", "m",
        "Sea surface height",
    ),
    VariableSpec(
        "somxl010", "MLD", 2, "tmask2D_r1x1.nc", "tmask", "m",
        "Mixed layer depth using the 0.01 density criterion",
    ),
    VariableSpec(
        "sozotaux", "TAUX", 2, "umask2D_r1x1.nc", "umask", "N m-2",
        "Zonal wind stress",
    ),
    VariableSpec(
        "sometauy", "TAUY", 2, "vmask2D_r1x1.nc", "vmask", "N m-2",
        "Meridional wind stress",
    ),
    VariableSpec(
        "sohefldo", "QNET", 2, "tmask2D_r1x1.nc", "tmask", "W m-2",
        "Net downward heat flux into the ocean",
    ),
    VariableSpec(
        "sowaflup", "WFLUX", 2, "tmask2D_r1x1.nc", "tmask", "kg m-2 s-1",
        "Net upward water flux from the ocean",
    ),
)


def parse_depths(raw: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in raw.split(',') if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("--depths 至少需要一个深度")
    if any(value < 0 for value in values):
        raise argparse.ArgumentTypeError("--depths 不能包含负值")
    if any(right <= left for left, right in zip(values, values[1:])):
        raise argparse.ArgumentTypeError("--depths 必须严格递增且不能重复")
    return values


def select_specs(names: Iterable[str]) -> tuple[VariableSpec, ...]:
    requested = list(names)
    if len(requested) != len(set(requested)):
        raise ValueError("--variables 不能包含重复变量")
    by_output = {spec.output_name: spec for spec in VARIABLE_SPECS}
    unknown = [name for name in requested if name not in by_output]
    if unknown:
        raise ValueError(
            f"未知输出变量 {unknown}；可选值为 {sorted(by_output)}"
        )
    return tuple(by_output[name] for name in requested)


def archive_url(spec: VariableSpec, year: int) -> str:
    filename = f"{spec.source_name}_ORAS5_1m_{year}_r1x1.tar.gz"
    return f"{ICDC_BASE_URL}/{spec.source_name}/opa0/{filename}"


def mask_url(filename: str) -> str:
    return f"{ICDC_BASE_URL}/LSM_r1x1/{filename}"


def depth_brackets(
    source_depths: np.ndarray,
    target_depths: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return lower/upper source indices and linear weights for each target."""
    source = np.asarray(source_depths, dtype=np.float64)
    target = np.asarray(target_depths, dtype=np.float64)
    if source.ndim != 1 or source.size == 0 or np.any(np.diff(source) <= 0):
        raise ValueError("ORAS5 source depth coordinate must be non-empty and increasing")
    if target.ndim != 1 or target.size == 0:
        raise ValueError("Target depth coordinate must be a non-empty vector")

    upper = np.searchsorted(source, target, side="left")
    upper = np.clip(upper, 0, source.size - 1)
    lower = np.maximum(upper - 1, 0)
    below_surface = target <= source[0]
    above_bottom = target >= source[-1]
    lower[below_surface] = upper[below_surface] = 0
    lower[above_bottom] = upper[above_bottom] = source.size - 1

    denominator = source[upper] - source[lower]
    weight = np.zeros_like(target, dtype=np.float64)
    distinct = denominator > 0
    weight[distinct] = (
        (target[distinct] - source[lower[distinct]]) / denominator[distinct]
    )
    return lower.astype(np.int64), upper.astype(np.int64), weight


def interpolate_masked_depths(
    values: np.ndarray,
    valid_mask: np.ndarray,
    source_depths: np.ndarray,
    target_depths: np.ndarray,
) -> np.ndarray:
    """Linearly interpolate a 3-D field while preserving the water-column mask."""
    data = np.asarray(values, dtype=np.float32)
    mask = np.asarray(valid_mask, dtype=bool)
    if data.ndim != 3 or mask.shape != data.shape:
        raise ValueError(
            f"3-D ORAS5 data/mask shape mismatch: data={data.shape}, mask={mask.shape}"
        )
    lower, upper, weight = depth_brackets(source_depths, target_depths)
    output = np.full(
        (len(target_depths), data.shape[1], data.shape[2]),
        np.nan,
        dtype=np.float32,
    )
    for index, (low, high, alpha) in enumerate(zip(lower, upper, weight)):
        if low == high:
            layer = data[low]
            valid = mask[low] & np.isfinite(layer)
        else:
            low_values = data[low]
            high_values = data[high]
            layer = (1.0 - alpha) * low_values + alpha * high_values
            valid = (
                mask[low] & mask[high]
                & np.isfinite(low_values) & np.isfinite(high_values)
            )
        output[index, valid] = layer[valid]
    return output


def apply_surface_mask(values: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    data = np.asarray(values, dtype=np.float32)
    mask = np.asarray(valid_mask, dtype=bool)
    if data.ndim != 2 or mask.shape != data.shape:
        raise ValueError(
            f"2-D ORAS5 data/mask shape mismatch: data={data.shape}, mask={mask.shape}"
        )
    return np.where(mask & np.isfinite(data), data, np.nan).astype(np.float32)


def _request(url: str, *, method: str = "GET", range_start: int | None = None):
    headers = {"User-Agent": "TSC-Fusion-ORAS5-preparer/1.0"}
    if range_start:
        headers["Range"] = f"bytes={range_start}-"
    return urllib.request.Request(url, headers=headers, method=method)


def remote_size(url: str) -> int | None:
    try:
        with urllib.request.urlopen(_request(url, method="HEAD"), timeout=60) as response:
            raw = response.headers.get("Content-Length")
            return int(raw) if raw is not None else None
    except (OSError, urllib.error.URLError, ValueError):
        return None


def download_file(url: str, target: Path, retries: int = 8) -> Path:
    """Download with a resumable .part file and atomic final rename."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.stat().st_size > 0:
        return target
    partial = target.with_suffix(target.suffix + ".part")

    for attempt in range(1, retries + 1):
        existing = partial.stat().st_size if partial.exists() else 0
        try:
            request = _request(url, range_start=existing)
            with urllib.request.urlopen(request, timeout=120) as response:
                append = existing > 0 and response.status == 206
                mode = "ab" if append else "wb"
                if not append:
                    existing = 0
                remaining = response.headers.get("Content-Length")
                total = existing + int(remaining) if remaining else None
                downloaded = existing
                next_report = downloaded + 64 * 1024 * 1024
                with partial.open(mode) as handle:
                    while True:
                        block = response.read(4 * 1024 * 1024)
                        if not block:
                            break
                        handle.write(block)
                        downloaded += len(block)
                        if downloaded >= next_report:
                            if total:
                                print(
                                    f"    {target.name}: {downloaded / 2**20:.1f}/"
                                    f"{total / 2**20:.1f} MiB"
                                )
                            next_report += 64 * 1024 * 1024
                if total is not None and downloaded != total:
                    raise IOError(
                        f"download size mismatch: expected={total}, actual={downloaded}"
                    )
            os.replace(partial, target)
            return target
        except (OSError, urllib.error.URLError) as exc:
            if attempt == retries:
                raise RuntimeError(f"下载失败: {url}") from exc
            print(f"    下载中断，第 {attempt}/{retries} 次重试: {exc}")
            time.sleep(min(2 ** attempt, 8))
    raise AssertionError("unreachable")


def load_corrected_masks(work_dir: Path, specs: tuple[VariableSpec, ...]):
    required = {(spec.mask_file, spec.mask_variable) for spec in specs}
    masks: dict[str, np.ndarray] = {}
    latitudes = longitudes = None
    for filename, variable in sorted(required):
        path = download_file(mask_url(filename), work_dir / "masks" / filename)
        with netCDF4.Dataset(path) as dataset:
            raw = dataset.variables[variable][:]
            valid = np.squeeze(np.ma.filled(raw, 0) > 0)
            if valid.ndim not in (2, 3):
                raise ValueError(f"Unexpected corrected mask shape in {path}: {valid.shape}")
            masks[filename] = np.asarray(valid, dtype=bool)
            current_latitudes = np.asarray(dataset.variables["lat"][:], dtype=np.float32)
            current_longitudes = np.asarray(dataset.variables["lon"][:], dtype=np.float32)
            if latitudes is None:
                latitudes, longitudes = current_latitudes, current_longitudes
            elif not (
                np.array_equal(latitudes, current_latitudes)
                and np.array_equal(longitudes, current_longitudes)
            ):
                raise ValueError("Corrected ORAS5 masks do not share one regular grid")
    return masks, latitudes, longitudes


def _create_output(
    path: Path,
    specs: tuple[VariableSpec, ...],
    years: range,
    depths: tuple[float, ...],
    latitudes: np.ndarray,
    longitudes: np.ndarray,
) -> netCDF4.Dataset:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset = netCDF4.Dataset(path, mode="w", format="NETCDF4")
    month_count = len(years) * 12
    dataset.createDimension("TIME", month_count)
    dataset.createDimension("LEVEL", len(depths))
    dataset.createDimension("LATITUDE", len(latitudes))
    dataset.createDimension("LONGITUDE", len(longitudes))

    time_variable = dataset.createVariable("TIME", "i4", ("TIME",))
    level_variable = dataset.createVariable("LEVEL", "f4", ("LEVEL",))
    latitude_variable = dataset.createVariable("LATITUDE", "f4", ("LATITUDE",))
    longitude_variable = dataset.createVariable("LONGITUDE", "f4", ("LONGITUDE",))
    time_variable[:] = np.asarray(
        [year * 100 + month for year in years for month in range(1, 13)],
        dtype=np.int32,
    )
    level_variable[:] = np.asarray(depths, dtype=np.float32)
    latitude_variable[:] = latitudes
    longitude_variable[:] = longitudes
    time_variable.long_name = "calendar month encoded as YYYYMM"
    level_variable.units = "m"
    level_variable.positive = "down"
    latitude_variable.units = "degrees_north"
    longitude_variable.units = "degrees_east"

    fill_value = np.float32(9.96921e36)
    for spec in specs:
        if spec.dimensionality == 3:
            dimensions = ("TIME", "LEVEL", "LATITUDE", "LONGITUDE")
            chunks = (1, len(depths), min(45, len(latitudes)), min(90, len(longitudes)))
        else:
            dimensions = ("TIME", "LATITUDE", "LONGITUDE")
            chunks = (1, min(45, len(latitudes)), min(90, len(longitudes)))
        variable = dataset.createVariable(
            spec.output_name,
            "f4",
            dimensions,
            zlib=True,
            complevel=4,
            shuffle=True,
            chunksizes=chunks,
            fill_value=fill_value,
        )
        variable.units = spec.units
        variable.long_name = spec.long_name
        variable.source_variable = spec.source_name
        variable.corrected_mask = spec.mask_file

    dataset.title = "ORAS5 monthly control member prepared for TSC-Fusion"
    dataset.source = "ECMWF ORAS5 via the ICDC r1x1 opa0 archive"
    dataset.source_doi = ORAS5_DOI
    dataset.license = "CC-BY-4.0"
    dataset.mask_correction = MASK_CORRECTION_URL
    dataset.history = (
        f"Prepared {datetime.now(timezone.utc).isoformat()} by scripts/prepare_oras5.py"
    )
    dataset.sync()
    return dataset


def _config_payload(
    specs: tuple[VariableSpec, ...],
    years: range,
    depths: tuple[float, ...],
) -> dict:
    return {
        "source": ICDC_BASE_URL,
        "years": [years.start, years.stop - 1],
        "depths_m": list(depths),
        "variables": [asdict(spec) for spec in specs],
    }


def _fingerprint(payload: dict) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _load_or_initialize_state(
    state_path: Path,
    partial_path: Path,
    payload: dict,
) -> dict:
    fingerprint = _fingerprint(payload)
    if state_path.exists() or partial_path.exists():
        if not (state_path.exists() and partial_path.exists()):
            raise RuntimeError(
                "发现不完整的 ORAS5 输出但缺少匹配状态；请检查后使用 --overwrite"
            )
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("fingerprint") != fingerprint:
            raise RuntimeError("现有 ORAS5 partial 输出与本次参数不一致，请使用 --overwrite")
        return state
    state = {"fingerprint": fingerprint, "completed": []}
    _write_json_atomic(state_path, state)
    return state


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _masked_array(values: np.ndarray) -> np.ma.MaskedArray:
    return np.ma.masked_invalid(np.asarray(values, dtype=np.float32))


def process_archive(
    archive: Path,
    spec: VariableSpec,
    year: int,
    start_year: int,
    output_variable,
    corrected_mask: np.ndarray,
    target_depths: np.ndarray,
) -> list[dict]:
    pattern = re.compile(
        rf"^{re.escape(spec.source_name)}_ORAS5_1m_(\d{{6}})_r1x1\.nc$"
    )
    seen_months: set[int] = set()
    depth_mapping = None
    with tarfile.open(archive, mode="r|gz") as container:
        for member in container:
            match = pattern.fullmatch(Path(member.name).name)
            if not member.isfile() or match is None:
                continue
            year_month = int(match.group(1))
            member_year, month = divmod(year_month, 100)
            if member_year != year or not 1 <= month <= 12:
                continue
            handle = container.extractfile(member)
            if handle is None:
                raise RuntimeError(f"无法读取 archive member: {member.name}")
            payload = handle.read()
            with netCDF4.Dataset("oras5-memory.nc", mode="r", memory=payload) as source:
                raw = source.variables[spec.source_name][0]
                values = np.ma.filled(raw, np.nan).astype(np.float32, copy=False)
                if spec.dimensionality == 3:
                    source_depths = np.asarray(
                        source.variables["deptht"][:], dtype=np.float64
                    )
                    converted = interpolate_masked_depths(
                        values,
                        corrected_mask,
                        source_depths,
                        target_depths,
                    )
                    if depth_mapping is None:
                        lower, upper, weight = depth_brackets(
                            source_depths, target_depths
                        )
                        depth_mapping = [
                            {
                                "target_m": float(target_depths[index]),
                                "lower_source_m": float(source_depths[lower[index]]),
                                "upper_source_m": float(source_depths[upper[index]]),
                                "upper_weight": float(weight[index]),
                            }
                            for index in range(len(target_depths))
                        ]
                else:
                    converted = apply_surface_mask(values, corrected_mask)
            time_index = (year - start_year) * 12 + month - 1
            output_variable[time_index] = _masked_array(converted)
            seen_months.add(month)

    expected = set(range(1, 13))
    if seen_months != expected:
        raise RuntimeError(
            f"{archive} 月份不完整: missing={sorted(expected - seen_months)}"
        )
    return depth_mapping or []


def validate_output(
    path: Path,
    specs: tuple[VariableSpec, ...],
    expected_months: int,
    expected_depths: int,
) -> None:
    with netCDF4.Dataset(path) as dataset:
        expected_dimensions = {
            "TIME": expected_months,
            "LEVEL": expected_depths,
            "LATITUDE": 180,
            "LONGITUDE": 360,
        }
        actual = {name: len(dataset.dimensions[name]) for name in expected_dimensions}
        if actual != expected_dimensions:
            raise ValueError(f"ORAS5 output dimensions mismatch: {actual}")
        for spec in specs:
            variable = dataset.variables[spec.output_name]
            for time_index in (0, expected_months - 1):
                if np.ma.count(variable[time_index]) == 0:
                    raise ValueError(
                        f"{spec.output_name} time index {time_index} is entirely missing"
                    )


def estimate_report(
    output: Path,
    specs: tuple[VariableSpec, ...],
    years: range,
    depths: tuple[float, ...],
    keep_archives: bool,
) -> dict:
    months = len(years) * 12
    cells_2d = months * 180 * 360
    raw_bytes = 0
    for spec in specs:
        multiplier = len(depths) if spec.dimensionality == 3 else 1
        raw_bytes += cells_2d * multiplier * np.dtype("float32").itemsize

    sample_archive_bytes = {}
    for spec in specs:
        size = remote_size(archive_url(spec, years.start))
        sample_archive_bytes[spec.output_name] = size
    if all(size is not None for size in sample_archive_bytes.values()):
        approximate_download = sum(sample_archive_bytes.values()) * len(years)
    else:
        approximate_download = None

    disk_root = output.parent if output.parent.exists() else Path.cwd()
    free_bytes = shutil.disk_usage(disk_root).free
    conservative_peak = raw_bytes
    if keep_archives and approximate_download is not None:
        conservative_peak += approximate_download
    else:
        conservative_peak += max(
            (size or 0) for size in sample_archive_bytes.values()
        )
    return {
        "source": "ICDC ORAS5 r1x1 opa0",
        "year_range": [years.start, years.stop - 1],
        "month_count": months,
        "variables": [spec.output_name for spec in specs],
        "depths_m": list(depths),
        "uncompressed_output_upper_bound_gib": raw_bytes / 2**30,
        "approximate_download_gib": (
            approximate_download / 2**30
            if approximate_download is not None else None
        ),
        "conservative_peak_gib": conservative_peak / 2**30,
        "local_free_gib": free_bytes / 2**30,
        "fits_by_conservative_estimate": free_bytes > conservative_peak,
        "archives_retained": keep_archives,
        "output": str(output),
        "note": (
            "NetCDF compression normally makes the final file smaller than the "
            "uncompressed upper bound. Training preprocessing caches require "
            "additional space and are not included."
        ),
    }


def prepare(args: argparse.Namespace) -> Path:
    if args.end_year < args.start_year:
        raise ValueError("--end-year 必须不小于 --start-year")
    if args.start_year < ICDC_FIRST_YEAR or args.end_year > ICDC_LAST_YEAR:
        raise ValueError(
            "ICDC ORAS5 r1x1 opa0 仅覆盖 "
            f"{ICDC_FIRST_YEAR}-{ICDC_LAST_YEAR}；实际请求为 "
            f"{args.start_year}-{args.end_year}"
        )
    years = range(args.start_year, args.end_year + 1)
    specs = select_specs(args.variables)
    depths = args.depths
    output = args.output.resolve()
    work_dir = args.work_dir.resolve()
    partial = output.with_suffix(output.suffix + ".partial")
    state_path = work_dir / "prepare_state.json"

    if args.dry_run:
        print(json.dumps(
            estimate_report(output, specs, years, depths, args.keep_archives),
            indent=2,
            ensure_ascii=False,
        ))
        return output

    if args.overwrite:
        for path in (output, partial, state_path):
            if path.exists():
                path.unlink()
    elif output.exists():
        raise FileExistsError(f"输出已存在: {output}；如需重建请使用 --overwrite")

    payload = _config_payload(specs, years, depths)
    masks, latitudes, longitudes = load_corrected_masks(work_dir, specs)
    if latitudes.shape != (180,) or longitudes.shape != (360,):
        raise ValueError(
            f"ICDC r1x1 grid changed: lat={latitudes.shape}, lon={longitudes.shape}"
        )
    if state_path.exists() or partial.exists():
        state = _load_or_initialize_state(state_path, partial, payload)
        output_dataset = netCDF4.Dataset(partial, mode="r+")
    else:
        output_dataset = _create_output(
            partial, specs, years, depths, latitudes, longitudes
        )
        state = {"fingerprint": _fingerprint(payload), "completed": []}
        _write_json_atomic(state_path, state)

    completed = set(state.get("completed", []))
    depth_mappings: dict[str, list[dict]] = state.get("depth_mappings", {})
    archives_dir = work_dir / "archives"
    try:
        for spec in specs:
            output_variable = output_dataset.variables[spec.output_name]
            corrected_mask = masks[spec.mask_file]
            for year in years:
                completion_key = f"{spec.output_name}:{year}"
                if completion_key in completed:
                    continue
                url = archive_url(spec, year)
                archive = archives_dir / spec.source_name / Path(url).name
                print(f"[{spec.output_name}] {year}: {url}")
                download_file(url, archive)
                mapping = process_archive(
                    archive,
                    spec,
                    year,
                    years.start,
                    output_variable,
                    corrected_mask,
                    np.asarray(depths, dtype=np.float64),
                )
                output_dataset.sync()
                if mapping:
                    depth_mappings[spec.output_name] = mapping
                completed.add(completion_key)
                state["completed"] = sorted(completed)
                state["depth_mappings"] = depth_mappings
                _write_json_atomic(state_path, state)
                if not args.keep_archives:
                    archive.unlink(missing_ok=True)
    finally:
        output_dataset.close()

    validate_output(partial, specs, len(years) * 12, len(depths))
    os.replace(partial, output)
    provenance = {
        **payload,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "output": str(output),
        "source_doi": ORAS5_DOI,
        "license": "CC-BY-4.0",
        "mask_correction": MASK_CORRECTION_URL,
        "mask_correction_applied": True,
        "depth_mappings": depth_mappings,
    }
    _write_json_atomic(output.with_suffix(".provenance.json"), provenance)
    state_path.unlink(missing_ok=True)
    print(f"ORAS5 数据准备完成: {output}")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=1979)
    parser.add_argument("--end-year", type=int, default=2014)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Data/oras5/ORAS5_197901_201412_1deg.nc"),
    )
    parser.add_argument(
        "--work-dir", type=Path, default=Path("Data/oras5/work")
    )
    parser.add_argument(
        "--variables",
        nargs="+",
        default=[spec.output_name for spec in VARIABLE_SPECS],
        help="输出变量名；默认准备完整的十变量方案",
    )
    parser.add_argument(
        "--depths",
        type=parse_depths,
        default=DEFAULT_DEPTHS_M,
        help="逗号分隔的目标深度（米）",
    )
    parser.add_argument("--keep-archives", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    prepare(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
