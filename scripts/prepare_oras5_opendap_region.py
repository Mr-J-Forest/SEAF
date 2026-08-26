#!/usr/bin/env python3
"""Prepare a fast regional ORAS5 dataset through ICDC monthly OPeNDAP.

The regular ORAS5 preparer downloads one compressed global archive per
variable/year.  ICDC also exposes each monthly NetCDF file through OPeNDAP.
This script uses that service to request only the configured latitude,
longitude, and source-depth slices, which is substantially faster for a
regional experiment.  The output keeps the project's standard
TIME/LEVEL/LATITUDE/LONGITUDE schema, but contains only the requested region.

The global tar.gz workflow remains in ``prepare_oras5.py``.  This regional
workflow is intentionally explicit so a reduced spatial domain cannot be
mistaken for the full-grid dataset.
"""

from __future__ import annotations

import argparse
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
import hashlib
import json
import multiprocessing as mp
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import netCDF4
import numpy as np

try:  # Package import when called from tests or ``python -m``.
    from .prepare_oras5 import (
        DEFAULT_DEPTHS_M,
        ICDC_BASE_URL,
        MASK_CORRECTION_URL,
        ORAS5_DOI,
        VARIABLE_SPECS,
        VariableSpec,
        _create_output,
        _masked_array,
        _write_json_atomic,
        apply_surface_mask,
        depth_brackets,
        interpolate_masked_depths,
        load_corrected_masks,
        parse_depths,
        select_specs,
    )
except ImportError:  # Direct script execution from the repository root.
    from prepare_oras5 import (
        DEFAULT_DEPTHS_M,
        ICDC_BASE_URL,
        MASK_CORRECTION_URL,
        ORAS5_DOI,
        VARIABLE_SPECS,
        VariableSpec,
        _create_output,
        _masked_array,
        _write_json_atomic,
        apply_surface_mask,
        depth_brackets,
        interpolate_masked_depths,
        load_corrected_masks,
        parse_depths,
        select_specs,
    )


ICDC_DODS_BASE_URL = ICDC_BASE_URL.replace(
    "/thredds/fileServer/", "/thredds/dodsC/"
)


def parse_range(raw: str, option: str) -> tuple[float, float]:
    """Parse a comma-separated inclusive coordinate range."""
    try:
        values = tuple(float(item.strip()) for item in raw.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"{option} 必须是 min,max"
        ) from exc
    if len(values) != 2 or not np.isfinite(values).all():
        raise argparse.ArgumentTypeError(f"{option} 必须是 min,max")
    if values[1] <= values[0]:
        raise argparse.ArgumentTypeError(f"{option} 必须严格递增")
    return values


def monthly_url(spec: VariableSpec, year: int, month: int) -> str:
    """Return the ICDC monthly NetCDF OPeNDAP URL."""
    filename = f"{spec.source_name}_ORAS5_1m_{year}{month:02d}_r1x1.nc"
    return f"{ICDC_DODS_BASE_URL}/{spec.source_name}/opa0/{filename}"


def _coordinate_slice(
    coordinates: np.ndarray,
    bounds: tuple[float, float],
    name: str,
) -> tuple[int, int, np.ndarray]:
    values = np.asarray(coordinates, dtype=np.float64)
    indices = np.flatnonzero(
        (values >= bounds[0] - 1e-6) & (values <= bounds[1] + 1e-6)
    )
    if indices.size == 0:
        raise ValueError(
            f"{name} 范围 {bounds} 不在 ICDC 坐标范围 "
            f"[{values.min()}, {values.max()}] 内"
        )
    expected = np.arange(indices[0], indices[-1] + 1)
    if not np.array_equal(indices, expected):
        raise ValueError(f"{name} 选区不是连续坐标切片")
    return int(indices[0]), int(indices[-1]), values[indices]


def _load_geometry(
    spec: VariableSpec,
    year: int,
    month: int,
    lat_range: tuple[float, float],
    lon_range: tuple[float, float],
) -> tuple[int, int, np.ndarray, int, int, np.ndarray, np.ndarray]:
    """Read coordinates once and resolve the requested inclusive slices."""
    url = monthly_url(spec, year, month)
    with netCDF4.Dataset(url, mode="r") as source:
        latitudes = np.asarray(source.variables["lat"][:], dtype=np.float32)
        longitudes = np.asarray(source.variables["lon"][:], dtype=np.float32)
        lat_start, lat_end, selected_lats = _coordinate_slice(
            latitudes, lat_range, "latitude"
        )
        lon_start, lon_end, selected_lons = _coordinate_slice(
            longitudes, lon_range, "longitude"
        )
        if spec.dimensionality == 3:
            depths = np.asarray(source.variables["deptht"][:], dtype=np.float64)
        else:
            depths = np.empty(0, dtype=np.float64)
    return (
        lat_start,
        lat_end,
        selected_lats,
        lon_start,
        lon_end,
        selected_lons,
        depths,
    )


def _read_month(
    spec: VariableSpec,
    year: int,
    month: int,
    *,
    lat_start: int,
    lat_end: int,
    lon_start: int,
    lon_end: int,
    source_depths: np.ndarray,
    source_depth_count: int,
    target_depths: np.ndarray,
    corrected_mask: np.ndarray,
    retries: int,
) -> np.ndarray:
    """Read and convert one regional monthly field with bounded retries."""
    url = monthly_url(spec, year, month)
    for attempt in range(1, retries + 1):
        source = None
        try:
            source = netCDF4.Dataset(url, mode="r")
            variable = source.variables[spec.source_name]
            if spec.dimensionality == 3:
                raw = variable[
                    0,
                    :source_depth_count,
                    lat_start : lat_end + 1,
                    lon_start : lon_end + 1,
                ]
                values = np.ma.filled(raw, np.nan).astype(np.float32, copy=False)
                converted = interpolate_masked_depths(
                    values,
                    corrected_mask,
                    source_depths[:source_depth_count],
                    target_depths,
                )
            else:
                raw = variable[
                    0,
                    lat_start : lat_end + 1,
                    lon_start : lon_end + 1,
                ]
                values = np.ma.filled(raw, np.nan).astype(np.float32, copy=False)
                converted = apply_surface_mask(values, corrected_mask)
            return np.asarray(converted, dtype=np.float32)
        except Exception as exc:
            if attempt == retries:
                raise RuntimeError(
                    f"OPeNDAP 月文件读取失败: {url}"
                ) from exc
            # The ICDC service can transiently reject or reset a request.  A
            # bounded exponential backoff avoids turning a transient 429 into
            # a sustained request storm while preserving parallelism.
            print(
                f"    {spec.output_name} {year}{month:02d} 读取中断，"
                f"第 {attempt}/{retries} 次重试: {exc}",
                flush=True,
            )
            time.sleep(min(2 ** (attempt - 1), 16))
        finally:
            if source is not None:
                source.close()
    raise AssertionError("unreachable")


def _payload(
    specs: tuple[VariableSpec, ...],
    years: range,
    depths: tuple[float, ...],
    lat_range: tuple[float, float],
    lon_range: tuple[float, float],
) -> dict:
    return {
        "source": ICDC_DODS_BASE_URL,
        "source_mode": "monthly_opendap_regional_subset",
        "years": [years.start, years.stop - 1],
        "depths_m": list(depths),
        "variables": [spec.output_name for spec in specs],
        "latitude_range": list(lat_range),
        "longitude_range": list(lon_range),
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
                "发现不完整的 OPeNDAP 输出但缺少匹配状态；"
                "请检查后使用 --overwrite"
            )
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("fingerprint") != fingerprint:
            raise RuntimeError(
                "现有 OPeNDAP partial 输出与本次参数不一致，请使用 --overwrite"
            )
        return state
    state = {"fingerprint": fingerprint, "completed": []}
    _write_json_atomic(state_path, state)
    return state


def _validate_output(
    path: Path,
    specs: tuple[VariableSpec, ...],
    expected_months: int,
    expected_depths: int,
    expected_lats: int,
    expected_lons: int,
) -> None:
    with netCDF4.Dataset(path, mode="r") as dataset:
        expected = {
            "TIME": expected_months,
            "LEVEL": expected_depths,
            "LATITUDE": expected_lats,
            "LONGITUDE": expected_lons,
        }
        actual = {name: len(dataset.dimensions[name]) for name in expected}
        if actual != expected:
            raise ValueError(f"OPeNDAP 区域输出维度不匹配: {actual}")
        for spec in specs:
            variable = dataset.variables[spec.output_name]
            for time_index in (0, expected_months - 1):
                if np.ma.count(variable[time_index]) == 0:
                    raise ValueError(
                        f"{spec.output_name} time index {time_index} 全部缺失"
                    )


def prepare(args: argparse.Namespace) -> Path:
    if args.end_year < args.start_year:
        raise ValueError("--end-year 必须不小于 --start-year")
    if args.workers < 1:
        raise ValueError("--workers 必须是正整数")
    if not (1979 <= args.start_year <= 2018 and 1979 <= args.end_year <= 2018):
        raise ValueError("ICDC ORAS5 月文件仅覆盖 1979-2018")
    years = range(args.start_year, args.end_year + 1)
    specs = select_specs(args.variables)
    depths = args.depths
    output = args.output.resolve()
    work_dir = args.work_dir.resolve()
    partial = output.with_suffix(output.suffix + ".partial")
    state_path = output.with_suffix(output.suffix + ".state.json")

    geometry_spec = next(
        (spec for spec in specs if spec.dimensionality == 3), specs[0]
    )
    (
        lat_start,
        lat_end,
        latitudes,
        lon_start,
        lon_end,
        longitudes,
        source_depths,
    ) = _load_geometry(
        geometry_spec,
        years.start,
        1,
        args.lat_range,
        args.lon_range,
    )
    if geometry_spec.dimensionality != 3:
        source_depths = np.empty(0, dtype=np.float64)
    _, upper, _ = (
        depth_brackets(source_depths, np.asarray(depths, dtype=np.float64))
        if source_depths.size
        else (None, None, None)
    )
    source_depth_count = int(max(upper) + 1) if upper is not None else 0

    payload = _payload(specs, years, depths, args.lat_range, args.lon_range)
    if args.dry_run:
        cells = len(years) * 12 * len(latitudes) * len(longitudes)
        print(json.dumps({
            **payload,
            "months": len(years) * 12,
            "regional_grid": [len(latitudes), len(longitudes)],
            "approximate_raw_float32_gib": (
                cells * sum(
                    len(depths) if spec.dimensionality == 3 else 1
                    for spec in specs
                ) * 4 / 2**30
            ),
            "workers": args.workers,
        }, indent=2, ensure_ascii=False))
        return output

    if args.overwrite:
        for path in (output, partial, state_path):
            path.unlink(missing_ok=True)
    elif output.exists():
        raise FileExistsError(
            f"输出已存在: {output}；如需重建请使用 --overwrite"
        )

    masks, mask_lats, mask_lons = load_corrected_masks(
        work_dir, specs, download_workers=1
    )
    # The geometry comes from the same ICDC grid; this explicit check catches
    # an accidental source/mask mismatch before any partial output is written.
    if not np.allclose(mask_lats[lat_start : lat_end + 1], latitudes):
        raise ValueError("OPeNDAP 数据与修正版 mask 的纬度坐标不一致")
    if not np.allclose(mask_lons[lon_start : lon_end + 1], longitudes):
        raise ValueError("OPeNDAP 数据与修正版 mask 的经度坐标不一致")

    state = _load_or_initialize_state(state_path, partial, payload)
    if partial.exists():
        output_dataset = netCDF4.Dataset(partial, mode="r+")
    else:
        output_dataset = _create_output(
            partial, specs, years, depths, latitudes, longitudes
        )
        output_dataset.source = "ECMWF ORAS5 via ICDC monthly OPeNDAP subset"
        output_dataset.history = (
            f"Prepared {datetime.now(timezone.utc).isoformat()} by "
            "scripts/prepare_oras5_opendap_region.py"
        )
        output_dataset.sync()

    target_depths = np.asarray(depths, dtype=np.float64)
    regional_masks: dict[str, np.ndarray] = {}
    for spec in specs:
        mask = np.asarray(masks[spec.mask_file], dtype=bool)
        if spec.dimensionality == 3:
            regional_masks[spec.output_name] = mask[
                :source_depth_count,
                lat_start : lat_end + 1,
                lon_start : lon_end + 1,
            ]
        else:
            regional_masks[spec.output_name] = mask[
                lat_start : lat_end + 1,
                lon_start : lon_end + 1,
            ]

    completed = set(state.get("completed", []))
    tasks = [
        (spec, year, month)
        for spec in specs
        for year in years
        for month in range(1, 13)
        if f"{spec.output_name}:{year}{month:02d}" not in completed
    ]
    total = len(tasks) + len(completed)
    executor_kind = args.executor
    if executor_kind == "auto":
        executor_kind = "process" if args.workers > 8 else "thread"
    print(
        f"OPeNDAP 区域准备: {len(years) * 12} 个月 × {len(specs)} 变量，"
        f"区域 {len(latitudes)}×{len(longitudes)}，{args.workers} "
        f"{executor_kind} workers；"
        f"已完成 {len(completed)}/{total}",
        flush=True,
    )
    started = time.time()
    finished = len(completed)
    try:
        if executor_kind == "process":
            # Spawn rather than fork: netCDF4/libcurl keeps native global
            # state, and inheriting it across a fork can reintroduce the
            # thread-safety crash this backend is designed to avoid.
            executor_context = ProcessPoolExecutor(
                max_workers=args.workers,
                mp_context=mp.get_context("spawn"),
            )
        else:
            executor_context = ThreadPoolExecutor(max_workers=args.workers)
        with executor_context as executor:
            futures = {
                executor.submit(
                    _read_month,
                    spec,
                    year,
                    month,
                    lat_start=lat_start,
                    lat_end=lat_end,
                    lon_start=lon_start,
                    lon_end=lon_end,
                    source_depths=source_depths,
                    source_depth_count=source_depth_count,
                    target_depths=target_depths,
                    corrected_mask=regional_masks[spec.output_name],
                    retries=args.retries,
                ): (spec, year, month)
                for spec, year, month in tasks
            }
            for future in as_completed(futures):
                spec, year, month = futures[future]
                converted = future.result()
                time_index = (year - years.start) * 12 + month - 1
                output_dataset.variables[spec.output_name][time_index] = _masked_array(
                    converted
                )
                key = f"{spec.output_name}:{year}{month:02d}"
                completed.add(key)
                finished += 1
                if finished % args.state_every == 0 or finished == total:
                    output_dataset.sync()
                    state["completed"] = sorted(completed)
                    state["geometry"] = {
                        "latitude_range": list(args.lat_range),
                        "longitude_range": list(args.lon_range),
                        "latitude_count": len(latitudes),
                        "longitude_count": len(longitudes),
                    }
                    _write_json_atomic(state_path, state)
                    elapsed = max(time.time() - started, 1e-6)
                    rate = finished / elapsed
                    print(
                        f"  进度 {finished}/{total} ({finished / total:.1%})，"
                        f"累计 {elapsed / 60:.1f} min，约 {rate:.2f} files/s",
                        flush=True,
                    )
    finally:
        output_dataset.close()

    _validate_output(
        partial,
        specs,
        len(years) * 12,
        len(depths),
        len(latitudes),
        len(longitudes),
    )
    os.replace(partial, output)
    provenance = {
        **payload,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "output": str(output),
        "source_doi": ORAS5_DOI,
        "license": "CC-BY-4.0",
        "mask_correction": MASK_CORRECTION_URL,
        "mask_correction_applied": True,
        "original_grid": {"latitude_count": 180, "longitude_count": 360},
        "regional_grid": {
            "latitude_count": len(latitudes),
            "longitude_count": len(longitudes),
        },
    }
    _write_json_atomic(output.with_suffix(".provenance.json"), provenance)
    state_path.unlink(missing_ok=True)
    print(f"ORAS5 OPeNDAP 区域数据准备完成: {output}", flush=True)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=1979)
    parser.add_argument("--end-year", type=int, default=2014)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Data/oras5/ORAS5_197901_201412_1deg_region.nc"),
    )
    parser.add_argument(
        "--work-dir", type=Path, default=Path("Data/oras5/work")
    )
    parser.add_argument(
        "--variables",
        type=lambda raw: tuple(item.strip() for item in raw.split(",") if item.strip()),
        default=tuple(spec.output_name for spec in VARIABLE_SPECS),
    )
    parser.add_argument(
        "--depths",
        type=parse_depths,
        default=DEFAULT_DEPTHS_M,
    )
    parser.add_argument(
        "--lat-range",
        type=lambda raw: parse_range(raw, "--lat-range"),
        default=(6.5, 27.5),
    )
    parser.add_argument(
        "--lon-range",
        type=lambda raw: parse_range(raw, "--lon-range"),
        default=(130.0, 162.0),
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--executor",
        choices=("auto", "thread", "process"),
        default="auto",
        help=(
            "并发后端；auto 在 workers>8 时使用独立进程，"
            "避免 netCDF4 原生库线程不安全"
        ),
    )
    parser.add_argument("--retries", type=int, default=6)
    parser.add_argument("--state-every", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.state_every < 1:
        raise SystemExit("--state-every 必须是正整数")
    prepare(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
