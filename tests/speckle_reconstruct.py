#!/usr/bin/env python3
from pathlib import Path
import csv
import sqlite3
import struct
import sys
import zlib

import numpy as np


CONFIG = {
    "test_dataset_dir": "/opt/data/CLEAR2026/runs/run_20260624T084552Z/png_dataset/",
    "train_dataset_dir": "/opt/data/CLEAR2026/runs/run_20260624T093655Z/png_dataset/",
    "dark_dataset_dir": None,  # optional laser-off png_dataset; averaged per camera
    "output_dir": "/opt/data/CLEAR2026/output/reconstructed_wednesday/",

    # One final side-by-side image will be saved for each selected cam2 sample.
    # With process_all_test_samples=False, this should normally contain one sample name only.
    "test_sample_names": [""],
    "process_all_test_samples": True,
    "max_test_samples": 500,

    "target_camera": "cam2",
    "source_cameras": ["cam1", "cam3"],
    "skip_missing_source_cameras": True,

    # Reconstruction method.
    "coefficient_method": "ridge",  # dot | ridge | pinv
    "ridge_lambda": 1e-2,
    "pinv_rcond": 1e-6,
    "coefficient_threshold": 0.0,
    "normalize_coefficients": False,

    # Source/speckle preprocessing. This is applied only to source-camera images.
    "normalize_speckles": True,
    "source_background_percentile": 5.0,
    "clip_after_background": True,
    "center_speckles": True,
    "dark_max_frames_per_camera": 50,
    "clip_after_dark_subtraction": True,

    "exclude_same_sample_when_same_dataset": True,

    # Ridge/pinv needs the training source cache.
    # Preprocessed source cache is float32, so it needs about 2x the memory of uint16 raw PNG arrays.
    "cache_training_source_images": True,
    "cache_training_target_images": False,
    "cache_max_gb": 24.0,
    "cache_chunk_rows": 32,

    # Output control.
    "save_final_side_by_side": True,
    "save_reconstruction_pngs": False,
    "save_coefficients_csv": False,
    "side_by_side_suffix": "side_by_side",

    "show_progress": True,
    "progress_bar_width": 28,
}


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def load_pairs(dataset_dir, target_camera, source_camera):
    dataset_dir = Path(dataset_dir).expanduser()
    conn = sqlite3.connect(dataset_dir / "dataset.db")
    rows = conn.execute(
        """
        SELECT gt.sample_id, gt.image_id, gt.png_path, sp.image_id, sp.png_path
        FROM images AS gt
        JOIN images AS sp ON gt.sample_id = sp.sample_id
        WHERE gt.camera = ? AND sp.camera = ?
        ORDER BY gt.sample_id
        """,
        (target_camera, source_camera),
    ).fetchall()
    conn.close()
    return [
        {
            "sample_id": sample_id,
            "target_id": target_id,
            "target_path": dataset_dir / target_path,
            "source_id": source_id,
            "source_path": dataset_dir / source_path,
            "source_camera": source_camera,
        }
        for sample_id, target_id, target_path, source_id, source_path in rows
    ]


def _paeth(a, b, c):
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def read_png_gray(path):
    data = Path(path).read_bytes()
    if not data.startswith(PNG_SIGNATURE):
        raise RuntimeError(f"not a PNG file: {path}")

    pos = len(PNG_SIGNATURE)
    width = height = bit_depth = color_type = interlace = None
    idat = []
    while pos < len(data):
        length = struct.unpack(">I", data[pos:pos + 4])[0]
        chunk_type = data[pos + 4:pos + 8]
        chunk = data[pos + 8:pos + 8 + length]
        pos += 12 + length
        if chunk_type == b"IHDR":
            width, height, bit_depth, color_type, compression, png_filter, interlace = struct.unpack(
                ">IIBBBBB", chunk
            )
            if compression != 0 or png_filter != 0 or interlace != 0:
                raise RuntimeError(f"unsupported PNG compression/filter/interlace in {path}")
        elif chunk_type == b"IDAT":
            idat.append(chunk)
        elif chunk_type == b"IEND":
            break

    if color_type != 0 or bit_depth not in (8, 16):
        raise RuntimeError(f"expected 8/16-bit grayscale PNG, got bit_depth={bit_depth}, color_type={color_type}")

    bytes_per_pixel = 1 if bit_depth == 8 else 2
    row_bytes = width * bytes_per_pixel
    raw = zlib.decompress(b"".join(idat))
    expected = height * (row_bytes + 1)
    if len(raw) != expected:
        raise RuntimeError(f"bad PNG payload length in {path}: expected {expected}, got {len(raw)}")

    out = bytearray(height * row_bytes)
    prev = bytearray(row_bytes)
    for row in range(height):
        src = row * (row_bytes + 1)
        filter_type = raw[src]
        cur = bytearray(raw[src + 1:src + 1 + row_bytes])
        if filter_type == 1:
            for i in range(row_bytes):
                left = cur[i - bytes_per_pixel] if i >= bytes_per_pixel else 0
                cur[i] = (cur[i] + left) & 0xFF
        elif filter_type == 2:
            for i in range(row_bytes):
                cur[i] = (cur[i] + prev[i]) & 0xFF
        elif filter_type == 3:
            for i in range(row_bytes):
                left = cur[i - bytes_per_pixel] if i >= bytes_per_pixel else 0
                cur[i] = (cur[i] + ((left + prev[i]) // 2)) & 0xFF
        elif filter_type == 4:
            for i in range(row_bytes):
                left = cur[i - bytes_per_pixel] if i >= bytes_per_pixel else 0
                up = prev[i]
                up_left = prev[i - bytes_per_pixel] if i >= bytes_per_pixel else 0
                cur[i] = (cur[i] + _paeth(left, up, up_left)) & 0xFF
        elif filter_type != 0:
            raise RuntimeError(f"unsupported PNG row filter {filter_type} in {path}")
        out[row * row_bytes:(row + 1) * row_bytes] = cur
        prev = cur

    dtype = np.uint8 if bit_depth == 8 else ">u2"
    arr = np.frombuffer(bytes(out), dtype=dtype).reshape((height, width))
    return arr.astype(np.uint16) if bit_depth == 16 else arr


def _png_chunk(chunk_type, payload):
    crc = zlib.crc32(chunk_type)
    crc = zlib.crc32(payload, crc) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + chunk_type + payload + struct.pack(">I", crc)


def write_png_gray(path, image):
    arr = np.asarray(image)
    if arr.ndim != 2:
        raise RuntimeError(f"PNG writer expects 2D grayscale image, got shape {arr.shape}")
    if arr.dtype == np.uint8:
        bit_depth = 8
        payload_rows = arr
    elif arr.dtype == np.uint16:
        bit_depth = 16
        payload_rows = arr.astype(">u2", copy=False)
    else:
        raise RuntimeError(f"PNG writer expects uint8 or uint16, got {arr.dtype}")

    height, width = arr.shape
    rows = [b"\x00" + np.ascontiguousarray(payload_rows[row]).tobytes() for row in range(height)]
    ihdr = struct.pack(">IIBBBBB", width, height, bit_depth, 0, 0, 0, 0)
    payload = (
        PNG_SIGNATURE
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", zlib.compress(b"".join(rows)))
        + _png_chunk(b"IEND", b"")
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def image_array(path):
    return read_png_gray(path)


def read_image(path):
    return image_array(path).astype(np.float32)


def warn(message):
    print(f"warning: {message}", flush=True)


def dark_correct(image, camera, dark_frames):
    dark = dark_frames.get(camera)
    if dark is None:
        return image
    if image.shape != dark.shape:
        raise RuntimeError(f"dark frame shape mismatch for {camera}: {dark.shape} != {image.shape}")
    image = image - dark
    if CONFIG["clip_after_dark_subtraction"]:
        image = np.maximum(image, 0.0)
    return image


def read_camera_image(path, camera, dark_frames):
    return dark_correct(read_image(path), camera, dark_frames)


def load_dark_frames(cameras):
    dark_dir = CONFIG.get("dark_dataset_dir")
    if not dark_dir:
        return {}

    dark_dir = Path(dark_dir).expanduser()
    db_path = dark_dir / "dataset.db"
    if not db_path.exists():
        warn(f"dark subtraction requested, but {db_path} does not exist")
        return {}

    max_frames = int(CONFIG["dark_max_frames_per_camera"])
    dark_frames = {}
    conn = sqlite3.connect(str(db_path))
    try:
        for camera in cameras:
            rows = conn.execute(
                """
                SELECT png_path
                FROM images
                WHERE camera = ?
                ORDER BY sample_id
                LIMIT ?
                """,
                (camera, max_frames),
            ).fetchall()
            if not rows:
                warn(f"no dark frames found for {camera}; no dark subtraction for this camera")
                continue

            total = None
            for (png_path,) in rows:
                path = Path(png_path)
                if not path.is_absolute():
                    path = dark_dir / path
                image = read_image(path)
                if total is None:
                    total = np.zeros_like(image, dtype=np.float32)
                elif image.shape != total.shape:
                    raise RuntimeError(f"dark frame shape changed for {camera}: {image.shape} != {total.shape}")
                total += image

            dark_frames[camera] = total / float(len(rows))
            print(f"dark subtraction enabled for {camera}: averaged {len(rows)} frame(s)", flush=True)
    finally:
        conn.close()

    return dark_frames


def preprocess_source_array(arr):
    """Preprocess a source/speckle image before dot products and Gram construction."""
    vec = np.asarray(arr, dtype=np.float32).ravel()

    percentile = CONFIG.get("source_background_percentile", None)
    if percentile is not None:
        bg = np.percentile(vec, float(percentile))
        vec = vec - bg
        if CONFIG.get("clip_after_background", True):
            vec = np.maximum(vec, 0.0)

    if CONFIG.get("center_speckles", True):
        vec = vec - float(vec.mean())

    if CONFIG["normalize_speckles"]:
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

    return vec.astype(np.float32, copy=False)


def preprocess_source_image(path, camera, dark_frames):
    return preprocess_source_array(read_camera_image(path, camera, dark_frames))


def speckle_vector(path, camera, dark_frames):
    return preprocess_source_image(path, camera, dark_frames)


def progress(current, total, label):
    if not CONFIG["show_progress"] or total <= 0:
        return
    current = min(int(current), int(total))
    step = max(1, int(total) // 100)
    if current not in (1, total) and current % step != 0:
        return
    width = int(CONFIG["progress_bar_width"])
    filled = int(round(width * current / float(total)))
    bar = "#" * filled + "-" * (width - filled)
    sys.stderr.write(f"\r{label} [{bar}] {current}/{total}")
    if current >= total:
        sys.stderr.write("\n")
    sys.stderr.flush()


def make_stack_cache(
    pairs,
    path_key,
    label,
    camera,
    dark_frames,
    already_cached_gb=0.0,
    preprocess_source=False,
):
    sample = read_camera_image(pairs[0][path_key], camera, dark_frames)
    n_images = len(pairs)
    n_pixels = int(sample.size)
    dtype = np.float32 if preprocess_source else sample.dtype
    itemsize = np.dtype(dtype).itemsize
    stack_gb = n_images * n_pixels * itemsize / (1024**3)
    total_gb = already_cached_gb + stack_gb
    if total_gb > float(CONFIG["cache_max_gb"]):
        print(
            f"skip {label} cache: needs {stack_gb:.2f} GB "
            f"({total_gb:.2f} GB total)",
            flush=True,
        )
        return None, already_cached_gb

    stack = np.empty((n_images, n_pixels), dtype=dtype)
    norms = np.empty(n_images, dtype=np.float64)
    shape = sample.shape
    for index, pair in enumerate(pairs, 1):
        arr = read_camera_image(pair[path_key], camera, dark_frames)
        if arr.shape != shape:
            raise RuntimeError(f"{label} image shape changed: {arr.shape} != {shape}")
        if preprocess_source:
            vec = preprocess_source_array(arr)
        else:
            vec = arr.ravel()
        stack[index - 1] = vec
        norms[index - 1] = np.linalg.norm(vec.astype(np.float32, copy=False))
        progress(index, n_images, f"cache {label}")

    print(f"cached {label}: {n_images} images, {stack_gb:.2f} GB", flush=True)
    return {
        "stack": stack,
        "norms": norms,
        "shape": shape,
        "gb": stack_gb,
        "preprocessed_source": bool(preprocess_source),
    }, total_gb


def cached_source_coeffs(test_path, source_camera, source_cache, dark_frames, progress_label):
    if source_cache.get("preprocessed_source", False):
        test = preprocess_source_image(test_path, source_camera, dark_frames)
    else:
        test = read_camera_image(test_path, source_camera, dark_frames).ravel()

    coeffs = np.zeros(source_cache["stack"].shape[0], dtype=np.float64)
    chunk_rows = int(CONFIG["cache_chunk_rows"])
    for start in range(0, len(coeffs), chunk_rows):
        stop = min(start + chunk_rows, len(coeffs))
        chunk = source_cache["stack"][start:stop].astype(np.float32)
        coeffs[start:stop] = chunk @ test
        progress(stop, len(coeffs), progress_label)

    if CONFIG["normalize_speckles"] and not source_cache.get("preprocessed_source", False):
        denom = source_cache["norms"] * np.linalg.norm(test)
        coeffs = np.divide(coeffs, denom, out=np.zeros_like(coeffs), where=denom > 0)
    return coeffs


def source_gram(source_cache, progress_label):
    stack = source_cache["stack"]
    n_images = stack.shape[0]
    chunk_rows = int(CONFIG["cache_chunk_rows"])
    gram = np.empty((n_images, n_images), dtype=np.float64)

    for start in range(0, n_images, chunk_rows):
        stop = min(start + chunk_rows, n_images)
        a = stack[start:stop].astype(np.float32)
        for col_start in range(0, n_images, chunk_rows):
            col_stop = min(col_start + chunk_rows, n_images)
            b = stack[col_start:col_stop].astype(np.float32)
            gram[start:stop, col_start:col_stop] = a @ b.T
        progress(stop, n_images, progress_label)

    if CONFIG["normalize_speckles"] and not source_cache.get("preprocessed_source", False):
        denom = np.outer(source_cache["norms"], source_cache["norms"])
        gram = np.divide(gram, denom, out=np.zeros_like(gram), where=denom > 0)

    gram = 0.5 * (gram + gram.T)
    return gram


def solve_coefficients(dot_coeffs, gram, active_mask):
    method = str(CONFIG["coefficient_method"]).strip().lower()
    if method == "dot":
        coeffs = dot_coeffs.copy()
        coeffs[~active_mask] = 0.0
        return coeffs

    active = np.flatnonzero(active_mask)
    if active.size == 0:
        return np.zeros_like(dot_coeffs)

    lhs = gram[np.ix_(active, active)]
    rhs = dot_coeffs[active]

    if method in ("ridge", "tikhonov"):
        lhs = lhs.copy()
        lhs.flat[:: lhs.shape[0] + 1] += float(CONFIG["ridge_lambda"])
        try:
            solved = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            solved = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    elif method in ("pinv", "pseudo_inverse", "pseudoinverse"):
        solved = np.linalg.pinv(lhs, rcond=float(CONFIG["pinv_rcond"])) @ rhs
    else:
        raise RuntimeError("coefficient_method must be dot, ridge, or pinv")

    coeffs = np.zeros_like(dot_coeffs)
    coeffs[active] = solved
    return coeffs


def cached_target_sum(coeffs, target_cache):
    recon = np.zeros(target_cache["stack"].shape[1], dtype=np.float64)
    chunk_rows = int(CONFIG["cache_chunk_rows"])
    for start in range(0, len(coeffs), chunk_rows):
        stop = min(start + chunk_rows, len(coeffs))
        chunk = target_cache["stack"][start:stop].astype(np.float32)
        recon += coeffs[start:stop].astype(np.float32) @ chunk
    return recon.reshape(target_cache["shape"])


def resolve_tests(pairs, names):
    by_name = {}
    for pair in pairs:
        by_name[pair["target_id"]] = pair
        by_name[pair["target_path"].name] = pair
        by_name[pair["target_path"].stem] = pair
    selected = []
    for name in names:
        path = Path(name)
        key = path.stem if path.suffix else path.name
        if key not in by_name:
            raise KeyError(f"unknown test sample {name!r}; expected a target image_id or filename")
        selected.append(by_name[key])
    return selected


def select_test_pairs(test_pairs):
    if CONFIG["process_all_test_samples"]:
        selected = list(test_pairs)
    else:
        selected = resolve_tests(test_pairs, CONFIG["test_sample_names"])

    max_test_samples = CONFIG["max_test_samples"]
    if max_test_samples is not None:
        selected = selected[:int(max_test_samples)]
    return selected


def display_u8(image):
    image = np.asarray(image, dtype=np.float64)
    lo, hi = np.percentile(image, [1, 99.8])
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((image - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)


def save_side_by_side(path, panels):
    """Save one PNG: original target | reconstruction(s). No labels, just panels."""
    if len(panels) < 2:
        raise RuntimeError("need original image plus at least one reconstruction to save side-by-side")

    images = [np.asarray(image, dtype=np.uint8) for _, image in panels]
    height = max(image.shape[0] for image in images)
    padded = []
    for image in images:
        if image.shape[0] < height:
            pad = np.zeros((height - image.shape[0], image.shape[1]), dtype=np.uint8)
            image = np.vstack([image, pad])
        padded.append(image)
        padded.append(np.full((height, 8), 255, dtype=np.uint8))
    write_png_gray(path, np.hstack(padded[:-1]))


def reconstruct_one(
    test_pair,
    train_pairs,
    source_camera,
    dark_frames,
    source_cache=None,
    target_cache=None,
    gram=None,
):
    same_dataset = Path(CONFIG["test_dataset_dir"]).resolve() == Path(CONFIG["train_dataset_dir"]).resolve()

    if source_cache is not None:
        dot_coeffs = cached_source_coeffs(
            test_pair["source_path"],
            source_camera,
            source_cache,
            dark_frames,
            f"{source_camera} dot",
        )
    else:
        test_vec = speckle_vector(test_pair["source_path"], source_camera, dark_frames)
        dot_coeffs = []
        for index, pair in enumerate(train_pairs, 1):
            dot_coeffs.append(
                float(np.dot(speckle_vector(pair["source_path"], source_camera, dark_frames), test_vec))
            )
            progress(index, len(train_pairs), f"{source_camera} dot")

    dot_coeffs = np.asarray(dot_coeffs, dtype=np.float64)
    active_mask = np.ones(len(train_pairs), dtype=bool)
    if same_dataset and CONFIG["exclude_same_sample_when_same_dataset"]:
        for index, pair in enumerate(train_pairs):
            if pair["sample_id"] == test_pair["sample_id"]:
                active_mask[index] = False

    coeffs = solve_coefficients(dot_coeffs, gram, active_mask)
    coeffs[np.abs(coeffs) < float(CONFIG["coefficient_threshold"])] = 0.0

    if CONFIG["normalize_coefficients"] and coeffs.sum() != 0.0:
        coeffs = coeffs / coeffs.sum()

    if target_cache is not None:
        recon = cached_target_sum(coeffs, target_cache)
    else:
        target_camera = CONFIG["target_camera"]
        recon = np.zeros_like(
            read_camera_image(train_pairs[0]["target_path"], target_camera, dark_frames),
            dtype=np.float64,
        )
        for coeff, pair in zip(coeffs, train_pairs):
            if coeff == 0.0:
                continue
            recon += coeff * read_camera_image(pair["target_path"], target_camera, dark_frames)

    used = int(np.count_nonzero(coeffs))
    return recon, coeffs, used, len(train_pairs)


def save_coefficients_csv(path, coeffs, train_pairs):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "target_id", "source_id", "coefficient"])
        for coeff, pair in zip(coeffs, train_pairs):
            writer.writerow([pair["sample_id"], pair["target_id"], pair["source_id"], coeff])


def remove_file(path):
    try:
        Path(path).unlink()
    except FileNotFoundError:
        pass


def main():
    output_dir = Path(CONFIG["output_dir"]).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not CONFIG["save_final_side_by_side"]:
        raise RuntimeError("save_final_side_by_side must be True for this simplified version")

    cameras = [CONFIG["target_camera"]]
    for camera in CONFIG["source_cameras"]:
        if camera not in cameras:
            cameras.append(camera)
    dark_frames = load_dark_frames(cameras)

    jobs = []
    expected_sources = {}
    target_paths = {}
    for source_camera in CONFIG["source_cameras"]:
        test_pairs = load_pairs(CONFIG["test_dataset_dir"], CONFIG["target_camera"], source_camera)
        train_pairs = load_pairs(CONFIG["train_dataset_dir"], CONFIG["target_camera"], source_camera)
        if not test_pairs or not train_pairs:
            if CONFIG["skip_missing_source_cameras"]:
                print(f"skip {source_camera}: missing in test or training dataset", flush=True)
                continue
            raise RuntimeError(f"{source_camera} missing in test or training dataset")

        selected_tests = select_test_pairs(test_pairs)
        if not selected_tests:
            print(f"skip {source_camera}: no selected test samples", flush=True)
            continue

        jobs.append((source_camera, train_pairs, selected_tests))
        for test_pair in selected_tests:
            target_id = test_pair["target_id"]
            target_paths[target_id] = test_pair["target_path"]
            expected_sources.setdefault(target_id, set()).add(source_camera)

    if not jobs:
        raise RuntimeError("no reconstructions were produced")

    tmp_dir = output_dir / "_tmp_recon"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    temp_recons = {}

    for source_camera, train_pairs, selected_tests in jobs:
        cached_gb = 0.0
        source_cache = None
        target_cache = None
        gram = None

        if CONFIG["cache_training_source_images"]:
            source_cache, cached_gb = make_stack_cache(
                train_pairs,
                "source_path",
                f"training {source_camera}",
                source_camera,
                dark_frames,
                cached_gb,
                preprocess_source=True,
            )

        if CONFIG["cache_training_target_images"]:
            target_cache, cached_gb = make_stack_cache(
                train_pairs,
                "target_path",
                f"training {CONFIG['target_camera']}",
                CONFIG["target_camera"],
                dark_frames,
                cached_gb,
                preprocess_source=False,
            )

        method = str(CONFIG["coefficient_method"]).strip().lower()
        if method != "dot":
            if source_cache is None:
                raise RuntimeError(
                    "ridge/pinv coefficient_method requires cache_training_source_images=True "
                    "and enough cache_max_gb for the training source images"
                )
            gram = source_gram(source_cache, f"{source_camera} gram")

        for sample_index, test_pair in enumerate(selected_tests, 1):
            target_id = test_pair["target_id"]
            print(
                f"{source_camera} selected sample {sample_index}/{len(selected_tests)}: {target_id}",
                flush=True,
            )

            recon, coeffs, used, total = reconstruct_one(
                test_pair,
                train_pairs,
                source_camera,
                dark_frames,
                source_cache=source_cache,
                target_cache=target_cache,
                gram=gram,
            )

            temp_recon = tmp_dir / f"{target_id}_from_{source_camera}.png"
            write_png_gray(temp_recon, display_u8(recon))
            temp_recons.setdefault(target_id, {})[source_camera] = temp_recon

            if CONFIG["save_reconstruction_pngs"]:
                recon_png = output_dir / f"{target_id}_from_{source_camera}_reconstruction.png"
                write_png_gray(recon_png, image_array(temp_recon))
                print(f"  reconstruction: {recon_png}", flush=True)

            if CONFIG["save_coefficients_csv"]:
                coeff_csv = output_dir / f"{target_id}_from_{source_camera}_coefficients.csv"
                save_coefficients_csv(coeff_csv, coeffs, train_pairs)
                print(f"  coefficients:   {coeff_csv}", flush=True)

            print(f"  used {used}/{total} coefficients", flush=True)

            available = temp_recons[target_id]
            if expected_sources[target_id].issubset(available):
                panels = [(
                    f"original {CONFIG['target_camera']}",
                    display_u8(read_camera_image(target_paths[target_id], CONFIG["target_camera"], dark_frames)),
                )]
                order = [f"original {CONFIG['target_camera']}"]
                for camera in CONFIG["source_cameras"]:
                    if camera in available:
                        panels.append((f"reconstructed from {camera}", image_array(available[camera])))
                        order.append(f"reconstructed from {camera}")

                side_by_side = output_dir / f"{target_id}_{CONFIG['side_by_side_suffix']}.png"
                save_side_by_side(side_by_side, panels)
                print(f"final image: {side_by_side}", flush=True)
                print("order: " + " | ".join(order), flush=True)

                for path in available.values():
                    remove_file(path)
                del temp_recons[target_id]

        # Release large memory before moving to the next source camera.
        del source_cache, target_cache, gram

    for target_id, recons in temp_recons.items():
        print(
            f"skip final image for {target_id}: only {len(recons)} reconstruction(s) available",
            flush=True,
        )
        for path in recons.values():
            remove_file(path)
    if tmp_dir.exists() and not any(tmp_dir.iterdir()):
        tmp_dir.rmdir()


if __name__ == "__main__":
    main()
