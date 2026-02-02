#!/usr/bin/env python3
import os
import sys
import subprocess
import multiprocessing
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Path to deepstream-app executable (assumed in PATH inside container)
DEEPSTREAM_APP = "deepstream-app"

# Template config file (your existing deepstream_app_config.txt)
TEMPLATE_CONFIG_FILE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../deepstream_app_config.txt")
)

# Base directory for resolving relative paths in config files
BASE_DIR = os.path.dirname(TEMPLATE_CONFIG_FILE)

# per-config generated dir 
RUN_CONFIGS_DIR = None
LOGS_DIR = None

def setup_logging(log_dir: str, process_name: str) -> logging.Logger:
    """File-only logging (no console output)."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{process_name}.log")

    logger = logging.getLogger(process_name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # Important: avoid duplicate handlers if the logger is reused
    if logger.handlers:
        for h in list(logger.handlers):
            logger.removeHandler(h)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def get_video_files(folder: str):
    exts = (".mp4", ".avi", ".mov", ".mkv")
    p = Path(folder)
    # print(f"Scanning for video files in {p.resolve()}")
    if not p.exists():
        return []
    files = [str(x) for x in p.iterdir() if x.is_file() and x.suffix.lower() in exts]
    files.sort()
    return files


def _sanitize_filename(s: str) -> str:
    # keep it simple: replace weird chars
    out = []
    for ch in s:
        if ch.isalnum() or ch in ("-", "_", ".", "+"):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)


def create_config_for_video(video_path: str, batch_size: int, out_dir: str) -> str:
    """
    Generate a per-video DeepStream app config that only overrides [source0] uri.
    Critically: we do NOT rewrite nvinfer config paths or copy any artifacts.
    Relative paths are expected to work when running with cwd=BASE_DIR.
    """
    os.makedirs(out_dir, exist_ok=True)

    with open(TEMPLATE_CONFIG_FILE, "r") as f:
        config_lines = f.readlines()

    video_uri = Path(video_path).resolve().as_uri()  # handles spaces safely

    modified_lines = []
    in_source0 = False
    uri_replaced = False
    config_files_found = []  # Track all config-file references

    for line in config_lines:
        stripped = line.strip()

        if stripped.startswith("[source0]"):
            in_source0 = True
            modified_lines.append(line)
            continue

        # leaving [source0] when next section begins
        if stripped.startswith("[") and in_source0 and not stripped.startswith("[source0]"):
            in_source0 = False
            modified_lines.append(line)
            continue

        if in_source0 and stripped.startswith("uri="):
            modified_lines.append(f"uri={video_uri}\n")
            uri_replaced = True
        elif stripped.startswith("config-file="):
            # Convert relative config-file paths to absolute paths
            config_value = stripped.split("=", 1)[1]
            if not os.path.isabs(config_value):
                abs_config_path = os.path.join(BASE_DIR, config_value)
                modified_lines.append(f"config-file={abs_config_path}\n")
                config_files_found.append({"original": config_value, "absolute": abs_config_path})
            else:
                modified_lines.append(line)
                config_files_found.append({"original": config_value, "absolute": config_value})
        else:
            modified_lines.append(line)

    if not uri_replaced:
        # If template doesn't have uri= in [source0], add it at the end of [source0]
        # This is conservative; most templates already include uri=
        out = []
        in_source0 = False
        inserted = False
        for line in modified_lines:
            out.append(line)
            if line.strip().startswith("[source0]"):
                in_source0 = True
                continue
            if in_source0 and line.strip().startswith("[") and not inserted:
                # insert before next section
                out.insert(len(out) - 1, f"uri={video_uri}\n")
                inserted = True
                in_source0 = False
        if not inserted:
            # if [source0] is last section
            out.append(f"uri={video_uri}\n")
        modified_lines = out

    video_basename = _sanitize_filename(Path(video_path).stem)
    cfg_path = os.path.join(out_dir, f"deepstream_config_{video_basename}.txt")
    with open(cfg_path, "w") as f:
        f.writelines(modified_lines)
    
    # Update batch size in the main config file
    subprocess.run(
        ["sed", "-i", f"s/^batch-size=.*/batch-size={batch_size}/", cfg_path],
        check=True,
    )
    # Update batch size in any referenced config files
    for cf in config_files_found:
        abs_path = cf["absolute"]
        # Update batch-size
        subprocess.run(
            ["sed", "-i", f"s/^batch-size=.*/batch-size={batch_size}/", abs_path],
            check=True,
        )
        # Update onnx-file with correct batch size: firedetect-11s-bX.onnx
        subprocess.run(
            ["sed", "-i", f"s/\\(onnx-file=.*b\\)[0-9]\\+\\(\\.onnx\\)/\\1{batch_size}\\2/", abs_path],
            check=True,
        )
        # Update model-engine-file with correct batch size: model_bX_gpu0_fp32.engine
        subprocess.run(
            ["sed", "-i", f"s/\\(model-engine-file=.*_b\\)[0-9]\\+\\(_gpu0_fp32\\.engine\\)/\\1{batch_size}\\2/", abs_path],
            check=True,
        )
            
    # Print config files found
    print(f"\n=== Config files for {video_basename} ===")
    if config_files_found:
        for cf in config_files_found:
            exists = "✓" if os.path.exists(cf["absolute"]) else "✗"
            print(f"  {exists} {cf['original']} -> {cf['absolute']}")
    else:
        print("  No config-file references found")
    print("="*50)

    return cfg_path


def run_pipeline(video_path: str, batch_size: int, logger: logging.Logger, process_id: int, logs_dir: str, run_configs_dir: str):
    """
    Run deepstream-app for one video.
    - No /tmp usage
    - cwd=BASE_DIR so relative paths in configs work (your requirement)
    - deepstream stdout/stderr redirected to log files only
    """
    video_name = os.path.basename(video_path)
    video_basename = _sanitize_filename(Path(video_path).stem)

    # Per-process config dir (repo-local)
    proc_cfg_dir = os.path.join(run_configs_dir, f"process_{process_id}")
    cfg_file = create_config_for_video(video_path, batch_size, proc_cfg_dir)
    print(f"====== Generated config for {video_name}: {cfg_file}")

    # Per-video deepstream output file (repo-local logs)
    ds_out_dir = os.path.join(logs_dir, f"process_{process_id}_deepstream")
    os.makedirs(ds_out_dir, exist_ok=True)
    ds_log_path = os.path.join(ds_out_dir, f"{video_basename}.deepstream.log")

    cmd = [DEEPSTREAM_APP, "-c", cfg_file]
    logger.info(f"Starting pipeline for {video_name}")
    logger.info(f"Config: {cfg_file}")
    logger.info(f"DeepStream output: {ds_log_path}")
    logger.info(f"cwd (for relative paths): {BASE_DIR}")
    logger.info(f"Command: {' '.join(cmd)}")

    # Redirect DeepStream output to file only (no terminal)
    with open(ds_log_path, "ab") as ds_log:
        # Put a separator so multiple runs are readable
        ts = datetime.now().isoformat(timespec="seconds")
        ds_log.write(f"\n\n===== {ts} START {video_name} =====\n".encode("utf-8"))

        try:
            result = subprocess.run(
                cmd,
                stdout=ds_log,
                stderr=ds_log,
                cwd=BASE_DIR,  # <-- critical fix: relative paths resolve from repo/config root
                check=False,
            )
        except Exception:
            logger.exception(f"Exception occurred while running DeepStream for {video_name}")
            ds_log.write(f"\n===== {ts} EXCEPTION {video_name} =====\n".encode("utf-8"))
            return

        ds_log.write(f"\n===== {ts} END {video_name} rc={result.returncode} =====\n".encode("utf-8"))

    if result.returncode != 0:
        logger.error(f"Pipeline failed for {video_name} (exit code {result.returncode})")
    else:
        logger.info(f"Pipeline finished successfully for {video_name}")


def worker(videos, batch_size, shared_index, lock, log_dir, run_configs_dir, process_id):
    process_name = f"process_{process_id}"
    logger = setup_logging(log_dir, process_name)
    logger.info(f"Worker process {process_id} started")

    while True:
        with lock:
            idx = shared_index.value
            shared_index.value += 1

        if idx >= len(videos):
            logger.info(f"Worker process {process_id} finished - no more videos")
            break

        video = videos[idx]
        run_pipeline(video, batch_size, logger, process_id, log_dir, run_configs_dir)


def main():
    multiprocessing.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(
        description="Run DeepStream YOLO pipeline on multiple videos in parallel (no /tmp, file-only logs)"
    )
    parser.add_argument("--video_folder", type=str, default="../../datasets/Boreal-Forest-Fire/Boreal-Forest-Fire-Subset-B/Evo-Videos", help="Path to folder containing video files")
    parser.add_argument("--num-processes", type=int, default=1, help="Number of parallel processes")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for inference (if applicable)")
    parser.add_argument("--max-videos", type=int, default=None, help="Maximum number of videos to process")
    parser.add_argument("--inference-log", type=str, default=None, help="Absolute path to save inference log")

    args = parser.parse_args()
    folder = args.video_folder
    num_processes = args.num_processes
    max_videos = args.max_videos
    batch_size = args.batch_size
    inference_log = args.inference_log
    
    # Logs + per-video generated configs (repo-local; no /tmp usage)
    global LOGS_DIR, RUN_CONFIGS_DIR
    LOGS_DIR = os.path.join(BASE_DIR, "logs") if inference_log is None else inference_log
    RUN_CONFIGS_DIR = os.path.join(BASE_DIR, "run_configs")
    
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(RUN_CONFIGS_DIR, exist_ok=True)

    main_logger = setup_logging(LOGS_DIR, "main")
    main_logger.info(f"BASE_DIR={BASE_DIR}")
    main_logger.info(f"TEMPLATE_CONFIG_FILE={TEMPLATE_CONFIG_FILE}")
    main_logger.info(f"LOGS_DIR={LOGS_DIR}")
    main_logger.info(f"RUN_CONFIGS_DIR={RUN_CONFIGS_DIR}")

    videos = get_video_files(folder)
    if not videos:
        main_logger.error(f"Current pwd: {os.getcwd()}")
        main_logger.error(f"No video files found in {folder}")
        sys.exit(1)

    if max_videos is not None and max_videos > 0:
        videos = videos[:max_videos]

    main_logger.info(f"Found {len(videos)} video(s) to process from {folder}")
    main_logger.info(f"Using {num_processes} parallel process(es)")

    shared_index = multiprocessing.Value("i", 0)
    lock = multiprocessing.Lock()

    processes = []
    for i in range(num_processes):
        p = multiprocessing.Process(
            target=worker, args=(videos, batch_size, shared_index, lock, LOGS_DIR, RUN_CONFIGS_DIR, i)
        )
        p.start()
        processes.append(p)
        main_logger.info(f"Started worker process {i} (pid={p.pid})")

    for p in processes:
        p.join()

    main_logger.info("All videos processed.")


if __name__ == "__main__":
    main()
