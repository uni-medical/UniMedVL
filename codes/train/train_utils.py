# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import logging
import os


def create_logger(logging_dir, rank, filename="log", script_name=None):
    """
    Create a logger that writes to a log file and stdout.
    
    Args:
        logging_dir: Directory to save log files
        rank: Process rank (only rank 0 creates real logger)  
        filename: Base filename for log (default: "log")
        script_name: Script name to include in filename (auto-detected if None)
    """
    if rank == 0 and logging_dir is not None:  # real logger
        # Auto-detect script name if not provided
        if script_name is None:
            import inspect
            import os
            frame = inspect.currentframe()
            try:
                # Go up the call stack to find the calling script
                caller_frame = frame.f_back.f_back if frame.f_back else frame
                script_path = caller_frame.f_globals.get('__file__', 'unknown')
                script_name = os.path.splitext(os.path.basename(script_path))[0]
            finally:
                del frame  # Avoid reference cycles
        
        # Create timestamp for unique log files
        import time
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        log_filename = f"{script_name}_{timestamp}_{filename}.log"
        
        # Ensure logging directory exists
        os.makedirs(logging_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[
                logging.StreamHandler(), 
                logging.FileHandler(f"{logging_dir}/{log_filename}")
            ]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def get_latest_ckpt(checkpoint_dir):
    step_dirs = [d for d in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, d))]
    # Remove print statements to avoid excessive duplicate output in distributed training
    # print("---> Checkpoint_dir", checkpoint_dir)
    # print(f"---> Current steps avaliable in the directory are {step_dirs}.")
    if not step_dirs:
        return None

    numeric_dirs = []
    for d in step_dirs:
        try:
            int(d)
            numeric_dirs.append(d)
        except ValueError:
            continue

    if not numeric_dirs:
        return None

    # Sort directories by step number in descending order
    numeric_dirs = sorted(numeric_dirs, key=lambda x: int(x), reverse=True)

    # Check for completeness of checkpoints, starting from the latest
    for step_dir_name in numeric_dirs:
        step_dir_path = os.path.join(checkpoint_dir, step_dir_name)
        
        # Base files that should exist, based on user-provided image.
        base_required_files = [
            "model.safetensors",
            "ema.safetensors",
            "scheduler.pt",
            "data_status.pt"
        ]

        # Check for base files.
        if not all(os.path.exists(os.path.join(step_dir_path, f)) for f in base_required_files):
            continue

        all_files_in_step = os.listdir(step_dir_path)
        
        # Check for optimizer state, which can be sharded or not.
        if "optimizer.pt" in all_files_in_step:
            # Non-sharded case, checkpoint is complete.
            return step_dir_path

        # Sharded optimizer case.
        optimizer_files = [f for f in all_files_in_step if f.startswith("optimizer.") and f.endswith(".pt")]
        if not optimizer_files:
            # No optimizer files at all, incomplete.
            continue
        
        try:
            # e.g., optimizer.00001-of-00008.pt -> 8
            num_shards_str = optimizer_files[0].split('-of-')[1].split('.pt')[0]
            num_shards = int(num_shards_str)
        except (IndexError, ValueError):
            # Malformed optimizer file name, skip this directory.
            continue
        
        if len(optimizer_files) == num_shards:
            # Found the correct number of optimizer shards.
            return step_dir_path

    # If no complete checkpoint is found after checking all directories.
    return None

