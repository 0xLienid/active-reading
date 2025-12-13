import os
import json
import time
import torch.distributed as dist
from pathlib import Path



def to_safe_model_name(model_name: str) -> str:
    """
    Convert a model name to a safe model name for saving to the Hugging Face Hub.
    """
    return model_name.replace("/", "-")


def gather_object_to_main(accelerator, local_obj, *, dst: int = 0):
    """
    Gather a (picklable) Python object from every rank onto the main process.

    Why this exists:
    - `accelerate.utils.gather_object(...)` / `accelerator.gather_object(...)` often performs an
      all-gather (every rank receives everything), which can be memory-heavy and can fail for
      large objects.
    - For these scripts we only need results on rank 0 to write/push a dataset.

    Returns:
    - On main process: a list of length `num_processes`, containing each rank's object.
    - On non-main processes: None
    """
    # Single-process: just return the object.
    if getattr(accelerator, "num_processes", 1) == 1:
        return [local_obj] if getattr(accelerator, "is_main_process", True) else None

    # Prefer torch.distributed.gather_object (gather-to-dst only, not all-gather).
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        try:
            gather_list = [None] * world_size if rank == dst else None
            dist.gather_object(local_obj, object_gather_list=gather_list, dst=dst)
            accelerator.wait_for_everyone()
            return gather_list if getattr(accelerator, "is_main_process", False) else None
        except Exception:
            # Fall back to file-based merge below.
            pass

    # Fallback: write per-rank shards and let rank 0 merge. This avoids collective object transfer.
    # Note: this requires a shared filesystem between ranks.
    base_dir = Path(os.environ.get("ACTIVE_READING_GATHER_DIR", "/tmp/active-reading-gather"))
    base_dir.mkdir(parents=True, exist_ok=True)

    # Shared run id so all ranks write to the same directory.
    run_id = os.environ.get("ACTIVE_READING_GATHER_RUN_ID")
    if run_id is None:
        if dist.is_available() and dist.is_initialized():
            # Create on dst and broadcast; this is small and should succeed even if the object gather failed.
            run_id_list = [f"{int(time.time_ns())}"] if dist.get_rank() == dst else [None]
            dist.broadcast_object_list(run_id_list, src=dst)
            run_id = run_id_list[0]
        else:
            # Single-node, best-effort fallback when torch.distributed isn't initialized for some reason.
            run_id = f"{os.getppid()}-{int(time.time())}"

    run_dir = base_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    rank = getattr(accelerator, "process_index", 0)
    world_size = getattr(accelerator, "num_processes", 1)

    shard_path = run_dir / f"rank{rank}.json"
    with shard_path.open("w", encoding="utf-8") as f:
        json.dump(local_obj, f)

    accelerator.wait_for_everyone()

    if getattr(accelerator, "is_main_process", False):
        gathered = []
        for r in range(world_size):
            p = run_dir / f"rank{r}.json"
            with p.open("r", encoding="utf-8") as f:
                gathered.append(json.load(f))
        return gathered

    return None
