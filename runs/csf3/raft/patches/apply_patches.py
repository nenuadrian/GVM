#!/usr/bin/env python
"""Patch the LMFlow v0.0.6 checkout for this replication.

Idempotent: every edit is skipped if its result is already present, and every
edit fails loudly if its anchor is missing, so a different LMFlow revision
cannot be silently half-patched.

Four changes, none of which touch the RAFT algorithm:

1. raft_aligner.py logs the reward curve to wandb. Upstream appends it to
   self.reward_seq / self.train_reawrd and renders a matplotlib PNG; the numbers
   never leave the node. reward_seq is *the* RAFT curve (Figure 4 of the paper),
   so it has to be in wandb.
2. raft_aligner.py records generation vs training wall clock per iteration.
3. raft_align.py scores a whole best-of-K group in one reward-model forward pass
   instead of K sequential ones (batch_size was hardcoded to 1), loads the 3.2B
   reward model in bf16 rather than fp32, and truncates to the reward model's
   2048-token window.
4. raft_align.py tolerates local_rank == -1, so the pipeline can be run on one
   GPU without a distributed launcher.
5. raft_aligner.py writes the final model from rank 0 only. Upstream calls
   wrapped_model.save() on every rank against one directory.
"""

import sys
from pathlib import Path

ALIGNER = "src/lmflow/pipeline/raft_aligner.py"
ALIGN_EX = "examples/raft_align.py"


def edit(path: Path, anchor: str, addition: str, marker: str, *, before=False):
    text = path.read_text()
    if marker in text:
        print(f"  [skip] {path.name}: already has {marker!r}")
        return
    if anchor not in text:
        sys.exit(
            f"ERROR: anchor not found in {path}\n"
            f"       expected:\n{anchor}\n"
            f"       This patch targets LMFlow v0.0.6; check out that tag."
        )
    if text.count(anchor) != 1:
        sys.exit(f"ERROR: anchor is ambiguous in {path} ({text.count(anchor)} matches)")
    new = addition + anchor if before else anchor + addition
    path.write_text(text.replace(anchor, new))
    print(f"  [ok]   {path.name}: {marker}")


def main(root: Path):
    aligner, align_ex = root / ALIGNER, root / ALIGN_EX
    for p in (aligner, align_ex):
        if not p.is_file():
            sys.exit(f"ERROR: {p} does not exist -- is {root} an LMFlow checkout?")

    print("patching", root)

    # --- 1. wandb init, alongside the existing reward bookkeeping ----------
    edit(
        aligner,
        anchor=(
            "        self.store_dir = aligner_args.output_dir\n"
            "        self.reward_seq = []\n"
            "        self.train_reawrd = []\n"
        ),
        addition=(
            "        # [csf3] rank 0 owns the wandb run; other ranks stay silent.\n"
            "        self._wandb = None\n"
            "        if training_args.local_rank in (-1, 0):\n"
            "            try:\n"
            "                import wandb as _wandb\n"
            "                if _wandb.run is None:\n"
            "                    _wandb.init(\n"
            "                        project=os.environ.get('WANDB_PROJECT', 'raft-lmflow'),\n"
            "                        name=os.environ.get('WANDB_NAME'),\n"
            "                        id=os.environ.get('WANDB_RUN_ID'),\n"
            "                        resume=os.environ.get('WANDB_RESUME', 'allow'),\n"
            "                        config={\n"
            "                            'num_raft_iteration': ITERATION,\n"
            "                            'raft_batch_size': sft_batch_size,\n"
            "                            'collection_strategy': collection_strategy,\n"
            "                            'top_reward_percentage': aligner_args.top_reward_percentage,\n"
            "                            'K': K if collection_strategy == 'local' else None,\n"
            "                            'prompts_per_rank_per_iter': M,\n"
            "                            'world_size': world_size,\n"
            "                            'learning_rate': training_args.learning_rate,\n"
            "                            'num_train_epochs': training_args.num_train_epochs,\n"
            "                            'output_min_length': aligner_args.output_min_length,\n"
            "                            'output_max_length': aligner_args.output_max_length,\n"
            "                            'model_name_or_path': model_args.model_name_or_path,\n"
            "                        },\n"
            "                    )\n"
            "                self._wandb = _wandb\n"
            "            except Exception as _e:\n"
            "                logger.warning('wandb disabled: %s', _e)\n"
            "        _raft_gen_seconds = float('nan')\n"
        ),
        marker="[csf3] rank 0 owns the wandb run",
    )

    # --- 2. stash generation wall clock before start_time is reused --------
    edit(
        aligner,
        anchor='            logger.info("It takes %.2f s to inference one stage", end_time - start_time)\n',
        addition="            _raft_gen_seconds = end_time - start_time  # [csf3]\n",
        marker="_raft_gen_seconds = end_time - start_time",
    )

    # --- 3. log the iteration to wandb ------------------------------------
    # Keyed on `iteration`, which is monotonic. The inner RaftTrainer is left on
    # report_to=none: it calls train() once per RAFT iteration, so its
    # global_step restarts at 0 every time and wandb drops the rewind.
    edit(
        aligner,
        anchor='            logger.info("It takes %.2f s to train one stage", end_time - start_time)\n',
        addition=(
            "            # [csf3] one wandb point per RAFT iteration.\n"
            "            if self._wandb is not None:\n"
            "                _m = getattr(train_result, 'metrics', {}) or {}\n"
            "                _row = {\n"
            "                    'raft/iteration': iteration,\n"
            "                    'raft/mean_reward': self.reward_seq[-1],\n"
            "                    'raft/mean_reward_selected': self.train_reawrd[-1],\n"
            "                    'raft/reward_gain': self.train_reawrd[-1] - self.reward_seq[-1],\n"
            "                    # selected_dataset holds the best-of-K winners; the\n"
            "                    # trainer's dataset is those texts concatenated and\n"
            "                    # re-chunked into block_size blocks, so they differ.\n"
            "                    'raft/n_selected': len(selected_dataset['train']),\n"
            "                    'raft/train_blocks': len(raft_trainer.train_dataset),\n"
            "                    'raft/prompts_seen': (iteration + 1) * M * world_size,\n"
            "                    'raft/samples_drawn': (iteration + 1) * M * world_size *\n"
            "                                         (K if collection_strategy == 'local' else 1),\n"
            "                    'raft/gen_seconds': _raft_gen_seconds,\n"
            "                    'raft/train_seconds': end_time - start_time,\n"
            "                    'raft/train_loss': _m.get('train_loss'),\n"
            "                }\n"
            "                self._wandb.log({k: v for k, v in _row.items() if v is not None},\n"
            "                                step=iteration)\n"
        ),
        marker="[csf3] one wandb point per RAFT iteration",
    )

    # --- 4. reward pipeline: bf16, batched, truncating, rank-safe ---------
    edit(
        align_ex,
        anchor="from transformers import HfArgumentParser, pipeline, AutoTokenizer\n",
        addition="import torch  # [csf3]\n",
        marker="import torch  # [csf3]",
    )
    # The next two are replacements rather than appends, so they bypass edit().
    text = align_ex.read_text()
    old_pipe = (
        "        hf_pipe = pipeline(\n"
        "            reward_args.reward_task,\n"
        "            model=reward_args.reward_model_or_path,\n"
        '            device=f"cuda:{pipeline_args.local_rank}",\n'
        "            tokenizer=rm_tokenizer\n"
        "        )\n"
    )
    new_pipe = (
        "        # [csf3] reward pipeline: local_rank is -1 without a distributed\n"
        "        # launcher, which would ask for cuda:-1. bf16 halves the 3.2B\n"
        "        # reward model's footprint; it was trained in bf16 anyway.\n"
        "        _rm_device = max(pipeline_args.local_rank, 0)\n"
        "        hf_pipe = pipeline(\n"
        "            reward_args.reward_task,\n"
        "            model=reward_args.reward_model_or_path,\n"
        '            device=f"cuda:{_rm_device}",\n'
        "            tokenizer=rm_tokenizer,\n"
        "            torch_dtype=torch.bfloat16,\n"
        "        )\n"
    )
    if "[csf3] reward pipeline" in text:
        print("  [skip] raft_align.py: already has reward pipeline patch")
    elif old_pipe not in text:
        sys.exit(f"ERROR: reward pipeline anchor not found in {align_ex}")
    else:
        text = text.replace(old_pipe, new_pipe)
        align_ex.write_text(text)
        print("  [ok]   raft_align.py: reward pipeline (bf16, rank-safe)")

    # --- 5. the final save is unguarded: every rank writes the same files ---
    # transformers 4.34's save_pretrained torch.saves straight to the target
    # path, so four ranks racing on one directory can leave a truncated
    # pytorch_model.bin. Rank 0 writes; the others wait at a barrier.
    text = aligner.read_text()
    old_save = (
        "        if aligner_args.output_dir is not None:\n"
        "            wrapped_model.save(aligner_args.output_dir)\n"
    )
    new_save = (
        "        if aligner_args.output_dir is not None:\n"
        "            # [csf3] rank 0 writes, everyone else waits.\n"
        "            if training_args.local_rank in (-1, 0):\n"
        "                wrapped_model.save(aligner_args.output_dir)\n"
        "            if dist.is_available() and dist.is_initialized():\n"
        "                dist.barrier()\n"
    )
    if "[csf3] rank 0 writes, everyone else waits" in text:
        print("  [skip] raft_aligner.py: already has guarded save")
    elif old_save not in text:
        sys.exit(f"ERROR: final-save anchor not found in {aligner}")
    else:
        aligner.write_text(text.replace(old_save, new_save))
        print("  [ok]   raft_aligner.py: final save guarded to rank 0")

    text = align_ex.read_text()
    old_kw = (
        "            pipe_kwargs = {\n"
        '                "return_all_scores": True,\n'
        '                "function_to_apply": "none",\n'
        '                "batch_size": 1\n'
        "            }\n"
    )
    new_kw = (
        "            # [csf3] score the whole best-of-K group in one forward pass;\n"
        "            # upstream pinned batch_size to 1. Truncate to the reward\n"
        "            # model's 2048-token window rather than overrun it.\n"
        "            pipe_kwargs = {\n"
        '                "return_all_scores": True,\n'
        '                "function_to_apply": "none",\n'
        '                "batch_size": int(os.environ.get("RAFT_RM_BATCH_SIZE", "8")),\n'
        '                "truncation": True,\n'
        '                "max_length": 2048,\n'
        "            }\n"
    )
    if "[csf3] score the whole best-of-K group" in text:
        print("  [skip] raft_align.py: already has pipe_kwargs patch")
    elif old_kw not in text:
        sys.exit(f"ERROR: pipe_kwargs anchor not found in {align_ex}")
    else:
        align_ex.write_text(text.replace(old_kw, new_kw))
        print("  [ok]   raft_align.py: batched + truncating reward scoring")

    print("PATCH_OK")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "."))
