# Notes

## Mac <-> devserver workflow

Code is written on the Mac and executed on `devserver` (Linux box with the
NVIDIA GTX 1070 - the Mac is Apple Silicon and has no CUDA GPU).

Everything below runs from the Mac terminal, from this project folder:

```bash
cd ~/zhukeke_github/LLM/LLM_From_Scratch
```

### Run any file on the server

```bash
./scripts/sync.sh run Ch01.py
```

This pushes local changes to devserver, then runs `uv run Ch01.py` there
(inside the project's uv-managed venv, with GPU-enabled PyTorch). Output
streams back to the Mac terminal.

Pass arguments to the script after its name - they go straight to the
script, not to `sync.sh`:

```bash
./scripts/sync.sh run train.py --epochs 5 --batch-size 32
```

### Other sync commands

```bash
./scripts/sync.sh push      # copy local changes -> devserver
./scripts/sync.sh pull      # copy devserver changes -> local
./scripts/sync.sh status    # dry-run preview of both directions, no transfer
./scripts/sync.sh check     # verify SSH + rsync work on both ends
./scripts/sync.sh shell     # open an interactive SSH shell, cd'd into the project
```

`.sync.conf` holds the target host/path; `.syncignore` lists what never
syncs (`.venv/`, `.git/`, checkpoints, data, model weights - those stay
local to whichever machine produced them).

### Manual equivalent (what `run` does under the hood)

```bash
./scripts/sync.sh push -q
ssh devserver 'export PATH="$HOME/.local/bin:$PATH"; cd ~/AI_Projects/LLM_From_Scratch && uv run Ch01.py'
```

### GPU gotcha

devserver's GTX 1070 is Pascal (compute capability 6.1). `uv`'s
`--torch-backend=auto` and the newest CUDA 13.x PyTorch wheels drop kernels
for CC < 7.5 - `torch.cuda.is_available()` still returns `True`, but every
real op crashes with `no kernel image is available for execution on the
device`. `pyproject.toml` pins the `cu126` PyTorch index on Linux, which
still ships `sm_61` kernels. Re-check this if the GPU or torch major
version ever changes.
