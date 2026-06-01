"""Add REPRODUCIBLE flag + set_seed() to the notebook config cell."""
import json

nb = json.load(open("models/notebook/patchtst-new-branch-test-sector.ipynb", encoding="utf-8"))

# ── 1. Inject REPRODUCIBLE + SEED into the config cell ───────────────────────
SEED_CONFIG = (
    "\n# ---- Reproducibility ----\n"
    "REPRODUCIBLE = True  # True = fixed seed every run; False = random init\n"
    "SEED = 42\n"
)

for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    if "RUN_GLOBAL_TRAINING" in src and "GLOBAL_NUM_EPOCHS" in src:
        # Append seed config lines right before label_config = LabelConfig(...)
        new_lines = []
        for line in cell["source"]:
            if line.startswith("label_config = LabelConfig("):
                new_lines.append(SEED_CONFIG)
            new_lines.append(line)
        cell["source"] = new_lines
        print("Injected REPRODUCIBLE + SEED into config cell")
        break

# ── 2. Inject set_seed() function + conditional call after imports ────────────
SEED_CELL_SOURCE = [
    "import random\n",
    "\n",
    "def set_seed(seed: int) -> None:\n",
    "    \"\"\"Fix all random sources for a fully reproducible training run.\"\"\"\n",
    "    random.seed(seed)\n",
    "    np.random.seed(seed)\n",
    "    torch.manual_seed(seed)\n",
    "    torch.cuda.manual_seed_all(seed)\n",
    "    # deterministic=True disables non-deterministic cuDNN kernels.\n",
    "    # benchmark=False stops cuDNN from auto-selecting faster non-deterministic algos.\n",
    "    torch.backends.cudnn.deterministic = True\n",
    "    torch.backends.cudnn.benchmark = False\n",
    "\n",
    "if REPRODUCIBLE:\n",
    "    set_seed(SEED)\n",
    "    print(f'Reproducibility ON  — seed={SEED}')\n",
    "else:\n",
    "    print('Reproducibility OFF — random init each run')\n",
]

seed_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": SEED_CELL_SOURCE,
}

# Insert the new cell right after the config cell
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    if "RUN_GLOBAL_TRAINING" in src and "GLOBAL_NUM_EPOCHS" in src:
        nb["cells"].insert(i + 1, seed_cell)
        print(f"Inserted set_seed cell at position {i + 1}")
        break

with open("models/notebook/patchtst-new-branch-test-sector.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=True)

print("Saved (ASCII-safe).")
