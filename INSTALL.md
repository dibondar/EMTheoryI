# Installing the software

The examples in this course use **FEniCSx** (Python module `dolfinx`, version 0.11) for the
finite element method and **Gmsh** for the meshes. Plots need **matplotlib**, and physical
units are handled by **pint**. Everything is installed with conda, which takes about 10 minutes
and 2 GB of disk space.

## 1. What you need

- A terminal (macOS: Terminal.app, Linux: any terminal, Windows: see the platform note below).
- **Miniconda**: download the installer for your system from
  <https://www.anaconda.com/download/success> (scroll to "Miniconda installers") and run it.
  After installing, close and reopen the terminal, then check that `conda --version` prints a version.
- **Visual Studio Code** (recommended editor, free): <https://code.visualstudio.com/>.
  You can edit and run the scripts in any editor, but the course instructions assume VS Code.
  Section 4 explains how to connect it to the FEniCSx environment.

### Platform notes

| System | How |
|---|---|
| macOS (Intel or Apple Silicon) | conda, as below |
| Linux | conda, as below |
| Windows | conda, as below, typed into the **Anaconda Prompt** that Miniconda installs. If the installation or the check in section 3 fails, install WSL2 with Ubuntu (<https://learn.microsoft.com/windows/wsl/install>), open the Ubuntu terminal and follow the Linux steps inside it |

## 2. Install FEniCSx

**Option A (recommended).** In the terminal, go to the folder with the course files and run

```bash
conda env create -f environment.yml
```

This creates an environment named `fenicsx` with DOLFINx, Gmsh, numpy, matplotlib, pillow and pint.

**Option B (by hand).** The same, spelled out:

```bash
conda create -n fenicsx -c conda-forge python=3.12 fenics-dolfinx=0.11 python-gmsh numpy matplotlib pillow pint
```

**Already have a `fenicsx` environment with a package missing?** Add it, for example:

```bash
conda install -n fenicsx -c conda-forge python-gmsh matplotlib pillow pint
```

## 3. Activate and check

Every time you open a new terminal, activate the environment first:

```bash
conda activate fenicsx
```

Then check that everything imports:

```bash
python -c "import dolfinx, gmsh, matplotlib, pint; print('DOLFINx', dolfinx.__version__)"
```

Expected output: `DOLFINx 0.11.0`.

## 4. Set up VS Code

VS Code gives you syntax colouring, a built-in terminal and a "Run" button. It has to be told
to use the `fenicsx` environment; otherwise it runs the scripts with a Python that has no DOLFINx.

### 4.1 Install the Python extension

1. Open VS Code.
2. Open the Extensions view (the four-squares icon on the left, or `Ctrl+Shift+X`; on macOS `Cmd+Shift+X`).
3. Search for **Python** and install the one published by **Microsoft**.

**Windows with WSL only.** Install VS Code on Windows itself, not inside Ubuntu. Then also install the
extension **WSL** (published by Microsoft). In the Ubuntu terminal, go to the course folder and type
`code .`: VS Code opens with "WSL: Ubuntu" in the green box in the bottom-left corner. Install the
Python extension once more in this window when VS Code asks (it has to be installed "in WSL").

### 4.2 Open the course folder

Use **File → Open Folder…** and choose the folder with the course files
(the one that contains `environment.yml`). Always open the *folder*, not a single `.py` file,
so that VS Code remembers your settings for it.

### 4.3 Select the environment (the important step)

1. Open a `.py` file of the course, e.g. `wave_current_dolfinx.py`.
2. Press `Ctrl+Shift+P` (macOS: `Cmd+Shift+P`) to open the Command Palette. The search box at the
   top now starts with `>`; without the `>` it searches file names, not commands.
3. Type **Python: Select Interpreter** and press Enter. Newer versions of the Python extension
   may show it as **Python: Set Project Environment** instead; use that one.
4. Pick the entry that mentions **fenicsx**, for example
   `Python 3.12.x ('fenicsx')`.

Shortcut: with a `.py` file open, click the Python version in the status bar at the bottom right;
it opens the same list. The status bar then shows the selected environment.
You only do this once per folder; VS Code remembers it.

**Neither command appears?** The Python extension is missing or disabled: check section 4.1.

**`fenicsx` is not in the list?** Tell VS Code where it is:

1. In a terminal, run `conda env list`. It prints the path of each environment, e.g.
   `fenicsx   /Users/yourname/miniconda3/envs/fenicsx`.
2. In **Python: Select Interpreter**, choose **Enter interpreter path…** and type that path
   followed by `/bin/python`, e.g. `/Users/yourname/miniconda3/envs/fenicsx/bin/python`.

### 4.4 Check the built-in terminal

Open a new terminal inside VS Code with **Terminal → New Terminal** (or ``Ctrl+` ``).
VS Code activates the selected environment automatically, so the prompt should start with
`(fenicsx)`. If it does not, type `conda activate fenicsx` in it.
Then run the check from section 3 in this terminal:

```bash
python -c "import dolfinx, gmsh, matplotlib, pint; print('DOLFINx', dolfinx.__version__)"
```

Terminals that were already open before you selected the environment are not updated:
close them (trash-can icon) and open a new one.

## 5. Run an example

**In VS Code:** open `wave_current_dolfinx.py` and click the ▷ ("Run Python File") button
in the top-right corner of the editor. The output appears in the built-in terminal.

**In a terminal** (inside VS Code or not), with the environment activated:

```bash
python wave_current_dolfinx.py
```

Either way, it prints progress and writes everything to the folder `results/`:

| File | What it is | Open with |
|---|---|---|
| `results/wave.gif` | animation of the solution | any image viewer or browser |
| `results/frames/frame_XXX.png` | the individual frames | any image viewer |
| `results/wave.pvd` | the full time series of the solution | ParaView |
| `results/faces.pvd` | the mesh with the two regions (vacuum / wire) | ParaView |
| `results/B_final.pvd` | magnetic field at the final time | ParaView |

The second example, `wave_square_wire_dolfinx.py`, runs the same way and writes to `results_square/`.

The default run takes about half a minute on a laptop, most of it for drawing the frames.
The first run is a little slower because DOLFINx compiles some code and caches it; lines
starting with `ld: warning` that may appear during this step are harmless.

**ParaView** (optional, free) reads the `.pvd` files: <https://www.paraview.org/download/>.
Open the `.pvd` file, click "Apply", and use the play button to step through time.

## 6. Troubleshooting

| Symptom | Fix |
|---|---|
| `conda: command not found` | Close and reopen the terminal after installing Miniconda. |
| `ModuleNotFoundError: No module named 'dolfinx'` or `'gmsh'` | The script was run with another Python. Run `conda activate fenicsx` first; in VS Code select the `fenicsx` interpreter (section 4). The path of the Python that runs the script must contain `envs/fenicsx`. |
| `ModuleNotFoundError: No module named 'matplotlib'`, `'PIL'` or `'pint'`, or `'gmsh'` inside `fenicsx` | See the end of section 2. |
| `pint.errors.DimensionalityError` | A parameter has the wrong kind of unit, for example a frequency where a length is expected. The message names the two units that do not match. |
| The run is too slow | Increase `h_target` and `h_wire`, lower `n_frames`, or set `surface_plot = False` at the top of the script. |
| An error mentions `ffcx`, `cffi` or a compiler | Delete the cache of compiled code with `rm -rf ~/.cache/fenics` and run again. |
| VS Code: `No module named 'dolfinx'` when clicking ▷ | The wrong interpreter is selected. Redo section 4.3 and check the status bar says `fenicsx`. |
| VS Code: ▷ fails but the terminal works | Close the old VS Code terminals and open a new one (section 4.4), then click ▷ again. |
| VS Code: the `code` command is not found (macOS) | In VS Code, open the Command Palette and run **Shell Command: Install 'code' command in PATH**. |
| `conda env create` fails to solve | Make sure you typed `-c conda-forge`; the packages exist only on that channel. |
