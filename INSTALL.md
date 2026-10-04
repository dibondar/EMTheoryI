# Installing the software

The examples in this course use **FEniCS 2019.1** (the "legacy" FEniCS, Python module `dolfin`).
It is *not* the newer FEniCSx / `dolfinx`; the scripts will not run there.
Plots need **matplotlib**, and physical units are handled by **pint**. Everything is installed with conda, which takes about 10 minutes
and 2 GB of disk space.

## 1. What you need

- A terminal (macOS: Terminal.app, Linux: any terminal, Windows: see the platform note below).
- **Miniconda**: download the installer for your system from
  <https://www.anaconda.com/download/success> (scroll to "Miniconda installers") and run it.
  After installing, close and reopen the terminal, then check that `conda --version` prints a version.
- **Visual Studio Code** (recommended editor, free): <https://code.visualstudio.com/>.
  You can edit and run the scripts in any editor, but the course instructions assume VS Code.
  Section 4 explains how to connect it to the FEniCS environment.

### Platform notes

| System | How |
|---|---|
| macOS (Intel or Apple Silicon) | conda, as below |
| Linux | conda, as below |
| Windows | there is no Windows build of legacy FEniCS. Install WSL2 with Ubuntu (<https://learn.microsoft.com/windows/wsl/install>), open the Ubuntu terminal and follow the Linux steps inside it |

The mesh generator `mshr` that many FEniCS tutorials use is **not** required (it has no
build for Apple Silicon). The scripts build their meshes with functions that come with FEniCS itself.

## 2. Install FEniCS

**Option A (recommended).** In the terminal, go to the folder with the course files and run

```bash
conda env create -f environment.yml
```

This creates an environment named `fenicsproject` with FEniCS, numpy, matplotlib and pint.

**Option B (by hand).** The same, spelled out:

```bash
conda create -n fenicsproject -c conda-forge python=3.11 fenics=2019.1.0 numpy matplotlib pint
```

**Already have a `fenicsproject` environment without matplotlib or pint?** Add them:

```bash
conda install -n fenicsproject -c conda-forge matplotlib pint
```

## 3. Activate and check

Every time you open a new terminal, activate the environment first:

```bash
conda activate fenicsproject
```

(On old conda versions the command is `source activate fenicsproject`.)
Then check that everything imports:

```bash
python -c "import dolfin, matplotlib, pint; print('FEniCS', dolfin.__version__)"
```

Expected output: `FEniCS 2019.1.0`. The first import can take a minute because FEniCS compiles
some code on first use.

## 4. Set up VS Code

VS Code gives you syntax colouring, a built-in terminal and a "Run" button. It has to be told
to use the `fenicsproject` environment; otherwise it runs the scripts with a Python that has no FEniCS.

### 4.1 Install the Python extension

1. Open VS Code.
2. Open the Extensions view (the four-squares icon on the left, or `Ctrl+Shift+X`; on macOS `Cmd+Shift+X`).
3. Search for **Python** and install the one published by **Microsoft**.

**Windows (WSL) only.** Install VS Code on Windows itself, not inside Ubuntu. Then also install the
extension **WSL** (published by Microsoft). In the Ubuntu terminal, go to the course folder and type
`code .`: VS Code opens with "WSL: Ubuntu" in the green box in the bottom-left corner. Install the
Python extension once more in this window when VS Code asks (it has to be installed "in WSL").

### 4.2 Open the course folder

Use **File → Open Folder…** and choose the folder with the course files
(the one that contains `environment.yml`). Always open the *folder*, not a single `.py` file,
so that VS Code remembers your settings for it.

### 4.3 Select the environment (the important step)

1. Press `Ctrl+Shift+P` (macOS: `Cmd+Shift+P`) to open the Command Palette.
2. Type **Python: Select Interpreter** and press Enter.
3. Pick the entry that mentions **fenicsproject**, for example
   `Python 3.11.x ('fenicsproject')`.

The selected environment is shown in the status bar at the bottom right when a `.py` file is open.
You only do this once per folder; VS Code remembers it.

**`fenicsproject` is not in the list?** Tell VS Code where it is:

1. In a terminal, run `conda env list`. It prints the path of each environment, e.g.
   `fenicsproject   /Users/yourname/miniconda3/envs/fenicsproject`.
2. In **Python: Select Interpreter**, choose **Enter interpreter path…** and type that path
   followed by `/bin/python`, e.g. `/Users/yourname/miniconda3/envs/fenicsproject/bin/python`.

### 4.4 Check the built-in terminal

Open a new terminal inside VS Code with **Terminal → New Terminal** (or ``Ctrl+` ``).
VS Code activates the selected environment automatically, so the prompt should start with
`(fenicsproject)`. If it does not, type `conda activate fenicsproject` in it.
Then run the check from section 3 in this terminal:

```bash
python -c "import dolfin, matplotlib, pint; print('FEniCS', dolfin.__version__)"
```

Terminals that were already open before you selected the environment are not updated:
close them (trash-can icon) and open a new one.

## 5. Run an example

**In VS Code:** open `wave_current_fenics.py` and click the ▷ ("Run Python File") button
in the top-right corner of the editor. The output appears in the built-in terminal.

**In a terminal** (inside VS Code or not), with the environment activated:

```bash
python wave_current_fenics.py
```

Either way, it prints progress and writes everything to the folder `results/`:

| File | What it is | Open with |
|---|---|---|
| `results/wave.gif` | animation of the solution | any image viewer or browser |
| `results/frames/frame_XXX.png` | the individual frames | any image viewer |
| `results/wave.pvd` | the full time series of the solution | ParaView |
| `results/faces.pvd` | the mesh with the two regions (vacuum / wire) | ParaView |
| `results/B_final.pvd` | magnetic field at the final time | ParaView |

The default run takes about one minute on a laptop (the first run is slower because FEniCS
compiles some code and caches it).

**ParaView** (optional, free) reads the `.pvd` files: <https://www.paraview.org/download/>.
Open the `.pvd` file, click "Apply", and use the play button to step through time.

## 6. Troubleshooting

| Symptom | Fix |
|---|---|
| `conda: command not found` | Close and reopen the terminal after installing Miniconda. |
| `ModuleNotFoundError: No module named 'dolfin'` | Run `conda activate fenicsproject` first. |
| `ModuleNotFoundError: No module named 'matplotlib'` or `'pint'` | See the end of section 2. |
| `pint.errors.DimensionalityError` | A parameter has the wrong kind of unit, for example a frequency where a length is expected. The message names the two units that do not match. |
| `OSError: pkg-config probably not installed` | The environment is not activated. Run `conda activate fenicsproject`. |
| The run is too slow | Lower `n_refine`, increase `h_target`, or set `surface_plot = False` at the top of the script. |
| FEniCS seems stuck "compiling" | Delete its cache with `rm -rf ~/.cache/dijitso ~/.cache/fenics` and run again. |
| VS Code: `No module named 'dolfin'` when clicking ▷ | The wrong interpreter is selected. Redo section 4.3 and check the status bar says `fenicsproject`. |
| VS Code: ▷ gives `pkg-config` errors but the terminal works | Close the old VS Code terminals and open a new one (section 4.4), then click ▷ again. |
| VS Code: the `code` command is not found (macOS) | In VS Code, open the Command Palette and run **Shell Command: Install 'code' command in PATH**. |
| `conda env create` fails to solve | Make sure you typed `-c conda-forge`; the packages exist only on that channel. |
