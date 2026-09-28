# Installing the software

The examples in this course use **FEniCS 2019.1** (the "legacy" FEniCS, Python module `dolfin`).
It is *not* the newer FEniCSx / `dolfinx`; the scripts will not run there.
Plots need **matplotlib**. Everything is installed with conda, which takes about 10 minutes
and 2 GB of disk space.

## 1. What you need

- A terminal (macOS: Terminal.app, Linux: any terminal, Windows: see the platform note below).
- **Miniconda**: download the installer for your system from
  <https://www.anaconda.com/download/success> (scroll to "Miniconda installers") and run it.
  After installing, close and reopen the terminal, then check that `conda --version` prints a version.

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

This creates an environment named `fenicsproject` with FEniCS, numpy and matplotlib.

**Option B (by hand).** The same, spelled out:

```bash
conda create -n fenicsproject -c conda-forge python=3.11 fenics=2019.1.0 numpy matplotlib
```

**Already have a `fenicsproject` environment without matplotlib?** Add it:

```bash
conda install -n fenicsproject -c conda-forge matplotlib
```

## 3. Activate and check

Every time you open a new terminal, activate the environment first:

```bash
conda activate fenicsproject
```

(On old conda versions the command is `source activate fenicsproject`.)
Then check that everything imports:

```bash
python -c "import dolfin, matplotlib; print('FEniCS', dolfin.__version__)"
```

Expected output: `FEniCS 2019.1.0`. The first import can take a minute because FEniCS compiles
some code on first use.

## 4. Run an example

```bash
python wave_current_fenics.py
```

It prints progress and writes everything to the folder `results/`:

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

## 5. Troubleshooting

| Symptom | Fix |
|---|---|
| `conda: command not found` | Close and reopen the terminal after installing Miniconda. |
| `ModuleNotFoundError: No module named 'dolfin'` | Run `conda activate fenicsproject` first. |
| `ModuleNotFoundError: No module named 'matplotlib'` | See the end of section 2. |
| `OSError: pkg-config probably not installed` | The environment is not activated. Run `conda activate fenicsproject`. |
| The run is too slow | Lower `n_refine`, increase `h_target`, or set `surface_plot = False` at the top of the script. |
| FEniCS seems stuck "compiling" | Delete its cache with `rm -rf ~/.cache/dijitso ~/.cache/fenics` and run again. |
| `conda env create` fails to solve | Make sure you typed `-c conda-forge`; the packages exist only on that channel. |
