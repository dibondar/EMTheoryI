# EM Theory I: radiation from an oscillating current (FEniCSx)

`wave_current_dolfinx.py` solves the 2D wave equation for the vector potential
`A3 = A_z(x, y, t)` produced by a current `I(t) = -I0 sin(omega t)` that
oscillates in a wire of radius `R_wire` at the centre of a large disc of radius
`R_max` (Gaussian units, `J0 = I0 / (pi R_wire^2)`):

```
(1/c^2) d^2A3/dt^2 - laplacian(A3) = -(4 pi / c) J0 sin(omega t)   inside the wire
(1/c^2) d^2A3/dt^2 - laplacian(A3) = 0                             in vacuum
A3 = 0 on the outer circle,   A3 = dA3/dt = 0 at t = 0
```

The default values are `R_wire = 1 cm`, `R_max = 20 cm`, a frequency of 5 GHz
(wavelength 6 cm) and `I0 = 1 A`.

**You are simulating a WiFi signal.** 5 GHz is one of the two bands a WiFi
router uses (the other is 2.4 GHz). The oscillating current in the wire stands
in for the current in the router's antenna, and the waves it sends out are the
radio waves that carry your data.

The script is a self-contained tutorial: read it from top to bottom, the
comments explain the physics and every numerical step. Installation
instructions are in [INSTALL.md](INSTALL.md). For a short introduction to
the weak form used in section 4, see [WEAK_FORM.md](WEAK_FORM.md).

## Physical units in, simulation units inside

Parameters are entered with their physical units, using the
[pint](https://pint.readthedocs.io) library:

```python
R_max     = 20.0 * ureg.cm
R_wire    = 1.0 * ureg.cm
f_current = 5.0 * ureg.GHz
I0        = 1.0 * ureg.ampere
```

Any unit of the right kind works, SI or Gaussian (`0.2 * ureg.m` is the same
as `20.0 * ureg.cm`). A quantity of the wrong kind stops the script with an
error before anything is computed.

The solver itself does not see these units. Section 1c of the script converts
every input to **simulation units**, in which the wire radius and the speed of
light are both 1 and all numbers are of order 1:

| Quantity | Unit | Default value |
|---|---|---|
| length | `L0 = R_wire` | 1 cm |
| time | `T0 = R_wire / c` | 33.36 ps |
| `A3` | `A0 = I0 / (pi c)` | 0.0318 G cm |
| `B` | `B0 = A0 / L0` | 0.0318 G |

Variables whose names end in `_sim` are plain numbers in these units. The
pictures and the printed numbers are converted back to physical units
(`plot_length_unit`, `plot_time_unit`, gauss). The `.pvd` files for ParaView
stay in simulation units; the script prints the four conversion factors.

## Running

```bash
conda activate fenicsx
python wave_current_dolfinx.py
```

Results go to `results/`: an animated GIF plus individual PNG frames, and `.pvd` files for ParaView with the full solution, the mesh
regions and the magnetic field at the final time.

The wave travels at speed `c`, so after two periods (`t = 400 ps`) the
front has reached `r = 12 cm`, well short of the wall at `r = 20 cm`. The zero
boundary condition therefore never reflects anything and the disc behaves like
free space. Make `R_max` smaller to see reflections.

`wave_square_wire_dolfinx.py` is the same simulation for a wire of square
cross-section in a square box; its results go to `results_square/`.

## What each section of the script does

| Section | Step | Key objects |
|---|---|---|
| 1 | Parameters with physical units, conversion to simulation units | pint `UnitRegistry`, `.to(...)`, `.magnitude` |
| 2 | Mesh of the disc, fine near the wire | `gmsh` (`addDisk`, `fragment`, size field), `model_to_mesh` |
| 3 | Labels of the triangles: vacuum (1) or wire (2) | `cell_tags`, `ufl.Measure("dx", subdomain_data=...)` |
| 4 | Weak form of the wave equation, discretised in time | `fem.functionspace`, `ufl.TrialFunction`, `ufl.TestFunction`, `fem.Constant`, `fem.form` |
| 5 | Boundary condition `A3 = 0` on the outer circle, matrix `M` | `fem.dirichletbc`, `assemble_matrix`, PETSc `KSP` |
| 6 | Initial conditions `A3 = dA3/dt = 0` | new `fem.Function`s are zero |
| 7 | Time loop: build right-hand side, solve, shift time levels | `assemble_vector`, `set_bc`, `solver.solve` |
| 8 | Magnetic field `B = curl A` | `fem.Expression`, `interpolate`, `A3.dx(0)`, `A3.dx(1)` |
| 9 | Pictures and animated GIF | matplotlib `plot_trisurf` / `tripcolor`, PIL |

Time stepping uses a fixed step and the Newmark scheme (central difference in
time, Laplacian averaged over three time levels), which is stable for any
`dt`. Exercise 5 below shows what goes wrong with a simpler explicit scheme.

## Exercises

Each one needs a change of one or two lines in the parameter block (sections
1a and 1b) or in the marked place in the code.

1. **Frequency.** Set `f_current = 10.0 * ureg.GHz`. How do the wavelength and
   the number of rings change? Check against `lambda = c / f`. Then try the
   other WiFi band, `f_current = 2.4 * ureg.GHz`: does the wave front still stay
   inside the disc? (Compare the printed wave-front position with `R_max`.)
2. **Reflections.** Set `R_max = 8.0 * ureg.cm`. The wave now hits the wall
   where `A3 = 0`. Describe what the reflected wave looks like and why its sign flips.
3. **Wire size.** Set `R_wire = 3.0 * ureg.cm`. Does the far field
   still look like that of a point source? Look at the wire-area check that the
   script prints.
4. **Accuracy.** Set `degree = 2` and compare `max|A3|` in the last frame with
   the `degree = 1` run. Then halve the time step by doubling `steps_per_frame`. Which
   change matters more?
5. **Explicit time stepping.** In section 4 change the Laplacian average
   `(A3_new + 2 A3_now + A3_old)/4` to just `A3_now` (the "leapfrog" scheme): in
   the form `a` drop the `0.25 * ufl.dot(ufl.grad(A3), ufl.grad(v))` term and in `L` replace
   `0.25 * ufl.dot(ufl.grad(2 * A3_now + A3_old), ufl.grad(v))` by `ufl.dot(ufl.grad(A3_now), ufl.grad(v))`.
   Run with `steps_per_frame = 10`, then `steps_per_frame = 2`. Explain what
   happens using the CFL condition `c dt < h_min` (in simulation units: `dt_sim < h_min_sim`).
6. **Magnetic field.** Move the `B = curl A` block (section 8) inside the time
   loop and save `B` at every frame. Animate `|B|` with `tripcolor` in ParaView
   or matplotlib. Where is `B` largest, and how does it decay with `r`?
7. **Units.** Enter `R_max` in metres, then in inches (`ureg.inch`), with the
   same physical size. Check that the printed numbers do not change. Then set
   `R_max = 2.0 * ureg.GHz` and read the error message. Finally set
   `plot_length_unit = "mm"` and `plot_time_unit = "ns"`.
8. **Scaling.** Divide `R_max`, `R_wire` and `h_target` by 10 and multiply
   `f_current` by 10 (a 50 GHz signal from a 1 mm wire). Compare the numbers
   "in simulation units" that the script prints with those of the default run.
   Which printed physical values change, and by what factor? Explain why `A3`
   stays the same while `B` becomes ten times larger.
