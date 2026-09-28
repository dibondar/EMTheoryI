# EM Theory I: radiation from an oscillating current (FEniCS)

`wave_current_fenics.py` solves the 2D wave equation for the vector potential
`A3 = A_z(x, y, t)` produced by a current that oscillates in a wire of radius 1
at the centre of a large disc of radius 20 (Gaussian units, `c = 1`):

```
(1/c^2) d^2A3/dt^2 - laplacian(A3) = -(4 pi / c) sin(omega t)   inside the wire
(1/c^2) d^2A3/dt^2 - laplacian(A3) = 0                          in vacuum
A3 = 0 on the outer circle,   A3 = dA3/dt = 0 at t = 0
```

The script is a self-contained tutorial: read it from top to bottom, the
comments explain the physics and every numerical step. Installation
instructions are in [INSTALL.md](INSTALL.md).

## Running

```bash
conda activate fenicsproject
python wave_current_fenics.py
```

Results go to `results/`: an animated GIF plus individual PNG frames, and `.pvd` files for ParaView with the full solution, the mesh
regions and the magnetic field at the final time.

The wave travels at speed `c = 1`, so after two periods (`t = 4 pi = 12.6`) the
front has reached `r = 12.6`, well short of the wall at `r = 20`. The zero
boundary condition therefore never reflects anything and the disc behaves like
free space. Make `R_max` smaller to see reflections.

## What each section of the script does

| Section | Step | Key FEniCS objects |
|---|---|---|
| 1 | Parameters: geometry, frequency, mesh size, time step | plain Python numbers |
| 2 | Mesh of the disc, refined near the wire | `UnitDiscMesh`, `refine` |
| 3 | Label each triangle as vacuum (1) or wire (2) | `MeshFunction`, `Measure("dx", subdomain_data=...)` |
| 4 | Weak form of the wave equation, discretised in time | `FunctionSpace`, `TrialFunction`, `TestFunction`, `Expression` |
| 5 | Boundary condition `A3 = 0` on the outer circle | `DirichletBC` |
| 6 | Initial conditions `A3 = dA3/dt = 0` | new `Function`s are zero |
| 7 | Time loop: build right-hand side, solve, shift time levels | `assemble`, `solve`, `assign` |
| 8 | Magnetic field `B = curl A` | `project`, `A3.dx(0)`, `A3.dx(1)` |
| 9 | Pictures and animated GIF | matplotlib `plot_trisurf` / `tripcolor`, PIL |

Time stepping uses a fixed step and the Newmark scheme (central difference in
time, Laplacian averaged over three time levels), which is stable for any
`dt`. Exercise 5 below shows what goes wrong with a simpler explicit scheme.

## Exercises

Each one needs a change of one or two lines in the parameter block (section 1)
or in the marked place in the code.

1. **Frequency.** Set `omega = 2`. How do the wavelength and the number of
   rings within `r < 12.6` change? Check against `lambda = 2 pi c / omega`.
2. **Reflections.** Set `R_max = 8`. The wave now hits the wall where `A3 = 0`.
   Describe what the reflected wave looks like and why its sign flips.
3. **Wire size.** Set `R_wire = 3` (keep `n_refine = 2`). Does the far field
   still look like that of a point source? Look at the wire-area check that the
   script prints.
4. **Accuracy.** Set `degree = 2` and compare `max|A3|` in the last frame with
   the `degree = 1` run. Then halve `dt` by doubling `steps_per_frame`. Which
   change matters more?
5. **Explicit time stepping.** In section 4 change the Laplacian average
   `(A3_new + 2 A3_now + A3_old)/4` to just `A3_now` (the "leapfrog" scheme): in
   the form `a` drop the `0.25 * dot(grad(A3), grad(v))` term and in `L` replace
   `0.25 * dot(grad(2*A3_now + A3_old), grad(v))` by `dot(grad(A3_now), grad(v))`.
   Run with `steps_per_frame = 10`, then `steps_per_frame = 2`. Explain what
   happens using the CFL condition `c dt < h_min`.
6. **Magnetic field.** Move the `B = curl A` block (section 8) inside the time
   loop and save `B` at every frame. Animate `|B|` with `tripcolor` in ParaView
   or matplotlib. Where is `B` largest, and how does it decay with `r`?
