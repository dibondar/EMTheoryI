"""
wave_current_dolfinx.py  --  Radiation from an oscillating current in 2D
========================================================================

A tutorial on solving a time-dependent PDE with the finite element method,
using FEniCSx (DOLFINx) and the mesh generator Gmsh.

The physics
-----------
A long straight wire along the z axis carries a current that oscillates in
time.  In the Lorenz gauge (Gaussian units) the vector potential obeys

    (1/c^2) d^2A/dt^2 - laplacian(A) = (4*pi/c) * J .

The current points along z and does not depend on z, so A = A_z(x, y, t) e_z
and the problem is two-dimensional.  We write A3 = A_z.  The total current
I(t) = -I0*sin(omega*t) is spread uniformly over the wire's cross-section:
J_z = -J0*sin(omega*t) with J0 = I0 / (pi*R_wire^2).  The problem is

    (1/c^2) d^2A3/dt^2 - laplacian(A3) = -(4*pi/c) * J0 * sin(omega*t)   inside the wire  (r < R_wire)
    (1/c^2) d^2A3/dt^2 - laplacian(A3) = 0                                in vacuum        (R_wire < r < R_max)
    A3 = 0                                                                on the outer circle r = R_max
    A3 = 0  and  dA3/dt = 0                                                at t = 0

The wire sends out cylindrical waves of wavelength lambda = c/f.  The default
frequency f = 5 GHz is a WiFi band: lambda = 6 cm, a wire of radius 1 cm in a
disc of radius 20 cm.

Space is cut off at r = R_max with A3 = 0 there.  As long as c*T_end < R_max
nothing reaches the edge and the cut-off does no harm.
The magnetic field follows from B = curl A.

Physical units and simulation units
-----------------------------------
Parameters are entered with physical units (section 1).  The computer works
in units in which all numbers are of order 1:

    lengths in units of  L0 = R_wire                       (so the wire radius is 1)
    times   in units of  T0 = R_wire / c                   (so the speed of light is 1)
    A3      in units of  A0 = J0 * L0^2 / c = I0/(pi*c)
    B       in units of  B0 = A0 / L0

With x = L0*x', t = T0*t', A3 = A0*A3' the constants drop out:

    d^2A3'/dt'^2 - laplacian'(A3') = -4*pi * sin(omega_sim * t')   inside the wire  (r' < 1)

with omega_sim = omega*T0.  Variables ending in "_sim" are plain numbers in
these units.  Section 1c converts the input, section 9 converts back.

The numerical method
--------------------
Space: a mesh of triangles; A3 is a polynomial on each triangle (finite elements).
Time: a fixed time step dt; every step solves one linear system  M * A3_new = b.

How to run (see INSTALL.md):

    conda activate fenicsx
    python wave_current_dolfinx.py

Output (in the folder "results/"):
    faces.pvd, wave.pvd, B_final.pvd  -- open with ParaView (simulation units;
                                         the script prints the conversion factors)
    frames/frame_XXX.png, wave.gif    -- the animation (physical units)
"""

import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")         # draw into files, no window needed
import matplotlib.pyplot as plt
from PIL import Image         # assembles the GIF
from pint import UnitRegistry # physical units
import gmsh                   # mesh generator
import ufl                    # the language of weak forms
from mpi4py import MPI
from petsc4py import PETSc    # linear solver
from dolfinx import fem, io
from dolfinx.mesh import exterior_facet_indices
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, set_bc


# ---------------------------------------------------------------------------
# 1. Parameters -- every knob of the simulation lives here
# ---------------------------------------------------------------------------
# A number times a unit (ureg.mm, ureg.GHz, ...) is a quantity that knows its unit.
# Any unit of the right kind works: 20.0 * ureg.cm is the same as 0.2 * ureg.m.
ureg = UnitRegistry()

# 1a. Physics (with units)
R_max     = 20.0 * ureg.cm      # radius of the computational domain
R_wire    = 1.0 * ureg.cm       # radius of the wire
f_current = 5.0 * ureg.GHz      # frequency of the current (a WiFi signal)
I0        = 1.0 * ureg.ampere   # amplitude of the total current
c         = (1.0 * ureg.speed_of_light).to("cm / s")

# 1b. Numerics and output
n_periods = 2     # periods of the current to simulate
n_frames  = 100   # snapshots of A3 to save and animate (including t = 0)

h_target = 5.0 * ureg.mm    # triangle size far from the wire; well below the wavelength
h_wire   = 1.25 * ureg.mm   # triangle size in and around the wire
degree   = 1                # polynomial degree: 1 = linear (fast), 2 = quadratic (more accurate)
steps_per_frame = 20        # time steps between two snapshots; more = smaller dt

surface_plot     = True      # True: 3D surface A3(x, y), False: flat colour map
plot_length_unit = "cm"      # unit of x and y in pictures and printed numbers
plot_time_unit   = "ps"      # unit of t in pictures and printed numbers
out_dir          = "results" # output folder
# (A3 is always shown in gauss*cm and B in gauss.)

# 1c. From physical units to simulation units
L0 = R_wire                                     # unit of length
T0 = L0 / c                                     # unit of time
I0_gauss = I0.to("statampere", "Gaussian")      # ampere (SI) -> statampere (Gaussian)
A0 = (I0_gauss / (np.pi * c)).to("gauss * cm")  # unit of A3:  A0 = J0 L0^2 / c
B0 = (A0 / L0).to("gauss")                      # unit of B

# .to("") checks that the units cancel; .magnitude gives a plain number.
R_max_sim    = (R_max / L0).to("").magnitude
R_wire_sim   = (R_wire / L0).to("").magnitude
h_target_sim = (h_target / L0).to("").magnitude
h_wire_sim   = (h_wire / L0).to("").magnitude
omega_sim    = (2 * np.pi * f_current * T0).to("").magnitude
# The speed of light is 1 in simulation units.

T_period_sim = 2 * np.pi / omega_sim              # period of the current
T_end_sim    = n_periods * T_period_sim           # final time
n_steps      = (n_frames - 1) * steps_per_frame   # number of time steps
dt_sim       = T_end_sim / n_steps                # time step

# physical value = simulation value * scale
length_scale = L0.to(plot_length_unit).magnitude
time_scale   = T0.to(plot_time_unit).magnitude
A3_scale     = A0.magnitude                       # gauss*cm
B_scale      = B0.magnitude                       # gauss

os.makedirs(out_dir, exist_ok=True)
print(f"Simulation units     : length L0 = {length_scale:.4g} {plot_length_unit},  time T0 = {time_scale:.4g} {plot_time_unit},  "
      f"A3: A0 = {A3_scale:.4g} G cm,  B: B0 = {B_scale:.4g} G")
print("                       (multiply the numbers in the .pvd files by these to get physical values)")
print(f"Wavelength           = {T_period_sim * length_scale:.3f} {plot_length_unit}  "
      f"({T_period_sim:.3f} in simulation units)")
print(f"Final time           = {T_end_sim * time_scale:.3f} {plot_time_unit}  "
      f"({n_periods} periods, {T_end_sim:.3f} in simulation units)")
print(f"Wave front at t_end  = {T_end_sim * length_scale:.3f} {plot_length_unit}  "
      f"(domain radius {R_max_sim * length_scale:.3f} {plot_length_unit})")
print(f"Time step            = {dt_sim * time_scale:.4f} {plot_time_unit}  "
      f"({n_steps} steps, {n_frames} frames, {dt_sim:.4f} in simulation units)")


# ---------------------------------------------------------------------------
# 2. Geometry and mesh (Gmsh)
# ---------------------------------------------------------------------------
# From here on everything is in simulation units.
gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)                  # no Gmsh messages

# Two discs: the whole domain and the wire.
domain_disc = gmsh.model.occ.addDisk(0, 0, 0, R_max_sim, R_max_sim)
wire_disc   = gmsh.model.occ.addDisk(0, 0, 0, R_wire_sim, R_wire_sim)

# Cut the domain along the wire outline, so the circle r = R_wire is made of triangle edges.
pieces, piece_map = gmsh.model.occ.fragment([(2, domain_disc)], [(2, wire_disc)])
gmsh.model.occ.synchronize()
wire_surfaces   = [tag for dim, tag in piece_map[1]]
vacuum_surfaces = [tag for dim, tag in pieces if tag not in wire_surfaces]

# Labels: 1 = vacuum, 2 = wire.
gmsh.model.addPhysicalGroup(2, vacuum_surfaces, 1)
gmsh.model.addPhysicalGroup(2, wire_surfaces, 2)

# Triangle size: h_wire up to a distance R_wire from the wire outline,
# growing to h_target at a distance 3 R_wire.
wire_outline = [tag for dim, tag in gmsh.model.getBoundary([(2, tag) for tag in wire_surfaces])]
gmsh.model.mesh.field.add("Distance", 1)
gmsh.model.mesh.field.setNumbers(1, "CurvesList", wire_outline)
gmsh.model.mesh.field.setNumber(1, "Sampling", 100)
gmsh.model.mesh.field.add("Threshold", 2)
gmsh.model.mesh.field.setNumber(2, "InField", 1)
gmsh.model.mesh.field.setNumber(2, "SizeMin", h_wire_sim)
gmsh.model.mesh.field.setNumber(2, "SizeMax", h_target_sim)
gmsh.model.mesh.field.setNumber(2, "DistMin", R_wire_sim)
gmsh.model.mesh.field.setNumber(2, "DistMax", 3 * R_wire_sim)
gmsh.model.mesh.field.setAsBackgroundMesh(2)
gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)   # the size field alone sets the size
gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

gmsh.model.mesh.generate(2)

# Hand the mesh and the labels over to DOLFINx.
mesh_data = io.gmsh.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)
gmsh.finalize()
mesh  = mesh_data.mesh
faces = mesh_data.cell_tags     # label of every triangle

n_triangles = mesh.topology.index_map(2).size_local
h_sim = mesh.h(2, np.arange(n_triangles))                     # size of every triangle
print(f"Mesh: {mesh.geometry.x.shape[0]} vertices, {n_triangles} triangles, "
      f"largest triangle h = {h_sim.max() * length_scale:.3f} {plot_length_unit}, "
      f"smallest h = {h_sim.min() * length_scale:.3f} {plot_length_unit}")


# ---------------------------------------------------------------------------
# 3. Sub-domains: wire and vacuum
# ---------------------------------------------------------------------------
# dx(1) = integral over the vacuum,  dx(2) = over the wire,  dx = over everything.
dx = ufl.Measure("dx", domain=mesh, subdomain_data=faces)

# Check: the wire should have area pi*R_wire^2.
wire_area_sim = fem.assemble_scalar(fem.form(fem.Constant(mesh, 1.0) * dx(2)))
print(f"Wire area from mesh  = {wire_area_sim * length_scale**2:.4f} {plot_length_unit}^2  "
      f"(exact: {np.pi * R_wire_sim**2 * length_scale**2:.4f} {plot_length_unit}^2)")

# Save the labels for ParaView (one constant per triangle = "DG" degree 0).
labels = fem.Function(fem.functionspace(mesh, ("DG", 0)), name="faces")
labels.x.array[faces.indices] = faces.values
with io.VTKFile(MPI.COMM_WORLD, out_dir + "/faces.pvd", "w") as faces_file:
    faces_file.write_function(labels)


# ---------------------------------------------------------------------------
# 4. The PDE in weak form
# ---------------------------------------------------------------------------
# Continuous functions, polynomial of the chosen degree on every triangle.
V = fem.functionspace(mesh, ("Lagrange", degree))
print(f"Number of unknowns   = {V.dofmap.index_map.size_global}")

A3 = ufl.TrialFunction(V)   # the unknown at the new time level
v = ufl.TestFunction(V)     # the test function

A3_old = fem.Function(V)              # solution at  t - dt
A3_now = fem.Function(V, name="A3")   # solution at  t
A3_new = fem.Function(V)              # solution at  t + dt  (what we solve for)

# Source, uniform in the wire:  f(t) = 4*pi * sin(omega_sim*t).  Updated in the time loop.
f = fem.Constant(mesh, 0.0)

# Newmark scheme (stable for any dt, second order):
#
#   (A3_new - 2 A3_now + A3_old) / dt^2  -  laplacian( (A3_new + 2 A3_now + A3_old) / 4 )  =  -f(t) * [inside wire]
#
# Weak form (multiply by v, integrate by parts, known terms to the right):
#
#   left  = (1/dt^2) * int A3_new v  +  (1/4) int grad(A3_new).grad(v)
#   right = (1/dt^2) * int (2 A3_now - A3_old) v  -  (1/4) int grad(2 A3_now + A3_old).grad(v)  -  int_wire f v
a = (1.0 / dt_sim**2) * A3 * v * dx  +  0.25 * ufl.dot(ufl.grad(A3), ufl.grad(v)) * dx
L = (1.0 / dt_sim**2) * (2 * A3_now - A3_old) * v * dx \
    - 0.25 * ufl.dot(ufl.grad(2 * A3_now + A3_old), ufl.grad(v)) * dx \
    - f * v * dx(2)                                     # source only inside the wire

a_form = fem.form(a)   # compile the forms once
L_form = fem.form(L)


# ---------------------------------------------------------------------------
# 5. Boundary condition
# ---------------------------------------------------------------------------
# A3 = 0 on the outer circle = all edges on the boundary of the mesh.
mesh.topology.create_connectivity(1, 2)
boundary_edges = exterior_facet_indices(mesh.topology)
boundary_dofs = fem.locate_dofs_topological(V, 1, boundary_edges)
bc = fem.dirichletbc(PETSc.ScalarType(0.0), boundary_dofs, V)

# The matrix M does not change in time: build and factorise it once.
# (M, not A, to avoid confusion with the vector potential.)
M = assemble_matrix(a_form, bcs=[bc])
M.assemble()
solver = PETSc.KSP().create(mesh.comm)
solver.setOperators(M)
solver.setType("preonly")        # direct solver ...
solver.getPC().setType("lu")     # ... by LU factorisation


# ---------------------------------------------------------------------------
# 6. Initial conditions
# ---------------------------------------------------------------------------
# New Functions are zero, so A3 = 0 and dA3/dt = 0 at t = 0 already hold.

# Snapshots are stored at the mesh vertices (degree 1 space), one column per frame.
V_plot = fem.functionspace(mesh, ("Lagrange", 1))
A3_plot = fem.Function(V_plot)
A3_frames_sim = np.zeros((V_plot.dofmap.index_map.size_local, n_frames))
t_frames_sim = np.zeros(n_frames)
wave_file = io.VTKFile(MPI.COMM_WORLD, out_dir + "/wave.pvd", "w")   # time series for ParaView
wave_file.write_function(A3_now, 0.0)                                # frame 0: the initial condition


# ---------------------------------------------------------------------------
# 7. Solve: march in time
# ---------------------------------------------------------------------------
print("\nTime stepping ...")
start = time.time()
t_sim = 0.0
frame = 1
for step in range(1, n_steps + 1):
    f.value = 4 * np.pi * np.sin(omega_sim * t_sim)   # source at the current time
    b = assemble_vector(L_form)                       # right-hand side
    apply_lifting(b, [a_form], [[bc]])                # boundary condition ...
    set_bc(b, [bc])                                   # ... A3 = 0 on the outer circle
    solver.solve(b, A3_new.x.petsc_vec)               # solve  M * A3_new = b
    A3_old.x.array[:] = A3_now.x.array                # old <- now
    A3_now.x.array[:] = A3_new.x.array                # now <- new
    t_sim = step * dt_sim

    if step % steps_per_frame == 0:                   # save a snapshot
        A3_plot.interpolate(A3_now)
        A3_frames_sim[:, frame] = A3_plot.x.array
        t_frames_sim[frame] = t_sim
        wave_file.write_function(A3_now, t_sim)
        print(f"  frame {frame + 1:2d}/{n_frames}   t = {t_sim * time_scale:7.3f} {plot_time_unit}   "
              f"max|A3| = {abs(A3_frames_sim[:, frame]).max() * A3_scale:.4g} G cm")
        frame += 1

wave_file.close()
print(f"Done in {time.time() - start:.1f} s.  Solution saved to {out_dir}/wave.pvd")


# ---------------------------------------------------------------------------
# 8. Derived field: B = curl(A)
# ---------------------------------------------------------------------------
# B = curl A = (dA3/dy, -dA3/dx, 0), one constant vector per triangle.
W = fem.functionspace(mesh, ("DG", 0, (2,)))
B = fem.Function(W, name="B")
B.interpolate(fem.Expression(ufl.as_vector((A3_now.dx(1), -A3_now.dx(0))), W.element.interpolation_points))
with io.VTKFile(MPI.COMM_WORLD, out_dir + "/B_final.pvd", "w") as B_file:
    B_file.write_function(B)
print(f"Magnetic field at the final time saved to {out_dir}/B_final.pvd")
B_max_sim = np.sqrt((B.x.array.reshape(-1, 2)**2).sum(axis=1)).max()   # largest |B| over all triangles
print(f"Largest |B| at the final time = {B_max_sim * B_scale:.4g} G")


# ---------------------------------------------------------------------------
# 9. Animation
# ---------------------------------------------------------------------------
# Back to physical units: simulation value * scale.
print(f"\nDrawing {n_frames} frames ...")
x = length_scale * V_plot.tabulate_dof_coordinates()[:, 0]   # vertex coordinates
y = length_scale * V_plot.tabulate_dof_coordinates()[:, 1]
triangles = V_plot.dofmap.list                               # the 3 vertices of each triangle
A3_frames = A3_scale * A3_frames_sim                 # gauss*cm
t_frames = time_scale * t_frames_sim
R_wire_plot = R_wire_sim * length_scale
R_max_plot = R_max_sim * length_scale
A3_min = A3_frames.min()                             # same z range for all frames
A3_max = A3_frames.max()
A3_colour_max = max(abs(A3_min), abs(A3_max))        # symmetric colour range: white at A3 = 0
# Negative A3 -> blue, zero -> white (see-through), positive -> yellow.
blue_white_yellow = matplotlib.colors.LinearSegmentedColormap.from_list(
    "blue_white_yellow",
    [(0.00, "darkblue"),
     (0.40, "blue"),
     (0.50, (1, 1, 1, 0.3)),
     (0.60, "gold"),
     (1.00, "darkorange")])

os.makedirs(out_dir + "/frames", exist_ok=True)

# The wire: a wireframe cylinder (3D plot) or a circle (2D plot).
wire_colour = "red"
phi = np.linspace(0, 2 * np.pi, 25)
wire_x = R_wire_plot * np.cos(phi)
wire_y = R_wire_plot * np.sin(phi)
if surface_plot:
    wire_z = np.linspace(A3_min, A3_max, 6)
    phi_grid, z_grid = np.meshgrid(phi, wire_z)
    cyl_x = R_wire_plot * np.cos(phi_grid)
    cyl_y = R_wire_plot * np.sin(phi_grid)
    cyl_z = z_grid

frame_files = []
for i in range(n_frames):
    fig = plt.figure(figsize=(7, 6))
    if surface_plot:
        # 3D surface: height = A3(x, y)
        ax = fig.add_subplot(projection="3d")
        ax.plot_trisurf(x, y, A3_frames[:, i], triangles=triangles,
                        cmap=blue_white_yellow, vmin=-A3_colour_max, vmax=A3_colour_max,
                        linewidth=0, antialiased=False)
        ax.plot_wireframe(cyl_x, cyl_y, cyl_z, color=wire_colour,
                          linewidth=0.8, rstride=1, cstride=2)
        ax.set_zlim(A3_min, A3_max)
        ax.set_zlabel("A3 [G cm]")
    else:
        # flat colour map: colour = A3(x, y)
        ax = fig.add_subplot()
        colours = ax.tripcolor(x, y, triangles, A3_frames[:, i],
                               shading="gouraud", cmap=blue_white_yellow,
                               vmin=-A3_colour_max, vmax=A3_colour_max)
        ax.plot(wire_x, wire_y, color=wire_colour, linewidth=1.5)
        fig.colorbar(colours, ax=ax, label="A3 [G cm]")
        ax.set_aspect("equal")
    ax.set_xlim(-R_max_plot, R_max_plot)
    ax.set_ylim(-R_max_plot, R_max_plot)
    ax.set_xlabel(f"x [{plot_length_unit}]")
    ax.set_ylabel(f"y [{plot_length_unit}]")
    ax.set_title(f"$A_3(x, y)$   t = {t_frames[i]:.2f} {plot_time_unit}")
    filename = f"{out_dir}/frames/frame_{i:03d}.png"
    fig.savefig(filename, dpi=300)
    plt.close(fig)
    frame_files.append(filename)

# Glue the frames into a GIF, 10 frames per second.
images = [Image.open(filename) for filename in frame_files]
images[0].save(out_dir + "/wave.gif", save_all=True, append_images=images[1:], duration=100, loop=0)
print(f"Animation saved to {out_dir}/wave.gif")
