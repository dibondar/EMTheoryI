"""
wave_current_fenics.py  --  Radiation from an oscillating current in 2D
=======================================================================

A tutorial on solving a time-dependent PDE with the finite element method,
using FEniCS (legacy dolfin 2019.1).

The physics
-----------
A long straight wire along the z axis carries a current that oscillates in
time.  In the Lorenz gauge (Gaussian units) the vector potential obeys the
inhomogeneous wave equation

    (1/c^2) d^2A/dt^2 - laplacian(A) = (4*pi/c) * J .

The current points along z and does not depend on z, so A = A_z(x, y, t) e_z
and the problem is two-dimensional.  We write A3 = A_z and take the current
density J_z = -sin(omega*t) uniformly inside the wire (the minus sign only
flips the overall sign of A3).  The problem we solve is then

    (1/c^2) d^2A3/dt^2 - laplacian(A3) = -(4*pi/c) * sin(omega*t)   inside the wire  (r < R_wire)
    (1/c^2) d^2A3/dt^2 - laplacian(A3) = 0                           in vacuum        (R_wire < r < R_max)
    A3 = 0                                                           on the outer circle r = R_max
    A3 = 0  and  dA3/dt = 0                                           at t = 0

The wire sends out cylindrical waves of wavelength  lambda = 2*pi*c/omega.
Space is infinite, but a computer can only handle a finite region, so we cut
it off at r = R_max and put A3 = 0 there.  Waves travel at speed c, so as long
as c*T_end < R_max nothing reaches the edge and the cut-off does no harm.
The magnetic field follows from B = curl A.

The numerical method, in one paragraph
--------------------------------------
Space: the disc is covered by a mesh of triangles and A3 is approximated by a
function that is a polynomial on each triangle (the finite element method).
Time: we step forward with a fixed time step dt; at every step one linear
system  M * A3_new = b  is solved.  Each section below explains its part.

The script is meant to be read from top to bottom.  All parameters you may
want to change are collected in section 1.

How to run (see INSTALL.md for setting up the environment):

    conda activate fenicsproject
    python wave_current_fenics.py

Output (in the folder "results/"):
    faces.pvd, wave.pvd, B_final.pvd  -- open with ParaView
    frames/frame_XXX.png, wave.gif    -- the animation
"""

import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")         # draw into files, no window needed
import matplotlib.pyplot as plt
from PIL import Image         # comes with matplotlib, used to assemble the GIF
from dolfin import *          # the FEniCS library (this star-import is the FEniCS convention)

set_log_level(LogLevel.WARNING)   # hide FEniCS's "Calling FFC just-in-time compiler" chatter


# ---------------------------------------------------------------------------
# 1. Parameters -- every knob of the simulation lives here
# ---------------------------------------------------------------------------
# Physics
R_max   = 20.0   # radius of the computational domain (vacuum around the wire)
R_wire  = 1.0    # radius of the wire that carries the current
c       = 1.0    # speed of light
omega   = 1.0    # angular frequency of the current

# How long to simulate and how many pictures to keep
n_periods = 2    # how many periods of the current to simulate
n_frames  = 50   # how many snapshots of A3 to save and animate (including t = 0)

# Discretisation
h_target = 0.5   # target size of the mesh triangles; should be well below the wavelength
n_refine = 2     # extra mesh refinements around the wire (0 = none), see section 2
degree   = 1     # polynomial degree of the finite elements: 1 = linear (fast),
                 # 2 = quadratic (more accurate, ~4x slower)
steps_per_frame = 10        # time steps between two saved snapshots; more steps = smaller dt = more accurate

# Output
surface_plot    = True      # True: 3D surface plot A3(x, y), False: flat 2D colour map
out_dir         = "results" # where all output files go

# Quantities derived from the parameters above
T_period = 2 * np.pi / omega                 # period of the current
T_end    = n_periods * T_period              # final time
n_steps  = (n_frames - 1) * steps_per_frame  # total number of time steps ...
dt       = T_end / n_steps                   # ... and the resulting time step

os.makedirs(out_dir, exist_ok=True)
print("Wavelength           = %.3f" % (2 * np.pi * c / omega))
print("Final time           = %.3f  (%d periods)" % (T_end, n_periods))
print("Wave front at t_end  = %.3f  (domain radius %.1f)" % (c * T_end, R_max))
print("Time step            = %.4f  (%d steps, %d frames)" % (dt, n_steps, n_frames))


# ---------------------------------------------------------------------------
# 2. Geometry and mesh
# ---------------------------------------------------------------------------
# The domain is a disc of radius R_max.  FEniCS itself has no tool that meshes
# arbitrary shapes (that is the job of the optional package "mshr", which is
# not available on every computer).  Instead we take the built-in mesh of the
# UNIT disc and stretch it to radius R_max.  UnitDiscMesh consists of
# "n_rings" concentric rings of triangles and its largest triangle has size
# about 1.43 / n_rings.  We therefore pick n_rings so that, after stretching
# by R_max, the triangles have size h_target.
n_rings = int(round(1.43 * R_max / h_target))
mesh = UnitDiscMesh.create(MPI.comm_world, n_rings, 1, 2)   # arguments: (communicator, rings, degree, dimension)
mesh.coordinates()[:] = R_max * mesh.coordinates()          # multiply every vertex (x, y) by R_max

# The wire is small compared with the domain, and the triangle edges do not
# follow its boundary r = R_wire (section 3 decides "wire or vacuum" triangle
# by triangle).  Smaller triangles near the wire make the labelled wire a
# better circle and resolve the source better.  Each refinement splits every
# marked triangle into 4 smaller ones.
for k in range(n_refine):
    cell_markers = MeshFunction("bool", mesh, 2, False)     # one True/False flag per triangle ("cell")
    for cell in cells(mesh):
        if cell.midpoint().norm() < 2.0 * R_wire:            # triangle centre closer than 2 R_wire to the origin
            cell_markers[cell] = True
    mesh = refine(mesh, cell_markers)

print("Mesh: %d vertices, %d triangles, largest triangle h = %.3f, smallest h = %.3f"
      % (mesh.num_vertices(), mesh.num_cells(), mesh.hmax(), mesh.hmin()))


# ---------------------------------------------------------------------------
# 3. Sub-domains: wire and vacuum
# ---------------------------------------------------------------------------
# The source acts only inside the wire, so we must know which triangles belong
# to it.  Give every triangle a label: 1 = vacuum, 2 = wire.  A triangle counts
# as "wire" if its centre lies inside r < R_wire.
faces = MeshFunction("size_t", mesh, 2, 1)                  # start with label 1 everywhere
for cell in cells(mesh):
    if cell.midpoint().norm() < R_wire:                      # triangle centre is inside the wire
        faces[cell] = 2

# Attach the labels to the integration measure: from now on
#   dx(1) = integral over the vacuum,  dx(2) = integral over the wire,  dx = over everything.
dx = Measure("dx", domain=mesh, subdomain_data=faces)

# Sanity check: the labelled wire should have area pi*R_wire^2.
wire_area = assemble(Constant(1.0) * dx(2))
print("Wire area from mesh  = %.4f  (exact: %.4f)" % (wire_area, np.pi * R_wire**2))

# Save the labels so you can look at the two regions in ParaView.
File(out_dir + "/faces.pvd") << faces


# ---------------------------------------------------------------------------
# 4. The PDE in weak form
# ---------------------------------------------------------------------------
# Finite element space: continuous functions that are polynomials of the
# chosen degree on every triangle.  The unknowns are the values of A3 at the
# mesh nodes (for degree 1: at the vertices).
V = FunctionSpace(mesh, "P", degree)
print("Number of unknowns   = %d" % V.dim())

A3 = TrialFunction(V)  # the unknown at the new time level (a symbol used to build the matrix)
v = TestFunction(V)    # the test function of the weak form

A3_old = Function(V)   # solution at the previous time  t - dt
A3_now = Function(V)   # solution at the current time   t
A3_new = Function(V)   # solution at the next time      t + dt  (what we solve for)
A3_now.rename("A3", "vector potential A_z")  # name shown in ParaView

# The source term, uniform in space, oscillating in time:  f(t) = (4*pi/c) * sin(omega*t).
# It is an Expression so that we can update its time "f.t" inside the loop.
f = Expression("f0 * sin(omega * t)", f0=4 * np.pi / c, omega=omega, t=0.0, degree=0)

# Time discretisation ("Newmark", average acceleration).  Replace the second
# time derivative by a central difference and evaluate the Laplacian as the
# average of the three time levels:
#
#   (A3_new - 2 A3_now + A3_old) / (c dt)^2  -  laplacian( (A3_new + 2 A3_now + A3_old) / 4 )  =  -f(t) * [inside wire]
#
# This scheme is stable for ANY time step (no CFL condition), and second-order accurate in dt.
# (Exercise 5 in README.md shows what happens with a simpler, explicit scheme.)
#
# Weak form: multiply by v, integrate over the domain, integrate the Laplacian
# by parts (the boundary term vanishes because A3 = 0 on the boundary), and
# move everything that is known (A3_now, A3_old, f) to the right-hand side:
#
#   left  = (1/(c dt)^2) * int A3_new v  +  (1/4) int grad(A3_new).grad(v)
#   right = (1/(c dt)^2) * int (2 A3_now - A3_old) v  -  (1/4) int grad(2 A3_now + A3_old).grad(v)  -  int_wire f v
a = (1.0 / (c * dt)**2) * A3 * v * dx  +  0.25 * dot(grad(A3), grad(v)) * dx
L = (1.0 / (c * dt)**2) * (2 * A3_now - A3_old) * v * dx \
    - 0.25 * dot(grad(2 * A3_now + A3_old), grad(v)) * dx \
    - f * v * dx(2)                                     # the source acts only inside the wire (label 2)


# ---------------------------------------------------------------------------
# 5. Boundary condition
# ---------------------------------------------------------------------------
# A3 = 0 on the boundary of the mesh, which is the outer circle r = R_max.
# "on_boundary" is FEniCS shorthand for "every point on the edge of the mesh".
# The wire's surface r = R_wire is inside the domain, not on its edge: there
# A3 and its derivative are simply continuous, which the weak form takes care of.
bc = DirichletBC(V, Constant(0.0), "on_boundary")

# The matrix M of the left-hand side does not change in time: build it once.
# (We call it M rather than A so it is not confused with the vector potential.)
M = assemble(a)
bc.apply(M)


# ---------------------------------------------------------------------------
# 6. Initial conditions
# ---------------------------------------------------------------------------
# A new Function is zero everywhere, so A3_old = A3_now = 0 already encodes
# A3(x, 0) = 0 and dA3/dt(x, 0) = 0.  Nothing to do.

# Storage for the snapshots: one column per frame, one row per mesh vertex.
A3_frames = np.zeros((mesh.num_vertices(), n_frames))
t_frames = np.zeros(n_frames)
wave_file = File(out_dir + "/wave.pvd")                 # time series for ParaView

A3_frames[:, 0] = A3_now.compute_vertex_values(mesh)    # frame 0 is the initial condition
t_frames[0] = 0.0
wave_file << (A3_now, 0.0)


# ---------------------------------------------------------------------------
# 7. Solve: march in time
# ---------------------------------------------------------------------------
# At every step: evaluate the source, build the right-hand side from the two
# known time levels, solve for the new one, and shift the time levels by one.
print("\nTime stepping ...")
start = time.time()
t = 0.0
frame = 1
for step in range(1, n_steps + 1):
    f.t = t                                # source at the current time (centre of the stencil)
    b = assemble(L)                        # right-hand side vector
    bc.apply(b)                            # enforce A3 = 0 on the outer circle
    solve(M, A3_new.vector(), b)           # linear solve for the new time level
    A3_old.assign(A3_now)                  # shift the time levels:  old <- now
    A3_now.assign(A3_new)                  #                         now <- new
    t = step * dt

    if step % steps_per_frame == 0:        # time to save a snapshot
        A3_frames[:, frame] = A3_now.compute_vertex_values(mesh)
        t_frames[frame] = t
        wave_file << (A3_now, t)
        print("  frame %2d/%d   t = %6.3f   max|A3| = %.4f" % (frame + 1, n_frames, t, abs(A3_frames[:, frame]).max()))
        frame += 1

print("Done in %.1f s.  Solution saved to %s/wave.pvd" % (time.time() - start, out_dir))


# ---------------------------------------------------------------------------
# 8. Derived field: B = curl(A)
# ---------------------------------------------------------------------------
# With A = A3(x, y) e_z the magnetic field is  B = curl A = (dA3/dy, -dA3/dx, 0).
# A3.dx(0) means dA3/dx and A3.dx(1) means dA3/dy.  The gradient of a linear
# function is constant on each triangle, hence "DG" (discontinuous) degree 0.
W = VectorFunctionSpace(mesh, "DG", 0)
B = project(as_vector((A3_now.dx(1), -A3_now.dx(0))), W)
B.rename("B", "magnetic field")
File(out_dir + "/B_final.pvd") << B
print("Magnetic field at the final time saved to %s/B_final.pvd" % out_dir)


# ---------------------------------------------------------------------------
# 9. Animation
# ---------------------------------------------------------------------------
# Draw every saved snapshot with matplotlib, write it to a PNG file, and glue
# the PNGs into an animated GIF.
print("\nDrawing %d frames ..." % n_frames)
x = mesh.coordinates()[:, 0]                         # vertex coordinates ...
y = mesh.coordinates()[:, 1]
triangles = np.asarray(mesh.cells(), dtype=int)      # ... and which 3 vertices form each triangle
A3_min = A3_frames.min()                             # fixed colour / z range for all frames
A3_max = A3_frames.max()
os.makedirs(out_dir + "/frames", exist_ok=True)

frame_files = []
for i in range(n_frames):
    fig = plt.figure(figsize=(7, 6))
    if surface_plot:
        # 3D surface: height = A3(x, y)
        ax = fig.add_subplot(projection="3d")
        ax.plot_trisurf(x, y, A3_frames[:, i], triangles=triangles,
                        cmap="viridis", vmin=A3_min, vmax=A3_max, linewidth=0, antialiased=False)
        ax.set_zlim(A3_min, A3_max)
        ax.set_zlabel("A3")
    else:
        # flat colour map: colour = A3(x, y)
        ax = fig.add_subplot()
        colours = ax.tripcolor(x, y, triangles, A3_frames[:, i],
                               shading="gouraud", cmap="viridis", vmin=A3_min, vmax=A3_max)
        fig.colorbar(colours, ax=ax, label="A3")
        ax.set_aspect("equal")
    ax.set_xlim(-R_max, R_max)
    ax.set_ylim(-R_max, R_max)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("t = %.2f" % t_frames[i])
    filename = out_dir + "/frames/frame_%03d.png" % i
    fig.savefig(filename, dpi=80)
    plt.close(fig)
    frame_files.append(filename)

# Glue the frames into an animated GIF, 10 frames per second (100 ms per frame).
images = []
for filename in frame_files:
    images.append(Image.open(filename))
images[0].save(out_dir + "/wave.gif", save_all=True, append_images=images[1:], duration=100, loop=0)
print("Animation saved to %s/wave.gif" % out_dir)
