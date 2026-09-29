"""
wave_square_wire_fenics.py  --  Radiation from an oscillating current in a SQUARE wire
=====================================================================================

A tutorial on solving a time-dependent PDE with the finite element method,
using FEniCS (legacy dolfin 2019.1).

This is the square-wire companion of wave_current_fenics.py, which treats a
round wire.  Only the shape of the wire (and, to go with it, the shape of the
computational domain) is different; the physics and the time stepping are the
same.  Compare the two animations: close to the wire the square shape is
visible in A3, but a few wavelengths away the waves become circular again.

The physics
-----------
A long straight wire of square cross-section, along the z axis, carries a current that oscillates in
time.  In the Lorenz gauge (Gaussian units) the vector potential obeys the
inhomogeneous wave equation

    (1/c^2) d^2A/dt^2 - laplacian(A) = (4*pi/c) * J .

The current points along z and does not depend on z, so A = A_z(x, y, t) e_z
and the problem is two-dimensional.  We write A3 = A_z and take the current
density J_z = -sin(omega*t) uniformly inside the wire (the minus sign only
flips the overall sign of A3).  The problem we solve is then

    (1/c^2) d^2A3/dt^2 - laplacian(A3) = -(4*pi/c) * sin(omega*t)   inside the wire  (|x| < a_wire and |y| < a_wire)
    (1/c^2) d^2A3/dt^2 - laplacian(A3) = 0                           in vacuum        (the rest of the box)
    A3 = 0                                                           on the edge of the box |x| = L_box or |y| = L_box
    A3 = 0  and  dA3/dt = 0                                           at t = 0

Here a_wire is HALF the side of the square wire and L_box is half the side of
the square computational box.  The wire sends out waves of wavelength
lambda = 2*pi*c/omega.  Space is infinite, but a computer can only handle a
finite region, so we cut it off at the edge of the box and put A3 = 0 there.
Waves travel at speed c, so as long as a_wire + c*T_end < L_box nothing
reaches the edge and the cut-off does no harm.
The magnetic field follows from B = curl A.

The numerical method, in one paragraph
--------------------------------------
Space: the box is covered by a mesh of triangles and A3 is approximated by a
function that is a polynomial on each triangle (the finite element method).
Time: we step forward with a fixed time step dt; at every step one linear
system  M * A3_new = b  is solved.  Each section below explains its part.

The script is meant to be read from top to bottom.  All parameters you may
want to change are collected in section 1.

How to run (see INSTALL.md for setting up the environment):

    conda activate fenicsproject
    python wave_square_wire_fenics.py

Output (in the folder "results_square/"; the round-wire script uses "results/"):
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
L_box   = 20.0   # half the side of the square computational domain (vacuum around the wire)
a_wire  = 1.0    # half the side of the square wire that carries the current
c       = 1.0    # speed of light
omega   = 1.0    # angular frequency of the current

# How long to simulate and how many pictures to keep
n_periods = 2    # how many periods of the current to simulate
n_frames  = 50   # how many snapshots of A3 to save and animate (including t = 0)

# Discretisation
h_target = 0.5   # target spacing of the mesh grid; should be well below the wavelength
n_refine = 2     # extra mesh refinements around the wire (0 = none), see section 2
degree   = 1     # polynomial degree of the finite elements: 1 = linear (fast),
                 # 2 = quadratic (more accurate, ~4x slower)
steps_per_frame = 10        # time steps between two saved snapshots; more steps = smaller dt = more accurate

# Output
surface_plot    = True      # True: 3D surface plot A3(x, y), False: flat 2D colour map
out_dir         = "results_square" # where all output files go

# Quantities derived from the parameters above
T_period = 2 * np.pi / omega                 # period of the current
T_end    = n_periods * T_period              # final time
n_steps  = (n_frames - 1) * steps_per_frame  # total number of time steps ...
dt       = T_end / n_steps                   # ... and the resulting time step

os.makedirs(out_dir, exist_ok=True)
print("Wavelength           = %.3f" % (2 * np.pi * c / omega))
print("Final time           = %.3f  (%d periods)" % (T_end, n_periods))
print("Wave front at t_end  = %.3f  (distance from wire to box edge %.1f)" % (c * T_end, L_box - a_wire))
print("Time step            = %.4f  (%d steps, %d frames)" % (dt, n_steps, n_frames))


# ---------------------------------------------------------------------------
# 2. Geometry and mesh
# ---------------------------------------------------------------------------
# The domain is the square box  -L_box < x, y < L_box.  FEniCS builds such a
# mesh itself: RectangleMesh cuts the box into n_grid x n_grid small squares
# and splits every square into triangles.  With the option "crossed" each
# square is split by both diagonals into 4 triangles, which makes the mesh look
# the same in all directions (the waves then travel equally well along x, y and
# the diagonals).
#
# The wire's edges x = +-a_wire and y = +-a_wire are straight lines.  If they
# coincide with grid lines, then no triangle is cut by the wire's surface and
# every triangle is either completely inside or completely outside the wire.
# The wire is then represented EXACTLY (compare with the round wire, where
# the triangle edges can only approximate the circle).  To achieve this the
# grid spacing h must fit a whole number of times into a_wire and into L_box.
n_wire = max(1, int(round(a_wire / h_target)))   # grid cells from the centre to the wire's edge
h = a_wire / n_wire                              # the actual grid spacing, close to h_target
n_grid = int(round(2 * L_box / h))               # grid cells across the whole box
if abs(n_grid * h - 2 * L_box) > 1e-9 * L_box:
    raise ValueError("The grid spacing h = %.4f does not fit into the box: choose L_box as a "
                     "multiple of h (for example a multiple of a_wire)." % h)
mesh = RectangleMesh(Point(-L_box, -L_box), Point(L_box, L_box), n_grid, n_grid, "crossed")

# The wire is small compared with the domain.  Smaller triangles near the
# wire resolve the source and the strong bending of A3 at the wire's corners
# better.  Each refinement splits every marked triangle into 4 smaller ones by
# cutting its edges in half; the old edges stay (in halves), so the wire's
# edges remain lines of the mesh.
for k in range(n_refine):
    cell_markers = MeshFunction("bool", mesh, 2, False)     # one True/False flag per triangle ("cell")
    for cell in cells(mesh):
        p = cell.midpoint()                                   # centre of the triangle
        if abs(p.x()) < 2.0 * a_wire and abs(p.y()) < 2.0 * a_wire:   # inside a square twice the wire's size
            cell_markers[cell] = True
    mesh = refine(mesh, cell_markers)

print("Mesh: %d vertices, %d triangles, largest triangle h = %.3f, smallest h = %.3f"
      % (mesh.num_vertices(), mesh.num_cells(), mesh.hmax(), mesh.hmin()))


# ---------------------------------------------------------------------------
# 3. Sub-domains: wire and vacuum
# ---------------------------------------------------------------------------
# The source acts only inside the wire, so we must know which triangles belong
# to it.  Give every triangle a label: 1 = vacuum, 2 = wire.  A triangle counts
# as "wire" if its centre lies inside the square |x| < a_wire, |y| < a_wire.
# (Since the wire's edges are mesh lines, the centre decides unambiguously.)
faces = MeshFunction("size_t", mesh, 2, 1)                  # start with label 1 everywhere
for cell in cells(mesh):
    p = cell.midpoint()                                      # centre of the triangle
    if abs(p.x()) < a_wire and abs(p.y()) < a_wire:          # ... is inside the wire
        faces[cell] = 2

# Attach the labels to the integration measure: from now on
#   dx(1) = integral over the vacuum,  dx(2) = integral over the wire,  dx = over everything.
dx = Measure("dx", domain=mesh, subdomain_data=faces)

# Sanity check: the labelled wire should have area (2*a_wire)^2, and here
# the agreement is exact up to rounding errors.
wire_area = assemble(Constant(1.0) * dx(2))
print("Wire area from mesh  = %.4f  (exact: %.4f)" % (wire_area, (2 * a_wire)**2))

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
# A3 = 0 on the boundary of the mesh, which is the edge of the box.
# "on_boundary" is FEniCS shorthand for "every point on the edge of the mesh".
# The wire's surface is inside the domain, not on its edge: there
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
    bc.apply(b)                            # enforce A3 = 0 on the edge of the box
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
# Colour scheme: negative A3 -> blue, A3 = 0 -> white, positive A3 -> yellow.
# The colour range is made symmetric around zero so that white sits exactly at A3 = 0.
A3_colour_max = max(abs(A3_min), abs(A3_max))
# The colours reach full saturation quickly on both sides of zero, so only a
# thin band around A3 = 0 is white and even small values are clearly coloured.
# The white is made partly transparent; the opacity fades smoothly back to fully
# opaque towards blue and gold.
blue_white_yellow = matplotlib.colors.LinearSegmentedColormap.from_list(
    "blue_white_yellow",
    [(0.00, "darkblue"),     # most negative A3
     (0.40, "blue"),
     (0.50, (1, 1, 1, 0.3)), # A3 = 0: white, half transparent (red, green, blue, opacity)
     (0.60, "gold"),
     (1.00, "darkorange")])  # most positive A3

os.makedirs(out_dir + "/frames", exist_ok=True)

# The wire itself, so you can see where the current (the source) sits.
# In the 3D plot we draw it as a see-through square column made of grid lines,
# standing upright and spanning the whole A3 range.
# In the 2D plot it is simply the outline of the square.
wire_colour = "red"
# Walk once around the square, starting and ending at the same corner.
# Each side gets 5 points so the 3D column has vertical lines along the
# sides, not only at the corners.
side = np.linspace(-a_wire, a_wire, 5)
wire_x = np.concatenate([side, np.full(5, a_wire), side[::-1], np.full(5, -a_wire)])
wire_y = np.concatenate([np.full(5, -a_wire), side, np.full(5, a_wire), side[::-1]])
if surface_plot:
    wire_z = np.linspace(A3_min, A3_max, 6)               # heights of the horizontal rings
    col_x, col_z = np.meshgrid(wire_x, wire_z)            # 2D grid (position along the outline, height)
    col_y, _ = np.meshgrid(wire_y, wire_z)

frame_files = []
for i in range(n_frames):
    fig = plt.figure(figsize=(7, 6))
    if surface_plot:
        # 3D surface: height = A3(x, y)
        ax = fig.add_subplot(projection="3d")
        ax.plot_trisurf(x, y, A3_frames[:, i], triangles=triangles,
                        cmap=blue_white_yellow, vmin=-A3_colour_max, vmax=A3_colour_max,
                        linewidth=0, antialiased=False)
        ax.plot_wireframe(col_x, col_y, col_z, color=wire_colour,   # the wire
                          linewidth=0.8, rstride=1, cstride=2)
        ax.set_zlim(A3_min, A3_max)
        ax.set_zlabel("A3")
    else:
        # flat colour map: colour = A3(x, y)
        ax = fig.add_subplot()
        colours = ax.tripcolor(x, y, triangles, A3_frames[:, i],
                               shading="gouraud", cmap=blue_white_yellow,
                               vmin=-A3_colour_max, vmax=A3_colour_max)
        ax.plot(wire_x, wire_y, color=wire_colour, linewidth=1.5)  # outline of the wire
        fig.colorbar(colours, ax=ax, label="A3")
        ax.set_aspect("equal")
    ax.set_xlim(-L_box, L_box)
    ax.set_ylim(-L_box, L_box)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("$A_3(x, y)$ t = %.2f" % t_frames[i])
    filename = out_dir + "/frames/frame_%03d.png" % i
    fig.savefig(filename, dpi=300)
    plt.close(fig)
    frame_files.append(filename)

# Glue the frames into an animated GIF, 10 frames per second (100 ms per frame).
images = [Image.open(filename) for filename in frame_files]
images[0].save(out_dir + "/wave.gif", save_all=True, append_images=images[1:], duration=100, loop=0)
print("Animation saved to %s/wave.gif" % out_dir)
