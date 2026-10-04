# The PDE in weak form: a short introduction

This note explains section 4 of `wave_current_fenics.py`: how a differential
equation is turned into something a computer can solve with the finite element
method (FEM). Read it before or alongside the code.

## 1. Why we need a "weak" form

We want $A_3(x, y)$ on a mesh of triangles. The simplest approximation is a
function that is **linear on each triangle** and continuous across edges, like
a surface made of flat triangular tiles.

Problem: the wave equation contains $\nabla^2 A_3$, i.e. second derivatives.
On a flat tile the second derivatives are zero, and across the edges they do
not exist. So the approximate $A_3$ cannot satisfy the equation point by point.

The fix: do not demand the equation at every point. Demand it only **on
average**, weighted by a family of test functions $v$. That weaker requirement
needs only first derivatives, which our tiled surface does have.

## 2. Deriving the weak form (static example)

Start with the simpler, time-independent problem on a domain $\Omega$:

```math
-\nabla^2 A_3 = g \quad \text{in } \Omega, \qquad A_3 = 0 \quad \text{on } \partial\Omega .
```

**Step 1: multiply by a test function $v$ and integrate** over the domain:

```math
-\int_\Omega \nabla^2 A_3 \, v \, dx = \int_\Omega g \, v \, dx .
```

**Step 2: integrate by parts** (Green's identity) to move one derivative from
$A_3$ onto $v$:

```math
\int_\Omega \nabla A_3 \cdot \nabla v \, dx - \oint_{\partial\Omega} \frac{\partial A_3}{\partial n} \, v \, ds = \int_\Omega g \, v \, dx .
```

**Step 3: drop the boundary term.** We choose $v = 0$ wherever $A_3$ is
prescribed (the outer circle), so the boundary integral vanishes:

```math
\boxed{\int_\Omega \nabla A_3 \cdot \nabla v \, dx = \int_\Omega g \, v \, dx \qquad \text{for every test function } v}
```

This is the **weak form**. Only first derivatives appear. If $A_3$ is smooth,
it is equivalent to the original equation.

## 3. From the weak form to a matrix

Each mesh vertex $j$ has a "hat" function $\varphi_j$: equal to 1 at vertex
$j$, 0 at all other vertices, linear on each triangle.

```
        phi_j
          /\
         /  \
   _____/    \_____
      j-1  j  j+1
```

Write the unknown as a sum of hats, $A_3 = \sum_j a_j \varphi_j$, where $a_j$
is the value of $A_3$ at vertex $j$. Use each hat as a test function,
$v = \varphi_i$. The weak form becomes one linear equation per vertex:

```math
\sum_j K_{ij} \, a_j = b_i, \qquad
K_{ij} = \int_\Omega \nabla\varphi_i \cdot \nabla\varphi_j \, dx, \qquad
b_i = \int_\Omega g \, \varphi_i \, dx .
```

$K$ is the **stiffness matrix**. Each hat overlaps only its neighbours, so $K$
is mostly zeros (sparse) and fast to solve. We will also need the **mass
matrix**

```math
\mathcal{M}_{ij} = \int_\Omega \varphi_i \, \varphi_j \, dx ,
```

which comes from terms without derivatives.

## 4. Adding time: the wave equation

Our actual equation is

```math
\frac{1}{c^2} \frac{\partial^2 A_3}{\partial t^2} - \nabla^2 A_3 = -f(t) \, \chi_\text{wire}(x, y),
\qquad f(t) = \frac{4\pi}{c} J_0 \sin \omega t ,
```

where $\chi_\text{wire}$ is 1 inside the wire and 0 outside, and $J_0$ is the
amplitude of the current density. Write
$A_3^{n+1}, A_3^{n}, A_3^{n-1}$ for the solution at times $t + \Delta t,\ t,\ t - \Delta t$
(`A3_new`, `A3_now`, `A3_old` in the code). Replace the time derivative by a
central difference, and the Laplacian by a weighted average of the three time
levels (this averaging makes the scheme stable for any $\Delta t$):

```math
\frac{A_3^{n+1} - 2A_3^{n} + A_3^{n-1}}{(c\,\Delta t)^2}
- \nabla^2 \left( \frac{A_3^{n+1} + 2A_3^{n} + A_3^{n-1}}{4} \right)
= -f(t) \, \chi_\text{wire} .
```

Now apply steps 1 to 3 to this equation. Keep the unknown $A_3^{n+1}$ on the
left, move the known $A_3^{n}$, $A_3^{n-1}$ and $f$ to the right:

```math
\underbrace{\frac{1}{(c\,\Delta t)^2} \int_\Omega A_3^{n+1} v \, dx
+ \frac14 \int_\Omega \nabla A_3^{n+1} \cdot \nabla v \, dx}_{\text{left side: } a}
```

```math
= \underbrace{\frac{1}{(c\,\Delta t)^2} \int_\Omega \left(2A_3^{n} - A_3^{n-1}\right) v \, dx
- \frac14 \int_\Omega \nabla\left(2A_3^{n} + A_3^{n-1}\right) \cdot \nabla v \, dx
- \int_\text{wire} f \, v \, dx}_{\text{right side: } L}
```

With hats, the left side gives the matrix

```math
M = \frac{1}{(c\,\Delta t)^2} \, \mathcal{M} + \frac14 K .
```

It does not change in time, so the script builds it once. At every step only
the right-hand side vector $b$ is rebuilt and $M a^{n+1} = b$ is solved.

## 5. The same thing in FEniCS

FEniCS lets you type the weak form almost as written above:

| Math | Code |
|---|---|
| unknown $A_3^{n+1}$ | `A3 = TrialFunction(V)` |
| test function $v$ | `v = TestFunction(V)` |
| $\int_\Omega \dots  dx$ | `* dx` |
| $\int_\text{wire} \dots  dx$ | `* dx(2)` (triangles labelled 2) |
| $\nabla A_3 \cdot \nabla v$ | `dot(grad(A3), grad(v))` |

```python
a = (1.0 / dt_sim**2) * A3 * v * dx  +  0.25 * dot(grad(A3), grad(v)) * dx
L = (1.0 / dt_sim**2) * (2 * A3_now - A3_old) * v * dx \
    - 0.25 * dot(grad(2 * A3_now + A3_old), grad(v)) * dx \
    - f * v * dx(2)
```

The code has no `c` and no `J0` in these lines. The script works in
"simulation units" (section 1c of the script), in which lengths are measured
in wire radii and times in `R_wire / c`. In these units $c = 1$ and $J_0 = 1$,
so $1/(c\,\Delta t)^2$ becomes `1.0 / dt_sim**2` and $f(t) = 4\pi \sin(\omega t)$.

`assemble(a)` turns `a` into the matrix $M$; `assemble(L)` turns `L` into the
vector $b$. Choosing hats, computing integrals triangle by triangle and adding
them up is all done for you.

## 6. Check your understanding

1. Why can a piecewise-linear $A_3$ not satisfy $-\nabla^2 A_3 = g$ point by point?
2. In step 3, why may we drop the boundary term? What would change if the
   boundary condition were $\partial A_3 / \partial n = 0$ instead of $A_3 = 0$?
3. Which term of `a` becomes the mass matrix and which the stiffness matrix?
4. Why is the source integrated over `dx(2)` and not `dx`?
