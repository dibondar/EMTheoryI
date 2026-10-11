# EM Theory I: computer examples

Numerical examples for the EM Theory I course. Each example is a self-contained
Python script that you read from top to bottom: the comments explain the physics
and every numerical step.

**Start here:** install the software as described in [INSTALL.md](INSTALL.md).
All examples use the same `fenicsx` conda environment.

## Examples

| Folder | What it shows | Main ideas |
|---|---|---|
| [01_radiating_wire](01_radiating_wire/README.md) | Waves sent out by an oscillating current in a wire (a 5 GHz WiFi antenna); round and square wire | wave equation for `A`, Lorenz gauge, `B = curl A`, finite elements in space and time |

Each folder has its own README with the equations, how to run the example,
and exercises. Every script writes its pictures and data to a `results` folder
next to it.

## Background notes

- [WEAK_FORM.md](WEAK_FORM.md): how a PDE is turned into its weak form for the
  finite element method.
