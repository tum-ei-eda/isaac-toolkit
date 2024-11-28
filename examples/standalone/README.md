# Standalone Examples

This directory contains independent examples for the ISAAC-Toolkit. Any prerequisite tools (Compiler, ISS,...) will be provided via download to allow an easy "out-of-the-box" experience.

#### SW Build Tools

Depending on the example different tools are used to build programs:
- CMake
- Makefile
- Custom (invoke compiler directly)

#### Targets

The examples are written for one ore more simulators.

- ETISS (Needs custom crt0.S, specs & linker script)
- Spike+PK (Needs proxy kernel)
- Generic (Static analysis only, no simulation!)

#### Programs

See subdirectories! (Coremark, Dhrystone,...)

## Prerequisites

### Ubuntu Packages

Required APT packages:

```sh
sudo apt install TODO
```

### Toolchains, ISS,...

See [`SETUP.md`](SETUP.md) for more instructions!