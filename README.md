# Soft Body Virtual Cutting using Projective Dynamics

## Overview

Academic prototyping project for soft body cutting using projective dynamics.
Different constraint types and cutting methods will be implemented 
using [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page) for matrix computations and [libigl](https://libigl.github.io/) for visualization and 
user interaction.

### Constraint types
- **positional**
- **edge length**
  ![edge length constrained cloth](./doc/pd-cloth-edge-length.gif)
- **deformation gradient**
  ![deformation gradient constrained bar](./doc/pd-bar-deformation-gradient.gif)
- **strain limiting**
  ![strain limiting constrained bar](./doc/pd-bar-strain-limiting.gif)

## Dependencies

- C++17 compiler
- [libigl](https://libigl.github.io/)

[libigl](https://libigl.github.io/) is included in the project using CMake's FetchContent and pulls in [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page), [glfw](https://www.glfw.org/), [Dear ImGui](https://github.com/ocornut/imgui) and [TetGen](http://wias-berlin.de/software/index.jsp?id=TetGen&lang=1) with it.

## Building

```
# Download repository
$ git clone https://github.com/Q-Minh/projective-dynamics
$ cd projective-dynamics

# Configure and build project
$ cmake -S . -B build
$ cmake --build build --target pd --config Release

# Run the program
$ ./build/Release/pd.exe
```
