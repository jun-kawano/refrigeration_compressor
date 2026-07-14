Reciprocating Compressor Dynamic Simulation

This is a Python numerical simulation of a reciprocating compressor. It solves the non-linear, coupled ODEs governing the cylinder's thermodynamic state and the mechanical dynamics of the suction and discharge valves.

Project Structure:

  main.py: The entry point that sets up and runs the simulations.

  config.py: Stores all physical, geometric, and dynamic parameters.

  compressor_model.py: Contains the OOP representation of the Compressor and Valves, including the Penalty Method (virtual bumpers) for handling valve limits.

  solver.py: Packages the state variables and runs the ODE solver.

Installation:
Requires Python 3.8+. Install dependencies using:
pip install numpy scipy pandas matplotlib CoolProp

Usage:
Update your target conditions in config.py, then execute the simulation by running main.py.

When executed, the plotter.py file creates plots for data visualization and prints key performance metrics.
