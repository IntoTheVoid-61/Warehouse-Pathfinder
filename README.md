# Warehouse Pathfinder

## Description

This repository contains a Python framework for creating structured warehouse layouts and finding optimal paths for an Autonomous Guided Vehicle (AGV) using Genetic Algorithms.
It provides an interactive GUI for warehouse generation, allowing users to define blocks, aisles, storage locations, and pick-up points.

The framework is designed for future integration with ROS2 for AGV navigation simulations.

For specific warehouse configurations and generated layouts, please refer to the examples in the example.py file.

This is a simplified and optimized version of an algorithm from the paper: https://doi.org/10.14743/apem2025.3.541


## Features
- Interactive warehouse creation with Pygame GUI
- Pick-up locations selection for AGV path optimization
- Numerical and color-coded warehouse representation
- Genetic Algorithm-based optimal path finding
- Future-ready for ROS2 integration

## Prerequisites
- Python 3.x
- Pygame
- Numpy
- Tkinter (usually included with Python)
- Optional: ROS2 if integrating with AGV simulations

## ROS2 Integration (Future)

This framework is designed to be extended into ROS2 environments.
Planned functionality includes:

- Generating warehouse layouts and exporting them as ROS2-compatible map messages

- Using generated pick-up locations as waypoints for AGV path planning

- Integration with ROS2 nodes for AGV simulation (navigation, localization, and sensor fusion)

Currently, the code provides the warehouse structure and path-finding logic in Python.
Future updates will include ROS2 packages that consume these layouts for autonomous navigation in Gazebo or real AGVs.

## Author

- Ziga Breznikar
- Mail: ziga.breznikar@student.um.si

