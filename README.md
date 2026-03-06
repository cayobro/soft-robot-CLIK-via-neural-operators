# Infinite-Dimensional Closed-Loop Inverse Kinematics for Soft Robots via Neural Operators

This repository contains the code accompanying the paper:  
**Infinite-Dimensional Closed-Loop Inverse Kinematics for Soft Robots via Neural Operators**  

by Carina Veil, Moritz Flaschel, Ellen Kuhl, Cosimo Della Santina.

Contact: cveil@stanford.edu

If you use this code in your research, please cite us: [https://arxiv.org/abs/2510.03547](https://arxiv.org/abs/2602.18655)  
The entire dataset for the shape library with 1,000,000 samples can be downloaded here: [https://doi.org/10.5281/zenodo.18802371](https://doi.org/10.5281/zenodo.18802371)

---

## Overview
For underactuated soft robots,not all configurations are attainable through control action, making kinematic inversion extremely challenging. Extensions of CLIK address this by introducing end-to-end mappings from actuation to task space for the controller to operate on, but typically assume finite dimensions of the underlying virtual configuration space. In this project, we formulate CLIK in the infinite-dimensional domain to reason about the entire soft robot shape while solving tasks. We do this by composing an actuation-to-shape map with a shape-to-task map, deriving the differential end-to-end kinematics via an infinite-dimensional chain rule, and thereby obtaining a Jacobian-based CLIK algorithm. Since this actuation-to-shape mapping is rarely available in closed form, we propose to learn it using differentiable neural operator networks.

![Graphical Abstract](img/Abstract.png)

---


## File Structure and Features
* a2s: Contains everything that is necessary to understand how the deep operator network was trained (a2s stands for *actuation-to-shape*)
* training-results: Pre-trained deep operator network based on this [dataset](https://doi.org/10.5281/zenodo.18802371).
* cc_clik: CLIK algorithm for constant curvature segment (toy example).
* cc_kinematics: Kinematics and Jacobian for constant curvature segment.
* neural_clik: Neural CLIK algorithm using the learned neural operator model.
* neural_kinematics: ``Kinematics'' and Jacobian based on the neural operator model.
* tasks: Different task formulatios.
* utils: Misc functions used in the code.

Please note that the respective paths need to be adapted in the `load_model` and `load_shape_library` functions in `utils.py`

---
