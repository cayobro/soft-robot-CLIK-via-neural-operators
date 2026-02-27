# Infinite-Dimensional Closed-Loop Inverse Kinematics for Soft Robots via Neural Operators

This repository contains the code accompanying the paper:  
**Infinite-Dimensional Closed-Loop Inverse Kinematics for Soft Robots via Neural Operators**  

by Carina Veil, Moritz Flaschel, Ellen Kuhl, Cosimo della Santina.

Contact: cveil@stanford.edu

If you use this code in your research, please cite us: [https://arxiv.org/abs/2510.03547](https://arxiv.org/abs/2602.18655)  
The entire dataset for the shape library with 1,000,000 samples can be downloaded here: [https://doi.org/10.5281/zenodo.18802371](https://doi.org/10.5281/zenodo.18802371)

---

## Overview
In progress

---

## Features



---

## File Structure
* a2s: Contains everything that is necessary to understand how the deep operator network was trained.
* training-results: Pre-trained deep operator network based on this [dataset](https://doi.org/10.5281/zenodo.18802371).
* cc_clik: CLIK algorithm for constant curvature segment (toy example).
* cc_kinematics: Kinematics and Jacobian for constant curvature segment.
* neural_clik: Neural CLIK algorithm using the learned neural operator model.
* neural_kinematics: ``Kinematics'' and Jacobian based on the neural operator model.
* tasks: Different task formulatios.
* utils: Misc functions used in the code.


---
