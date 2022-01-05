# Ginger Anchors

Implementation and Extensions of [Anchors](https://homes.cs.washington.edu/~marcotcr/aaai18.pdf)

### Submission Deadline: Feb. 15th 2022, 00:00

## Goals
 * Reproduction (10): Implement Anchors via the Bottom-up Construction approach.
 * Simplification (5): The published code is hard to read and optimized to view the results.
  Write simple interfaces and extract the main functions accordingly.
 * Extension (10): Implement Beam Search on top of the Bottom-up Construction.
 * Analysis (5): Perform an analysis on how 𝐵, 𝛿 and 𝜖 influence the results.
 * Alternative Optimizer (10): Replace Beam Search/Bottom-up Construction with SMAC
(Bayesian Optimization)

Additionally:
* clean code
* well documented
* unit tested
* all requirements well documented (use requirements.txt)
* Installation instructions (in this README.md)
* If feasible, run your experiments with several random seeds. Try to create reproducabile results.

References:
* Paper Reference: https://homes.cs.washington.edu/~marcotcr/aaai18.pdf
* Code Reference: https://github.com/marcotcr/anchor

## Installation

Create a conda environment
  ```bash
  conda env create
  conda activate ginger-anchors
  ```
  
 ...


## Authors

[Jim Rhotert](https://github.com/Dschimm) & [Julian Bilsky](https://github.com/julianbil)

