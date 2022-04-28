# Ginger Anchors

Implementation and Extensions of [Anchors: High-Precision Model-Agnostic Explanations](https://homes.cs.washington.edu/~marcotcr/aaai18.pdf).
The re-implementation was done in the iML lecture (WS21/22).


## Contributions
* Implementation of Anchors via the Bottom-up construction approach.
* Implementation of Beam Search on top of the Bottom-up construction.
* Simple interfaces with well-splitted main functions.
* Analysis on how 𝐵, 𝛿 and 𝜖 influence the results.
* SMAC3 as alternative anchor finder.


## References
* Paper Reference: https://homes.cs.washington.edu/~marcotcr/aaai18.pdf
* Code Reference: https://github.com/marcotcr/anchor


## Installation

*Note: swig is needed to install SMAC3. See [installation instructions](https://automl.github.io/SMAC3/master/pages/getting_started/installation.html).*

Create a conda environment
```bash
$ conda create -n GingerAnchors python=3.9
$ conda activate GingerAnchors
$ pip install ginger-anchors
```


## Usage

You can get an explanation by setting up an Explainer and calling one of three search functions.

```python
exp = Explainer(X_df)
anchor = exp.explain_bottom_up(instance, model, tau=0.95)
print(anchor.get_explanation())
```

For a more detailed example, see [src/main.py](https://github.com/LUH-AI/ginger-anchors/blob/main/src/main.py).


## Analysis

The plots were too large to put them into this repository. Please download them from [seafile](https://seafile.cloud.uni-hannover.de/d/1ba613292c774f8c87dc/).
To reproduce the raw data, run:

```bash
ginger-anchors> python src/analysis.py
```
A preview can be found in our writeup: [analysis.md](https://github.com/automl-classroom/iml-ws21-projects-ginger-anchors/blob/main/analysis.md)


## Authors

[Jim Rhotert](https://github.com/Dschimm) & [Julian Bilsky](https://github.com/julianbil)

