
# Analysis of hyperparameters 

## Global influence of B, delta and epsilon

These plots portray the influence of B, delta and epsilon (rows) on anchors coverage, precision and search runtime (columns). Each plots' data was gathered on a single instance across 4 different seeds. We used the wheat seeds dataset ([source](https://archive.ics.uci.edu/ml/datasets/seeds)).

The analysis data was acquired by running grid search with the following ranges:

  * Delta in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
  * Epsilon in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
  * B in [1, 2, 3, 4, 5, 6, 7]

Running grid search took roughly 24 hours on an AMD EPYC™ 7532 with 500 GB RAM :)

You can click on the images to enlarge them.

<img src="https://github.com/automl-classroom/iml-ws21-projects-ginger-anchors/blob/main/analysis/2d_parameter_effect_3.png" width="500" height="500"></img>
<img src="https://github.com/automl-classroom/iml-ws21-projects-ginger-anchors/blob/main/analysis/2d_parameter_effect_111.png" width="500" height="500"></img>
<img src="https://github.com/automl-classroom/iml-ws21-projects-ginger-anchors/blob/main/analysis/2d_parameter_effect_155.png" width="500" height="500"></img>


As we can see, the biggest effect is caused by altering B or epsilon.

 * Greater epsilon leads to lower runtime and slightly higher precision.
 * Greater B leads to higher coverage and higher precision but also to higher runtime.

Delta did not have any effect whatsoever, at least not in the bounds we tested.

A more detailed look into the performance of different configurations can be acquired by looking at our 3D plots:
[Click here](https://seafile.cloud.uni-hannover.de/library/69a7cea5-bd5c-4d84-8e58-d0fc141feeb8/IML%20Analysis/images)
