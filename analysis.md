
# Analysis of hyperparameters 

## Global influence of B, delta and epsilon

These plots portray the influence of B, delta and epsilon (rows) on anchors coverage, precision and search runtime (columns). Each plots' data was gathered on a single instance across 4 different seeds.
The data was acquired by running grid search with the following ranges:

  * Delta in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
  * Epsilon in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
  * B in [1, 2, 3, 4, 5, 6, 7]

You can click on the images to enlarge them.

<img src="https://seafile.cloud.uni-hannover.de/seafhttp/files/96030b00-0b7b-4fad-bfe3-cc12a43537a2/2d_parameter_effect_3.png" width="500" height="500"></img>
<img src="https://seafile.cloud.uni-hannover.de/seafhttp/files/d0816860-0333-4c5f-ba70-4a599b14e424/2d_parameter_effect_111.png" width="500" height="500"></img>
<img src="https://seafile.cloud.uni-hannover.de/seafhttp/files/ccc4ac14-d054-40ec-b3c8-8884cc4e0eec/2d_parameter_effect_155.png" width="500" height="500"></img>


As we can see, the biggest effect is caused by altering B or epsilon.

 * Greater epsilon leads to lower runtime and slightly higher precision.
 * Greater B leads to higher coverage and higher precision but also to higher runtime.

Delta did not have any effect whatsoever, at least not in the bounds we tested.

A more detailed look into the performance of different configurations can be acquired by looking at our 3D plots:
[Click here](https://seafile.cloud.uni-hannover.de/library/69a7cea5-bd5c-4d84-8e58-d0fc141feeb8/IML%20Analysis/images)
