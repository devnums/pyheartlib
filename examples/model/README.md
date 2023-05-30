# Example: Beat Detection

This example contains the code for training a deep learning model for the heartbeat detection task. As the task is not complex, it can be trained with high accuracy on a small amount of data.
In this example, the electrocardiogram signals are segmented into ten seconds ecxerpts. Each excerpt corresponds to an annotation list of length 100 containing zeros and ones. A subsegment with an R-peak corresponds to one in the annotation list.

The following steps prepare the data and train the model.


[Data preparation](data_preparation.py) <br>
[Training](train.py) <br>
[Inference](inference.py) <br>

[Result](result.txt) <br>

![Example: heartbeat detection using deep learning.](plots/mis.png)