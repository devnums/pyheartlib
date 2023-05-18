# Beat Detection

An example is provided along with this package which demonstrates a deep learning model designed using Keras library for the heartbeat detection task. As this task is less complex compared to the heartbeat and arrhythmia classification tasks, it can be trained with high accuracy using the available public datasets.

This directory contains an example code for training a heartbeat detection model. As the task is not complex, it can be trained with high accuracy on few samples.

The ECG signals are segmented into 10 seconds ecxerpts. Each excerpt has a label of length 100 containing zeros and ones. For each 100ms segment the entry is one if it is a peak, otherwise it is zero.

The following steps prepares data and trains model.

Step 1: data_preparaton.py
<code>

Step 2: train.py
<code>

Step 3: inference.py



```bash
wget https://www.physionet.org/static/published-projects/mitdb/mit-bih-arrhythmia-database-1.0.0.zip
```
