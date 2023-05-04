# Beat Detection Example

This directory contains the code for training a heartbeat detection model. As the task is not complex, it can be trained with high accuracy on few samples.

The ECG signals are segmented into 10 seconds ecxerpts. Each excerpt has a label of length 100 containing zeros and ones. For each 100ms segment the entry is one if it is a peak, otherwise it is zero.

The following steps prepares data and trains model.

Step 1: data_preparaton.py

Step 2: train.py

Step 3: inference.py

