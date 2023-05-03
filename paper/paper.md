title: 'pyecg: A Python package for processing and modeling electrocardiogram signals'

# Summary

This paper introduces ``pyecg``, a python package for processing electrocardiogram (ECG) signals. This package faciliates three necessary tasks for analysing ECG signals. The main functionality of this software is processing signals for heartbeat detection, heartbeat classification, and arhythmia classification. Using this package researchers can focus on these tasks without the burden of designing data processing pipelines. The package supports both raw signals and computed features for model building. Therefore both traditional machine learning models and deep learning models can be trained for each task. The first release of this software works well with keras and tensorflow libraries and supports MIT-BIH Arrhythmia Database format. The package has optional preprocessing functionalities to remove noise and baseline wander from the raw signals.

# Example

An example model is provided with this package which demonstrates a deep learning model designed using keras library for heartbeat detection task. As this task is less complex compared to heartbeat and arhythmia classification tasks, it can be trained with high accuracy using the availble public datasets.

![Heartbeat detection example](mis.png "Heartbeat detection example")




# References

Moody, George B., and Roger G. Mark. "The impact of the MIT-BIH arrhythmia database." IEEE engineering in medicine and biology magazine 20.3 (2001): 45-50.

Goldberger, Ary L., et al. "PhysioBank, PhysioToolkit, and PhysioNet: components of a new research resource for complex physiologic signals." circulation 101.23 (2000): e215-e220.

Abadi, Martín, et al. "Tensorflow: a system for large-scale machine learning." Osdi. Vol. 16. No. 2016. 2016.

@misc{chollet2015keras,
  title={Keras},
  author={Chollet, Fran\c{c}ois and others},
  year={2015},
  howpublished={\url{https://keras.io}},
}

