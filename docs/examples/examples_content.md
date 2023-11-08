- First example shows how to use the `BeatData` class to make a dataset for analyzing heartbeats. This class processes the provided ECG records and generates a dataset comprising the extracted heartbeat waveforms, computed features, and their corresponding labels. This type of dataset can be utilized for the heartbeat classification task. For this purpose, any type of machine learning algorithm can be used. The example shows how to include predefined or custom features to the dataset. This requires that an instance of the class `BeatInfo` be passed as an argument.

    **{doc}`Heartbeat dataset <heartbeat>`**

    <br>


- The second example demonstrates how to use the `RhythmData` class provided in the `pyheartlib` to create a dataset for arrhythmia classification. The dataset will contain signals and annotations, as well as metadata about the excerpts. The metadata for each excerpt includes the record ID, the onset and offset of the excerpt on the raw signal, and the annotation. Finally, the `ECGSequence` class will be used to generate batches of sample data from the information available in the dataset.


    **{doc}`Arrhythmia dataset <arrhythmia>`**

    <br>


- The third example shows how to use the ‍‍‍`RpeakData‍‍‍` class to make a dataset. This class is similar to the `RhythmData` class. However, unlike the `RhythmData` class, it does not provide only one label for each excerpt. Instead, it provides a list of labels for each excerpt. Each element in this annotation list corresponds to a subsegment of the excerpt. The `ECGSequence` class uses the created dataset to generate batches of sample data.

   **{doc}`Rpeak dataset <rpeak>`**

    <br>


- Example 4 contains the code for training a deep learning model for the R-peak detection task. In this example, the electrocardiogram signals are segmented into ten-second ecxerpts. Each excerpt is associated with an annotation list of length 600, containing zeros and ones. A subsegment of an excerpt that contains an R-peak corresponds to 1 in the annotation list. To achieve this goal, the `RpeakData` class is used to create the required train, validation, and test datasets.  Each dataset consists of signals and annotations, as well as metadata about the excerpts. The metadata for each excerpt includes the record ID, the onset and offset of the excerpt on the raw signal, and the annotation.

    **{doc}`R-peak detection <rpeak_detection>`**
