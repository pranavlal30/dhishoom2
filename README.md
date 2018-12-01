# dhishoom
Real time gunshot detection system


Files:

1. Consumer.py - Test Kafka consumer. Isn't used in the final pipeline
2. PlotAudio.ipynd - Dashboard code
3. Producer.py - Kafka Producer code. Iterates through a test audio set and published to our Kafka topic.
4. StreamConsumer.py - Spark streaming logic. Responsible for consuming DStreams and passing each audio file to the classification model.
5. AudioClassifier - VGG model Audio Classification code.
6. CNN - Audio classification code with helper functions

Abstract

A gunfire detection system for police departments is a very helpful use case for using a big data pipeline performing real time analysis using IoT sensor gathered data. The motivation behind this system is that more than 80% of the shootings are never detected or brought to the notice of the authorities, as noted by ShotSpotter, a Chicago based Location and Forensic Analysis firm. The  project is aimed at detecting gunshots in near-real time and send out an alert giving the location where the shot was fired. This will help authorities take timely action against such crimes.



1.Data

The dataset used is the Google’s research AudioSet. AudioSet consists of an expanding ontology of 632 audio event classes and a collection of 2,084,320 human-labeled 10-second sound clips drawn from YouTube videos. The dataset is divided in three disjoint sets: a balanced evaluation set, a balanced training set, and an unbalanced training set. These sets are stored as CSV files, with columns for the YouTube video ID,, the start and end seconds of the YouTube video analyzed, and the positive human-labelled sound classes for the video file. Since we were interested in identifying gunshots in an urban setting, we restricted our dataset to the following classes:

Positive classes:
Gunshot, Gunfire, Machine Gun, Cap Gun

Negative Classes:
Car, Vehicle horn, car horn, honking, Car alarm, Air horn, truck horn, Reversing beeps, Police car (siren), Ambulance (siren), Fire engine, fire truck (siren), Motorcycle, Traffic noise, roadway noise, Speech


2000 positive and classes, with an even balance within the subclasses were selected from the AudioSet for classification. Once the YouTube IDs for the dataset subset were identified, the script developed by Alex Nichol [10] was used to fetch the audio WAV files from YouTube.


2. Architecture:
Sensors
In a real life scenario, there would be sensors in a particular area capturing the audio signals. The audio files would then be published to a Kafka topic. Wave files gathered from various sensors would then be aggregated, and published further to the downstream pipeline for classification and storage. However, due to the limitations of resources available for the project, we plan to simulate this for demonstration purposes. This is achieved by storing a random set of audio files from our dataset, consisting of 10 second audio clips, and publishing them periodically via a Kafka Producer to our Kafka Topic. The audio files consist of an even mix between positive examples (Gunshot videos) and negative examples (Traffic sound etc.). 

Data Flow Pipeline
The 10 second audio clips, stored in .wav format, are sent to a Kafka topic, with the key in the message depicting a sensor ID, and the value would be an n-dimensional numpy array containing the audio data. The SensorID key plays an important role since it is the sole identifier of the location of the source audio. A spark streaming application running on Hadoop, using the createDirectStream in KafkaUtils package, consumes audio files in batches of 2 seconds. Since we receive audio files from a particular sensor every 10 seconds, a batch length of 2 seconds would evenly distribute the processing load of handling all sensors. Our current implementation is on a single Kafka broker, which can be horizontally scaled without impact on the existing system. Each RDD in the DStream depicting individual audio clips are sent to the classification model. The classification model returns an array of the identified sound classes and the respective probabilities for each sound class. 
Once the classification is complete, two steps take place. First, the audio file is stored in HDFS with the data partitioned at the SensorID level. Second, a CSV file is updated with the details of the audio file that include SensorID, Timestamp, Classification Flag (set to 1 for gunshots, 0 otherwise), the array of predicted classes, and the HDFS location of the stored file. This CSV file is essential for building the dashboard for live monitoring.

3. Audio Classification Model


The CNN ‘Configuration A’ was used to train which comprises of 11 weight layers. An embedding layer was added at the end of the architecture. The input to this model are is a log-mel-scale spectrogram patch. The mel-spectrogram is obtained by taking the short-time Fourier transform and mapping its spectral magnitudes onto the perceptually motivated mel-scale using a filterbank in the frequency domain. It is the starting point for computing Mel-Frequency Cepstral Coefficients (MFCCs), and a popular representation for many audio analysis algorithms including ones based on unsupervised feature learning. We compute log-scaled mel-spectrograms with 40 bands between 0-22050 Hz using a 23 ms long Hann window (1024 samples at a sampling rate of 44.1 kHz) and a hop size of equal length. Each input spectrogram patch was labeled where ‘1’ indicates gunshots and ‘0’ indicates otherwise.  
The output is the activations produced by the 128-D embedding layer, which is usually the penultimate layer when used as a part of the full model with a final classifier layer. The embeddings have a quantization and PCA transformations applied to it in order to stay compatible with YouTube - 8M project  which provides visual embedding for a large set of YouTube videos. Also, the difference in the implementation of the model for classifying the gunshots is that the training was not batch sized training as we don’t train the classifier on a million or more samples. These modified embeddings are fed to a pre-trained, open source, frame-level Youtube- 8M model to perform the final classifications.


4. Dashboard
The dashboard provides a simple interface to display whether the file being analyzed real time is a gunshot or not. The dashboard is built using matplotlib on a Jupyter notebook. It displays the sensor ID that captured the data, along with the time of data capture. It also plots the waveform of the incoming audio signal and displays a red waveform each time it detects a gunshot and blue otherwise. The dashboard is updated every 5 seconds, plotting the latest available audio form from the streaming pipeline.


Future Scope

1.Audio Classifier
A possible improvement to the project in the future would be to train the Youtube - 8M frame-level model, to classify the learnt embeddings from the VGGish model into 2 classes instead of 527 classes. Hopefully, this showcases the trained model performance better when it’s paired with a frame-level model that too is trained to classify embeddings into 2 classes.

2. Dashboard
Currently, we implemented a basic dashboard displaying only the latest audio file for a sensor and its associated class predictions. In the future, we would like to implement a more thorough dashboard including historical information of WAV files, and plotting the data on a city map to better view the impacted area. The dashboard can also be deployed on a Flask application.



Citations
[1] Karen Simonyan & Andrew Zisserman , ‘Very Deep Convolutional Networks for Large-Scale Image Recognition’  in arXiv:1409.1556 
[2] Shawn Hershey, Sourish Chaudhuri, Daniel P. W. Ellis, Jort F. Gemmeke, Aren Jansen, R. Channing Moore, Manoj Plakal, Devin Platt, Rif A. Saurous, Bryan Seybold, Malcolm Slaney, Ron J. Weiss, Kevin Wilson , ‘CNN Architectures for Large-Scale Audio Classification’ in arXiv:1609.09430 
[3]Jort F. Gemmeke, Daniel P. W. Ellis, Dylan Freedman, Aren Jansen, Wade Lawrence, R. Channing Moore,  Manoj Plakal,  Marvin Ritter, ‘Audio Set: An ontology and human-labeled dataset for audio events’  in 10.1109/ICASSP.2017.7952261
[4] Sami Abu-El-Haija, Nisarg Kothari, Joonseok Lee, Paul Natsev, George Toderici, Balakrishnan Varadarajan, Sudheendra Vijayanarasimhan , ‘YouTube-8M: A Large Scale Video Classification Benchmark’ in arXiv:1609.08675
[5]Kai Wähner, ‘Apache Kafka and MQTT: End-to-end IOT Integration’ in https://dzone.com/articles/apache-kafka-mqtt-end-to-end-iot-integration-githu
[6] Hortonworks Inc, ‘Deploying Machine Learning Models Using Structured Spark Streaming’: https://hortonworks.com/tutorial/deploying-machine-learning-models-using-spark-structured-streaming/
[7] Evan Mouzakitis, ‘Monitoring Kafka performance metrics’: https://www.datadoghq.com/blog/monitoring-kafka-performance-metrics/
[8]Tathagata Das, Matei Zaharia and Patrick Wendell, ‘Diving into Apache Spark Streaming’s Execution Model’: https://databricks.com
[9]Konstantin Shvachko, Hairong Kuang, Sanjay Radia, Robert Chansler, ‘The Hadoop Distributed File System’: 10.1109/MSST.2010.5496972
[10] Alex Nichol, ‘Downloading AudioSet’: github.com/unixpickle/audioset/tree/master/download
