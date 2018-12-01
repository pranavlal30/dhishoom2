# dhishoom
Real time gunshot detection system


Files:

1. Consumer.py - Test Kafka consumer. Isn't used in the final pipeline
2. PlotAudio.ipynd - Dashboard code
3. Producer.py - Kafka Producer code. Iterates through a test audio set and published to our Kafka topic.
4. StreamConsumer.py - Spark streaming logic. Responsible for consuming DStreams and passing each audio file to the classification model.
5. AudioClassifier - VGG model Audio Classification code.
6. CNN - Audio classification code with helper functions

