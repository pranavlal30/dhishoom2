import sys
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils
import numpy as np
import datetime
from scipy.io import wavfile
from AudioClassifier.parse_file import process_data
import csv

#outdir = "/Users/pranavlal/Documents/Big_Data/Project/dhishoom/DetectedGunshots/"
outdir = "/Users/pranavlal/Documents/Big_Data/Project/dhishoom2/ProcessedFiles/Sensor1/"

def updateTable(sensorName, dt, flag, prediction, filepath):
    myFile = open('/Users/pranavlal/Documents/Big_Data/Project/dhishoom2/audio_clips.csv', 'a')
    with myFile:  
        myFields = ['sensorName', 'dt', 'flag', 'filepath']
        writer = csv.writer(myFile, delimiter = "|")  
        writer.writerow([sensorName, dt, flag, prediction, filepath])
    
def classify(rdd):
    wav_data = np.asarray(rdd.take(1), dtype=np.int16)
    outfile = ""
    if len(wav_data) > 0:
        wav_data = wav_data.reshape(220500,(wav_data.size/220500))
        prediction = process_data(wav_data)        
        print prediction
        gunshot_flag = 0
        outfile = outdir + datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S") + ".wav"
        for cat in prediction:
            if "Gun" in str(cat[0]) or "Explosion" in str(cat[0]):
                print "Gunshot detected!"
                gunshot_flag = 1
                print wav_data
                print type(wav_data)
            else:
                print "All Good"
        print outfile
        updateTable("Sensor1",datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S"), gunshot_flag, prediction, outfile)
        
        if len(outfile) > 0:
            wavfile.write(outfile, 22050, wav_data)
	
if __name__ == "__main__":
    sc = SparkContext(appName="PythonStreamingDirectKafkaWordCount")
    ssc = StreamingContext(sc, 2)
    brokers = 'localhost:9092'
    topic = 'test2'
    kafkaParams = {"metadata.broker.list": brokers}
    audioStream = KafkaUtils.createDirectStream(ssc, [topic], kafkaParams, valueDecoder=lambda x: x)
    wav_data = audioStream.map(lambda x: np.loads(x[1]))
    key = audioStream.map(lambda x: x[0])
    pred = "All good"
    pred = wav_data.foreachRDD(lambda rdd: classify(rdd))
    
    #predictions = process_data(wav_data)
    ssc.start()
    ssc.awaitTermination()
