import sys
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils
import numpy as np
import datetime
from scipy.io import wavfile
from AudioClassifier.parse_file import process_data

outdir = "/Users/pranavlal/Documents/Big_Data/Project/dhishoom/DetectedGunshots/"

def classify(rdd):
    wav_data = np.asarray(rdd.take(1), dtype=np.int16)
    outfile = ""
    if len(wav_data) > 0:
        wav_data = wav_data.reshape(220500,2)
        prediction = process_data(wav_data)        
        print prediction
        for cat in prediction:
            if "Gun" in str(cat[0]) or "Explosion" in str(cat[0]):
                print "Gunshot detected!!!!!!!!"
                outfile = outdir + "gun_" + datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S") + ".wav"
                print wav_data
                print type(wav_data)
            else:
                outfile = outdir + "safe_" + datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S") + ".wav"
                print "All Good"
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
