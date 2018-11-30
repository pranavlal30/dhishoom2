#import sys
#from pyspark import SparkContext
#from pyspark.streaming import StreamingContext
#from pyspark.streaming.kafka import KafkaUtils
from kafka import KafkaProducer
from scipy.io import wavfile
import glob
import time

bootstrap_servers=['localhost:9092']
topic = 'test2'


def publish_message(producer_instance, topic_name, value):
    try:
        producer_instance.send(topic_name, key='Sensor1', value=value.dumps())
        producer_instance.flush()
        print('Message published successfully.')
    except Exception as ex:
        print('Exception in publishing message')
        print ex


def connect_kafka_producer():
    _producer = None
    try:
        _producer = KafkaProducer(bootstrap_servers=bootstrap_servers)
    except Exception as ex:
        print('Exception while connecting Kafka')
        print(ex)
    finally:
        return _producer

def iterate_files():
    #list_of_files = glob.glob('/Users/pranavlal/Documents/Big_Data/Project/dhishoom/VGG/wavefiles/eval/gunshots/*')
    list_of_files = glob.glob('/Users/pranavlal/Documents/Big_Data/Project/dhishoom/VGG/wavefiles/eval/negative/*')
    #list_of_files = glob.glob('/Users/pranavlal/Documents/Big_Data/Project/dhishoom2/TestAudio/*')
    kafka_producer = connect_kafka_producer()
    for wave_file in list_of_files:
    	print wave_file
        sr, wav_data = wavfile.read(wave_file)
        if len(wav_data) == 220500:
        	publish_message(kafka_producer, topic, wav_data)
        	time.sleep(10)

    
if __name__ == "__main__":

    iterate_files()
    #kafka_producer = connect_kafka_producer()

    #wave_file = '/Users/pranavlal/Documents/Big_Data/Project/dhishoom/VGG/wavefiles/eval/gunshots/OuvUTzK6Ix4_40.000.wav'
    #wave_file = '/Users/pranavlal/Documents/Big_Data/Project/dhishoom/VGG/wavefiles/eval/gunshots/Adwb_rxQRXI_590.000.wav'
    #sr, wav_data = wavfile.read(wave_file)
    #publish_message(kafka_producer, topic, wav_data)
    #print sr



    

