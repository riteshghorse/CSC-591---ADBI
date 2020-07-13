from pyspark import SparkConf, SparkContext
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils
import operator
import numpy as np
import matplotlib.pyplot as plt

def main():
    conf = SparkConf().setMaster("local[2]").setAppName("Streamer")
    sc = SparkContext(conf=conf)
    ssc = StreamingContext(sc, 10)   # Create a streaming context with batch interval of 10 sec
    ssc.checkpoint("checkpoint")

    pwords = load_wordlist("positive.txt")
    nwords = load_wordlist("negative.txt")
   
    counts = stream(ssc, pwords, nwords, 100)
    make_plot(counts)


def make_plot(counts):
    """
    Plot the counts for the positive and negative words for each timestep.
    Use plt.show() so that the plot will popup.
    """
    # YOUR CODE HERE
    #ntime = len(counts)
    #time = [i for i in range(ntime/2)]
    positives = [item[0][1] for item in counts if not (not item)]
    negatives = [item[1][1] for item in counts if not (not item)]
    plt.plot(positives, 'bo-', label='Positives')
    plt.plot(negatives, 'go-', label='Negatives')
    plt.xlabel("Time step")
    plt.ylabel("Word count")
    plt.legend(loc='upper left')
    plt.show()



def load_wordlist(filename):
    """ 
    This function should return a list or set of words from the given filename.
    """
    # YOUR CODE HERE
    with open(filename) as f:
        words = f.readlines()
    word_list = [word.strip() for word in words]
    return word_list 
   
def updateFunction(newValues, last_sum):
    if last_sum == None:
        last_sum = 0
    return sum(newValues, last_sum)

def stream(ssc, pwords, nwords, duration):
    kstream = KafkaUtils.createDirectStream(
        ssc, topics = ['twitterstream'], kafkaParams = {"metadata.broker.list": 'localhost:9092'})
    tweets = kstream.map(lambda x: x[1])

    # Each element of tweets will be the text of a tweet.
    # You need to find the count of all the positive and negative words in these tweets.
    # Keep track of a running total counts and print this at every time step (use the pprint function).
    # YOUR CODE HERE
    words = tweets.flatMap(lambda line: line.split(" "))

    pairs = words.map(lambda word: ("positive", 1) if word in pwords else ("positive", 0)).union(words.map(lambda word: ("negative", 1) if word in nwords else ("negative", 0)))

    word_counts = pairs.reduceByKey(lambda x,y: x+y)

    

    word_counts = pairs.updateStateByKey(updateFunction) 
    word_counts.pprint()

    # Let the counts variable hold the word counts for all time steps
    # You will need to use the foreachRDD function.
    # For our implementation, counts looked like:
    #   [[("positive", 100), ("negative", 50)], [("positive", 80), ("negative", 60)], ...]
    counts = []
    # YOURDSTREAMOBJECT.foreachRDD(lambda t,rdd: counts.append(rdd.collect()))
    word_counts.foreachRDD(lambda t,rdd: counts.append(rdd.collect()))

    
    ssc.start()                         # Start the computation
    ssc.awaitTerminationOrTimeout(duration)
    ssc.stop(stopGraceFully=True)

    return counts


if __name__=="__main__":
    main()

