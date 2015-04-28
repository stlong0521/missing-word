########################################################################################
# Author: Tianlong Song
# Name: LocationAndFillingCrossValidation.py
# Description: Locating and filling the missing word, working on cross validation
# Date created: 04/24/2015
########################################################################################
from random import randint
import numpy

#---------------------------------------------------------------------------------------
# Function: Main function
#---------------------------------------------------------------------------------------
def main():
    # Global settings
    trainSize = 30000000
    testSize = 10000
    M = 1       # Maximum distance considered
    gamma = 0.01
    
    # Initialization
    vocab = {}   # vocabulary for high-frequency words
    featureTable = {}   # Bigram with particular distance
    triGramTable = {}   # Trigram table
    
    # Read high-frequency words from vocabulary file
    fVocab = open('data/vocabulary-14216.txt','r')
    while True:
        line = fVocab.readline()
        if line=='':
            break
        words = line.split()
        vocab[words[0]] = int(words[1])
    vocab['UNKA'] = 58417315
    print 'High-frequency vocabulary size: ' + str(len(vocab))
    fVocab.close()
    
    # Statistics collection
    f = open('data/train_v2.txt','r')
    cnt = 0
    while cnt<trainSize:
        line = f.readline()
        cnt = cnt + 1
        words = line.split()
        for k in xrange(0,len(words)):
            if words[k] not in vocab:
                words[k] = 'UNKA'
        for i in xrange(0,len(words)):
            for m in xrange(0,M+1):
                if i+m+1>=len(words):
                    break
                key = words[i] + ' ' + words[i+m+1] + ' ' + str(m)
                if key in featureTable:
                    featureTable[key] = featureTable[key] + 1
                else:
                    featureTable[key] = 1
            if i<=len(words)-3:
                key = words[i] + ' ' + words[i+1] + ' ' + words[i+2]
                if key in triGramTable:
                    triGramTable[key] = triGramTable[key] + 1
                else:
                    triGramTable[key] = 1
    print 'Feature table size: ' + str(len(featureTable))
    print 'Trigram table size: ' + str(len(triGramTable))
    
    # Missing word location and filling
    line = f.readline()
    cnt = 0
    cntLocation = 0
    cntFilling= 0
    while cnt<testSize:
        line = f.readline() 
        
        # Sentence preprocessing
        words = line.split()
        if len(words)<=2: # Sentence too short, ignore!
            continue
        cnt = cnt + 1
        location = randint(1,len(words)-2)
        trueWord = words[location]
        del words[location]
        for k in xrange(0,len(words)):
            if words[k] not in vocab:
                words[k] = 'UNKA'
        
        # Missing word location
        score = numpy.zeros(len(words)-1)
        for k in xrange(1,len(words)):
            key = words[k-1] + ' ' + words[k] + ' ' + str(0)
            if key in featureTable:
                numNeg = featureTable[key]
            else:
                numNeg = 0
            key = words[k-1] + ' ' + words[k] + ' ' + str(1)
            if key in featureTable:
                numPos = featureTable[key]
            else:
                numPos = 0
            if numNeg+numPos!=0:
                if words[k-1]=='UNKA' or words[k]=='UNKA':
                    score[k-1] = 1.0*numPos/(numNeg+numPos) - 1.0*numNeg/(numNeg+numPos)
                else:
                    score[k-1] = 1.0*numPos**(1+gamma)/(numNeg+numPos) - 1.0*numNeg**(1+gamma)/(numNeg+numPos)
        locationPredicted = numpy.argmax(score) + 1
        if locationPredicted==location:
            cntLocation = cntLocation + 1
        
        # Missing word filling
        maxWord = 'UNKA'
        maxScore = 0
        for word in vocab:
            if word=='UNKA':
                continue
            score = 0
            key = words[location-1] + ' ' + word + ' ' + str(0)
            if key in featureTable:
                score = score + 0.25*featureTable[key]/vocab[words[location-1]]
                if location-2>=0:
                    key = words[location-2] + ' ' + words[location-1] + ' ' + word
                    if key in triGramTable:
                        score = score + 0.5*triGramTable[key]/featureTable[words[location-2]+' '+words[location-1]+' '+str(0)]
            key = word + ' ' + words[location] + ' ' + str(0)
            if key in featureTable:
                score = score + 0.25*featureTable[key]/vocab[words[location]]
                if location+1<len(words):
                    key = word + ' ' + words[location] + ' ' + words[location+1]
                    if key in triGramTable:
                        score = score + 0.5*triGramTable[key]/featureTable[words[location]+' '+words[location+1]+' '+str(0)]
                key = words[location-1] + ' ' + word + ' ' + words[location]
                if key in triGramTable:
                    score = score + 1.0*triGramTable[key]/featureTable[words[location-1]+' '+words[location]+' '+str(1)]
            if score>maxScore:
                maxScore = score
                maxWord = word
        if maxWord==trueWord:
            cntFilling = cntFilling + 1
                
    f.close()
    print 'Missing word location accuracy: ' + str(1.0*cntLocation/testSize)
    print 'Missing word filling accuracy: ' + str(1.0*cntFilling/testSize)   
    
if __name__ == '__main__':
    main()