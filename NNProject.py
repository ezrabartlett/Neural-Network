import numpy
import random
import math
from NeuralNetwork import *
from bdflib import reader
import matplotlib.pyplot as plot

# For converting the hex BDF lines into bit arrays
def hex_to_binary(hex_number):
    updated_hex_number = hex_number[0:4]
    scale = 16
    num_of_bits = 8
    converted = bin(int(updated_hex_number, scale))[2:].zfill(num_of_bits)
    return converted

# Prints out the character array in a readable form
def printChar(bitArray):
    strn = ""
    for i in range(1,len(bitArray)):
        strn+=str(bitArray[i])+" "
        if(i%9==0):
            print(strn)
            strn = ""

# Flip bits randomly with a 10% chance
def makeNoise(bits):
    noiseData = []
    for bit in bits:
        if random.uniform(0, 1)<.5:
            bit = abs(1-bit)
        noiseData.append(bit)
    return noiseData

def singleCharTrain():
    for i in range(1,100):
        errorData.append(.5*(1-Brain.getError(charSolutions[0])))
        iterationData.append(i)
        Brain.train([charData[0]],charSolutions[0],1)


Brain = NeuralNetwork([126,64,26])

alphabet = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]

# These will hold the character bit arrays and relevant training solutions
charData = []
charSolutions = []

errorData = []
iterationData = []

# Read the bdf data from the files and convert it into useable bit arrays
def getData():
    with open("ie9x14u.bdf", "rb") as handle:
        font = reader.read_bdf(handle)
        for i,char in enumerate(alphabet):
            charData.append([])
            char = font[ord(char)].get_data()
            for c in range(0,len(char)):
                ba = hex_to_binary(char[c])
                for j in range(0,len(ba)):
                    charData[i].append(int(ba[j]))
                charData[i].append(int(0))
            while(len(charData[i])<126):
                charData[i].append(0)

            #charData[i].append()
            solution = [0]*26
            solution[i]=1
            charSolutions.append(solution)

# Function for training the network with the font data
def trainForCharData(n,k):
    iterations = 0
    initialError = 0
    # Calculate starting error (typically around 50%)
    for i in range(0,25):
        Brain.forwardProp(charData[i])
        initialError+=Brain.getError(charSolutions[i])
    print("Initial Error {}".format(initialError/26))
    # Pass through the entire set n times
    for i in range(1,n+1):
        totalError = 0
        print("Training {} of {}".format(i,n))
        iterationData.append(iterations)
        # Test each character
        for j,char in enumerate(charData):
            data = [char]
            solution = [charSolutions[j]]
            # Do a training pass with the individual character k times
            for c in range(1,k+1):
                Brain.train(data,solution,1)
            totalError+=Brain.getError(solution[0])
        print("Average Error: {}".format(totalError/26))
        # Gather data for plotting
        iterations+=k;
        errorData.append(totalError/26)


# Generate noisy data fo the test function
noisyCharData = []
for char in charData:
    noisyCharData.append(makeNoise(char))

# Test the network with clean data
def NetworkTest():
    for i,char in enumerate(charData):
        Brain.forwardProp(char)
        print("Expected {} - Returned {}".format(alphabet[i],alphabet[Brain.getOutput()]))

# Test the network with noisy data
def NoisyNetworkTest():
    for i,char in enumerate(noisyCharData):
        Brain.forwardProp(char)
        print("Expected {} - Returned {}".format(alphabet[i],alphabet[Brain.getOutput()]))

def main():
    getData()
    #NetworkTest()
    #trainForCharData(5,5)
    #NetworkTest()
    #NoisyNetworkTest()
    singleCharTrain()
    plot.plot(iterationData, errorData, 'ro')
    plot.axis([0, 100, 0, 1])
    plot.show()

main()
