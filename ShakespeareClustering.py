# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 22:05:01 2019

@author: Alex Canfield, Jeremy Giese, Cameron Berry
"""
import os
from os import listdir
from os.path import isfile, join
from sklearn.feature_extraction.text import TfidfVectorizer 
import pandas as pd
import scipy.spatial.distance as distance
import os
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

# The below methods were used more in testing and to provide a different look at our measurements. 

# accepts the plays matrix and the word importance matrix as params. 
# takes each play and makes a table of the distances between each. 
# Not super useful, but will be good to include in the final document to show process we used.
def printDistanceTable(playsMatrix, vectorizerMatrix):
    #Clears the testFile to write to
    testFile = open("Table.txt", "w")
    testFile.write("")
    testFile.close()
    testFile = open("Table.txt", "a")
    
    playNameList = []
    for item in playsMatrix:
        playNameList.append(item[1].replace(".txt", ""))
        
    columns = playNameList.copy()
    
    colString = ("%30s" % "")
    for item in columns:
        colString += ("%30s" % item)
    
    testFile.write(colString + "\n")
    
    for i in range(0, len(vectorizerMatrix)):
        rowString = ("%30s" % columns[i])
        for j in range(0, len(vectorizerMatrix)):
            rowString += ("%30f" % (1 - distance.cosine(vectorizerMatrix[i], vectorizerMatrix[j])))
        testFile.write(rowString + "\n")
    testFile.close()

#method to print out all the members of each cluster
#Takes the kmeans.labels_ as the first parameter and the plays matrix as the second
def printClusterMembers(clusterNumberList, playsMatrix):
    # get a list, in order, of all the plays
    playNameList = []
    for item in playsMatrix:
        playNameList.append(item[1].replace(".txt", ""))
    
    # iterate over the plays and print out clusters.
    for clusterIdx in range(0, np.max(clusterNumberList)+1):
        print("Cluster ", clusterIdx)
        for playIdx in range(0, clusterNumberList.size):
            if clusterNumberList.item(playIdx) == clusterIdx:
                print(playNameList[playIdx])
        print()

#method to print out all the members of each cluster and their x, y positions
#Takes the kmeans.labels_ as the first parameter, the distances dictionary as second, and plays matrix as third
def clusterMembersLocations(clusterNumberList, distancesDict, playsMatrix):
    # get a list, in order, of all the plays
    playNameList = []
    for item in playsMatrix:
        playNameList.append(item[1].replace(".txt", ""))
    toReturn = "Play Name,Play X, Play Y,ClusterNumber\n"
    # iterate over the plays and print out clusters.
    for clusterIdx in range(0, np.max(clusterNumberList)+1):
        for item in distancesDict:
            if clusterNumberList.item(playNameList.index(item)) == clusterIdx:
                toReturn+=(str(item) + "," + str(distancesDict[item][0])+","+str(distancesDict[item][1])+","+str(clusterIdx)+"\n")
    return toReturn

# this will count up the frequency of all the plays in our list. This is our vectorizser
countVectorizer=TfidfVectorizer(use_idf=True) 

#put every play into a list. This list will be given to our Vectorizer.
# element in plays would be [playContent, playFileName]
plays = [] #row 0 is the play content. Row 1 is the play name

# loop to iterate through the 'Plays' directory and add the plays to the list.
directory = os.listdir("./Plays/")

# iterate over the files in the directory
for item in directory:
    playData = [] #list to add to plays with play and name in list
    
    # add the contents of the file to the list
    filePath = "./Plays/" + item
    file = open(filePath, encoding="utf8")
    playContents = file.read()
    playData.append(playContents)
    
    # add the play name to the list
    playData.append(os.path.basename(file.name))
    file.close()
    plays.append(playData)

# get an array of just the play contents for each play. This will correspond to the plays matrix which will provide the play names.
playContentArray = []
for item in plays:
    playContentArray.append(item[0])    


#we get a scipy sparse matrix from this. We are telling our vectorizer to do a count on the plays in this list
importanceMatrix = countVectorizer.fit_transform(playContentArray)

#take our sparse matrix and make it just a regular matrix
wordImportance = importanceMatrix.toarray()

#take the wordImportance Vectors and turn that into a numpy dataset. 
#Then scale that dataset down into the wordScaled matrix. Keeps all our values in a proper range.
wordFrame = pd.DataFrame(wordImportance)
wordNumpy = wordFrame.values
min_max_scaler = preprocessing.MinMaxScaler()
wordScaled = min_max_scaler.fit_transform(wordNumpy) # WordScaled is the matrix we will use for our PCA and KMeans analysis.

# =============================================================================
# The first for loop creates a dictionary which contains vectors and filenames
# The second creates a stack which contains all File Names
# The nested loop creates a dictionary which contains all combinations of plays and their distances
# =============================================================================
allVectors = {}

stackOfPlays = []
distances = {}
for item in plays:
    stackOfPlays.append(item[1].replace(".txt", ""))


#Use PCA to reduce the dimensionality of our data set. 
reduced_data = PCA(n_components=2).fit_transform(wordScaled)
# initialize kmeans with 7 clusters. This will show us a split in the large lower left cluster
kmeans = KMeans(init='k-means++', n_clusters=7, n_init=10)
kmeans.fit(reduced_data)
for i in range(0, len(reduced_data)):
    allVectors[stackOfPlays[i]] =reduced_data[i]

for i in reversed(range(0,(len(stackOfPlays)))):
    for j in range(0, len(stackOfPlays)):
        if(i!=j):
            distances[stackOfPlays[i]+", "+stackOfPlays[j]] = distance.cosine(allVectors[stackOfPlays[i]],allVectors[stackOfPlays[0]])
            
    stackOfPlays.pop()

# plotting the clusters from kmeans.
h= .02
# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
# Plot the centroids as a white X
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
plt.title('K-means clustering on Shakespeare\'s Plays \n'
          'Centroids are marked with white cross')
# TODO: label the axis of the table with values.

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()

printClusterMembers(kmeans.labels_, plays)

# Write the members of each cluster and their positions to a file named "clustersFile.csv.
# This was so that we could use excel to examine the data further.
# =============================================================================
# clusterInfo = (clusterMembersLocations(kmeans.labels_, allVectors, plays))
# clustersFile = open("clustersFile.csv", "w")
# clustersFile.write(clusterInfo)
# clustersFile.close()
# =============================================================================


