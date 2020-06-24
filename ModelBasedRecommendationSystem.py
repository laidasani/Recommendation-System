#Needed for XGB Regressor
from sklearn.metrics import mean_squared_error

from pyspark import SparkContext
import os
import sys
import json
import numpy as np
import xgboost as xgb
import time

startTime = time.time()

sc=SparkContext("local[*]","Sample2.2")
sc.setLogLevel("ERROR")


def trainFeaturesAndLabels(trainFinalDictionary):
    trainData = []
    trainRatings = []
    for row in trainFinalDictionary:
        trainData.append(getFeatures(row))
        trainRatings.append(row[2])

    trainDataFinal = np.asarray(trainData)
    labelsFinal = np.asarray(trainRatings)

    return trainDataFinal,labelsFinal

def testFeatures(testFinalDictionary):
    testFeatures = []
    for row in testFinalDictionary:
        testFeatures.append(getFeatures(row))

    testFeaturesFinal = np.asarray(testFeatures)

    return testFeaturesFinal

def getFeatures(x):
    features = []
    features.extend(userRDD.get(x[0]))
    features.extend(businessRDD.get(x[1]))
    return features


folderPath = sys.argv[1]
testFile = sys.argv[2]
outputFile = sys.argv[3]

#folderPath = 'data/'
#testFile = 'data/yelp_val.csv'
#outputFile = './output2_2.csv'

testRDD = sc.textFile(testFile).map(lambda x: x.split(','))
testFirstRow = testRDD.first()
testTempRDD = testRDD.filter(lambda x: x != testFirstRow)
testFinalRDD = testTempRDD.map(lambda x: (x[0], x[1]))
testFinalDictionary=testFinalRDD.collect()

trainRDD = sc.textFile(os.path.join(folderPath, 'yelp_train.csv')).map(lambda x: x.split(","))
trainFirstRow = trainRDD.first()
trainTempRDD = trainRDD.filter(lambda x: x != trainFirstRow)
trainFinalRDD=trainTempRDD.map(lambda x: (x[0], x[1], float(x[2])))
trainFinalDictionary=trainFinalRDD.collect()

userTempRDD = sc.textFile(os.path.join(folderPath, 'user.json')).map(json.loads)
userRDD = userTempRDD.map(lambda x: (x["user_id"], (x["review_count"], x["average_stars"], x["useful"], x["fans"]))).collectAsMap()
businessTempRDD = sc.textFile(os.path.join(folderPath, 'business.json')).map(json.loads)
businessRDD = businessTempRDD.map(lambda x: (x['business_id'], (x['stars'], x['review_count']))).collectAsMap()

trainDataFinal,labelsFinal=trainFeaturesAndLabels(trainFinalDictionary)
testFeaturesFinal=testFeatures(testFinalDictionary)



#For XGB
#xgbInputRDD = xgb.DMatrix(trainDataFinal, label=labelsFinal)
#model = xgb.train({'eta': 0.3, 'booster': 'gbtree', 'max-depth': 15, 'objective': 'reg:linear', 'silent': 1}, xgbInputRDD, 50)

#result = model.predict(xgb.DMatrix(testFeaturesFinal))


#For XGB Regressor
model = xgb.XGBRegressor(max_depth=25, steps=30, gamma=20, learning_rate=0.2, n_estimators=150, booster = 'gbtree')
model.fit(trainDataFinal, labelsFinal)

result=model.predict(testFeaturesFinal)



with open(outputFile, 'w') as file:
    file.write("user_id, business_id, prediction\n")
    for i in range(0, len(result)):
        p1=min(5,result[i])
        prediction=max(1,p1)
        file.write(testFinalDictionary[i][0] + "," + testFinalDictionary[i][1] + "," + str(prediction) + "\n")

#RMSE Calculations Below: Comment Out When Submit on VOCAREUM
#outputRDD = sc.textFile(outputFile)
#outputFirstRow = outputRDD.first()
#outputTempRDD = outputRDD.filter(lambda x: x != outputFirstRow).map(lambda x: x.split(','))
#outputFinalRDD = outputTempRDD.map(lambda x: (((x[0]), (x[1])), float(x[2])))
#testDataRDD = testTempRDD.map(lambda x: (((x[0]), (x[1])), float(x[2])))
#rmseRDD = testDataRDD.join(outputFinalRDD).map(lambda x: (abs(x[1][0] - x[1][1])))
#rmseFinalRDD = rmseRDD.map(lambda x: x ** 2).reduce(lambda x, y: x + y)
#rmse = ((rmseFinalRDD / outputFinalRDD.count())**0.5)
#print("RMSE", rmse)

endTime = time.time()

print("Duration : ", endTime - startTime)