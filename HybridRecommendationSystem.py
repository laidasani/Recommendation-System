#Needed for XGB Regressor
from sklearn.metrics import mean_squared_error

from pyspark import SparkContext
import time
import sys
import os
import sys
import json
import numpy as np
import xgboost as xgb
import time

startTime = time.time()

sc=SparkContext("local[*]","Sample2.3")
sc.setLogLevel("ERROR")

folderPath = sys.argv[1]
testFile = sys.argv[2]
outputFile = sys.argv[3]

#folderPath = 'data/'
#testFile = 'data/yelp_val.csv'
#outputFile = './output2_2.csv'

def pearsonCoefficientDenominatorHelper(currentBusinessDifferences,candidateBusinessDifferences):
    #Denominator of W
    currentBusinessDenominator = sum([x * x for x in currentBusinessDifferences])** 0.5
    candidateBusinessDenominator = sum([x * x for x in candidateBusinessDifferences])** 0.5

    return currentBusinessDenominator, candidateBusinessDenominator

def pearsonCoefficientNumeratorHelper(currentBusinessDifferences,candidateBusinessDifferences):
    #Numerator of W
    numeratorW = sum([currentBusinessDifferences[x] * candidateBusinessDifferences[x] for x in range(len(candidateBusinessDifferences))])

    return numeratorW

def pearsonCoefficientHelper(currentBusiness,candidateBusiness,currentBusinessRatings,candidateBusinessRatings,commonUsers):
    #Ratings of currentBusiness and candidateBusiness by all common users
    currentBusinessRatingsByCommonUsers = [businessAndRatedByUsersAndRatingDictionary[currentBusiness][user] for user in commonUsers]
    candidateBusinessRatingsByCommonUsers = [businessAndRatedByUsersAndRatingDictionary[candidateBusiness][user] for user in commonUsers]

    #Averages of ALL the ratings of currentBusiness and candidateBusiness 
    currentBusinessAverage = sum(currentBusinessRatings) / float(len(currentBusinessRatings))
    candidateBusinessAverage = sum(candidateBusinessRatings) / float(len(candidateBusinessRatings))

    #Diff of rating and avg rating for active business and business candidateBusiness, but only for common users
    currentBusinessDifferences = [x - currentBusinessAverage for x in currentBusinessRatingsByCommonUsers]
    candidateBusinessDifferences = [x - candidateBusinessAverage for x in candidateBusinessRatingsByCommonUsers]

    currentBusinessDenominator, candidateBusinessDenominator= pearsonCoefficientDenominatorHelper(currentBusinessDifferences,candidateBusinessDifferences)
    numeratorW=pearsonCoefficientNumeratorHelper(currentBusinessDifferences,candidateBusinessDifferences)
    
    return numeratorW, currentBusinessDenominator, candidateBusinessDenominator

def pearsonCoefficient(x):

    currentUser = x[0]
    currentBusiness = x[1]

    #All business rated by the current user
    restBusinesses = x[2]

    weightsAndOtherDetailsList = []

    #All users that rated the currentBusiness
    usersWhoRatedCurrentBusiness = businessAndRatedByUsersDictionary[currentBusiness]

    #Ratings of currentBusiness by ALL the users that rated it
    currentBusinessRatings = [businessAndRatedByUsersAndRatingDictionary[currentBusiness][user] for user in usersWhoRatedCurrentBusiness]
    #currentBusinessRatings =  businessRatingBroadcasted.value[currentBusiness]

    for candidateBusiness in restBusinesses:
        #All users who rated candidateBusiness
        usersWhoRatedCandidateBusiness = businessAndRatedByUsersDictionary[candidateBusiness]

        #Ratings of candidateBusiness by all the users who rated it
        candidateBusinessRatings = [businessAndRatedByUsersAndRatingDictionary[candidateBusiness][user] for user in usersWhoRatedCandidateBusiness]
        #candidateBusinessRatings = businessRatingBroadcasted.value[candidateBusiness]

        #Common Users to currentBusiness and candidateBusiness
        commonUsers = usersWhoRatedCandidateBusiness.intersection(usersWhoRatedCurrentBusiness)

        #When no common users 
        if len(commonUsers) == 0:
            s1=sum(currentBusinessRatings)/len(currentBusinessRatings)
            s2=sum(candidateBusinessRatings)/len(candidateBusinessRatings)
            w1=float(s1)/s2
            w2=float(s2)/s1
            wFinal=min(w1,w2)
            weightsAndOtherDetailsList.append([candidateBusiness, wFinal, businessAndRatedByUsersAndRatingDictionary[candidateBusiness][currentUser]])
            continue

        numeratorW, currentBusinessDenominator, candidateBusinessDenominator= pearsonCoefficientHelper(currentBusiness,candidateBusiness,currentBusinessRatings,candidateBusinessRatings,commonUsers)
       
        weight=0

        #If denominator equals 0(implies numerator also equals 0) then set weight to 1 
        if(currentBusinessDenominator==0 or candidateBusinessDenominator==0):
            weight=1
        else:
            weight = numeratorW / (currentBusinessDenominator * candidateBusinessDenominator)

        weightsAndOtherDetailsList.append([candidateBusiness, weight, businessAndRatedByUsersAndRatingDictionary[candidateBusiness][currentUser]])

    #Sorted to get top 50 W
    sortedWeightsAndOtherDetailsList = sorted(weightsAndOtherDetailsList, key=lambda x: -x[1])
    return [(currentUser, currentBusiness), sortedWeightsAndOtherDetailsList[:50]]

def predictRatings(data):
    currentUser = data[0][0]
    currentBusiness = data[0][1]
    weightsList = data[1]

    numeratorP = 0
    denominatorP = 0

    for w in weightsList:
        if(w[1]>0):
            numeratorP += w[1] * float(w[2])
            denominatorP += abs(w[1])

    prediction = 3.0

    if (denominatorP != 0):
        prediction = numeratorP / float(denominatorP)

        if(prediction>5):
            prediction=5.0
        elif(prediction<1):
            prediction=1.0

    return [(currentUser), {currentBusiness: prediction}]

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


#-----------------------------------------------------Item-Based------------------------------------------------------------------------

trainRDD = sc.textFile(os.path.join(folderPath, 'yelp_train.csv')).map(lambda x: x.split(","))
trainRDDFirstRow = trainRDD.first()
trainRDDFinal = trainRDD.filter(lambda row: row != trainRDDFirstRow)

testRDD = sc.textFile(testFile).map(lambda line: line.split(","))
testRDDFirstRow = testRDD.first()
testRDDFinal=testRDD.filter(lambda row: row != testRDDFirstRow)


#RDD containing user and its ratings
userRating = trainRDDFinal.map(lambda row: (row[0], {float(row[2])})).reduceByKey(lambda a, b: a | b)
#Dictionary of the above
userRatingDictionary = dict(userRating.collect())
#Broadcasting the above dictionary
userRatingBroadcasted = sc.broadcast(userRatingDictionary)

#RDD containing business and its ratings
businessRating = trainRDDFinal.map(lambda row: (row[1], {float(row[2])})).reduceByKey(lambda a, b: a | b)
#Dictionary of the above
businessRatingDictionary = dict(businessRating.collect())
#Broadcasting the above dictionary
businessRatingBroadcasted = sc.broadcast(businessRatingDictionary)

#RDD containing all distinct businesses in training set
trainBusinessRDD = trainRDDFinal.map(lambda row: row[1]).distinct()
#Set containing all distinct business in training set
trainBusinessSet = set(trainBusinessRDD.collect())

#RDD with user, business from test set
testDataRDD = testRDDFinal.map(lambda row: (row[0], row[1]))

#RDD containing user and all its rated business
userAndRatedBusinessesRDD = trainRDDFinal.map(lambda row: (row[0], {row[1]})).reduceByKey(lambda a, b: a | b)
#Dictionary of the above
userAndRatedBusinessesDictionary = dict(userAndRatedBusinessesRDD.collect())
#Broadcasting the above dictionary
userAndRatedBusinessesBroadcasted = sc.broadcast(userAndRatedBusinessesDictionary)

#RDD containing business and all users that rated it
businessAndRatedByUsersRDD = trainRDDFinal.map(lambda row: (row[1], {row[0]})).reduceByKey(lambda a, b: a | b)
#Dictionary of the above
businessAndRatedByUsersDictionary = dict(businessAndRatedByUsersRDD.collect())

#RDD containing business as key and user and rating as value
businessAndRatedByUsersAndRatingRDD = trainRDDFinal.map(lambda row: (row[1], {row[0]: float(row[2])})).reduceByKey(lambda x, y: {**x, **y})
#Dictionary of the above
businessAndRatedByUsersAndRatingDictionary = dict(businessAndRatedByUsersAndRatingRDD.collect())

#RDD of TEST user, business where business is present in the train dataset
testDataFilteredRDD = testDataRDD.filter(lambda x: x[1] in trainBusinessSet)

#RDD of TEST user, business, all business rated by the user
userBusinessAndOtherRatedBusinessesRDD = testDataFilteredRDD.map(lambda row: (row[0], row[1], userAndRatedBusinessesBroadcasted.value[row[0]]))

#RDD  of TEST key: (currentUser, businessToBePredicted) and value: top 50 similar business as per W-> (business, W, Rating of this business by currentUser)
top50BusinessRDD = userBusinessAndOtherRatedBusinessesRDD.map(pearsonCoefficient)

#RDD of TEST currentUser as key and currentBusiness, predicted_rating as its value
predictedRatingsRDD = top50BusinessRDD.map(predictRatings).reduceByKey(lambda x, y: {**x, **y})
#Dictionary of the above
predictedRatingsDictionary = dict(predictedRatingsRDD.collect())


#----------------------------------------------------------Model based-------------------------------------------------------------------

testRDD2 = sc.textFile(testFile).map(lambda x: x.split(','))
testFirstRow = testRDD2.first()
testTempRDD = testRDD2.filter(lambda x: x != testFirstRow)
testFinalRDD = testTempRDD.map(lambda x: (x[0], x[1]))
testFinalDictionary=testFinalRDD.collect()

trainRDD2 = sc.textFile(os.path.join(folderPath, 'yelp_train.csv')).map(lambda x: x.split(","))
trainFirstRow = trainRDD2.first()
trainTempRDD = trainRDD2.filter(lambda x: x != trainFirstRow)
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
        user=testFinalDictionary[i][0]
        business=testFinalDictionary[i][1]
        p1=min(5,result[i])
        prediction1=max(1,p1)
        if user in predictedRatingsDictionary and business in predictedRatingsDictionary[user]:
            prediction2=predictedRatingsDictionary[user][business]
            combinedPrediction=0.99*prediction1+0.01*prediction2
            file.write(user + "," + business + "," + str(combinedPrediction) + "\n")

        else:
            #When no business and user present in training data
            if user not in predictedRatingsDictionary and business not in predictedRatingsDictionary[user]:
                prediction2=3
                combinedPrediction=0.99*prediction1+0.01*prediction2
                file.write(user + "," + business + "," + str(combinedPrediction) + "\n")

            #When no business present in training data   
            elif business not in predictedRatingsDictionary[user]:
                userRatingArray=userRatingBroadcasted.value[user]
                userAverageRating=sum(userRatingArray)/float(len(userRatingArray))
                prediction2=userAverageRating
                combinedPrediction=0.99*prediction1+0.01*prediction2
                file.write(user + "," + business + "," + str(combinedPrediction) + "\n")

            #When no user present in training data 
            else:
                businessRatingArray=businessRatingBroadcasted.value[business]
                businessAverageRating=sum(businessRatingArray)/float(len(businessRatingArray))
                prediction2=businessAverageRating
                combinedPrediction=0.99*prediction1+0.01*prediction2
                file.write(user + "," + business + "," + str(combinedPrediction) + "\n")


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