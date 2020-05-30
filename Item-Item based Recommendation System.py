import json
import sys
# import findspark
from pyspark import SparkContext
import time
# findspark.init()
import numpy as np
if __name__ == "__main__":
	start_time = time.time()
	sc = SparkContext()
	rdd = sc.textFile(sys.argv[1])

	rdd_dict = rdd.map(json.loads)

	"""
	Function to get the latest Review based on timestamp
	Returns (timestamp,LatestReview)
	"""
	def getLatestReviewReduceByKey(a,b):
		if a[0] > b[0]:
			return a
		else:
			return b

	"""
	Function to combine all user,item reviews so that we can take latest review
	Returns ((reviewerdId,itemId),(timestamp,review)) 
	"""
	def getReviewTimeStamp(record):
		timeStamp = record['unixReviewTime']
		productId = record['asin']
		reviewerId = record['reviewerID']
		out = []
		out.append(((reviewerId,productId),(timeStamp,record)))
		return out

	def getLatestReview(record):
		key = record[0]
		val = record[1]
		latestTime = 0
		review = ""
		for reviews in val:
			if (reviews[0] > latestTime):
				latestTime = reviews[0]
				review = reviews[1]
		return review

	"""
	Function to combine the items with their reviewerId and rating
	Returns ((itemId),(reviewerId, ratingByReviewer))
	"""
	def groupByProduct(record):
		reviewerId = record['reviewerID']
		productId = record['asin']
		rating = record['overall']
		out = []
		out.append(((productId),(reviewerId,rating)))
		return out

	"""
	Filter out items which has been rated by less than 25 distinct users
	Return item if rated by 25 or more users
	"""
	def filterItemsOnUserCount(record):
		val = record[1]
		if len(val) >= distinctUsers.value:
			return True
		return False

	"""
	Function to combine the users with the itemId they reviewed and rating
	Returns ((userId),(itemId, rating))
	"""
	def groupByUser(record):
		productId = record[0]
		reviews = record[1]
		out = []
		for review in reviews:
			out.append(((review[0]),(productId,review[1])))
		return out

	"""
	Filter out user who has rated less than 5 distinct items
	Return user if rated 5 or more items
	"""
	def filterUserOnReviewCount(record):
		key = record[0]
		val = record[1]
		if len(val) >= distinctItems.value:
			return True
		return False
	"""
	Function to filter out users from each item who 
	have not rated atleast 5 other items
	"""
	def generateUtilityMatrix(record):
		productId = record[0]
		reviews = record[1]
		userProductTempList = []
		userProductFinalList = []
		count = 0
		rating = 0.0
		for review in reviews:
			if review[0] in userList.value:
				count = count+1
				rating = rating + review[1]
				userProductTempList.append((review[0],review[1]))
		mean = 0.0
		if count != 0:
			mean = (float(rating/count))
		for items in userProductTempList:
			userProductFinalList.append((items[0],items[1],items[1]-mean))
		return (productId,userProductFinalList)

	"""
	Function to get the cosine Similarity between other items and the input items
	Returns (targetProductId,(cosineSimWithOtherItem, list of users and ratings 
	who have rated other items). Filters neighbors with cosine Similarity <=0 or if 
	target item and current item do not have 2 users in common whi have rated both items 
	"""

	def getCosineSimilarity(item):
		productId = item[0]
		# if(productId in recommendationItems.value):
		# 	return []
		userReviews = item[1]
		##Make dictionary of users for recommeded items
		userListRCItems = recommendationItemsBCList.value
		###
		sumNeigh = 0.0
		neighItem = {}
		for userReview in userReviews:
			userId = userReview[0]
			userRating = userReview[1]
			userNormRating = userReview[2]
			sumNeigh = sumNeigh + (userNormRating**2)
			neighItem[userId] = (userRating,userNormRating)
		##
		cosineSimDict = {}
		outList =[]
		cosineSimList = []
		for (targetProductId,dictItem) in userListRCItems: #list of recommedationItems
			if productId != targetProductId:
				count = 0
				cosineSimNum = 0.0
				sumTarget = 0.0
				for user, ratingNormRating in dictItem.items():
					rating = ratingNormRating[0]
					normRating = ratingNormRating[1]
					##
					if user in neighItem:
						count = count+1
						neighItemUserRatingNormRating = neighItem[user]
						neighItemUserRating = neighItemUserRatingNormRating[0]
						neighItemUserNormRating = neighItemUserRatingNormRating[1]
						cosineSimNum = cosineSimNum + (normRating*neighItemUserNormRating)
					sumTarget = sumTarget+ (normRating**2)
				##
				if count >=2 and sumNeigh >0.0 and sumTarget>0.0 :
					cosineSim = cosineSimNum/(np.sqrt(sumTarget)*np.sqrt(sumNeigh))
					# cosineSimList.append((targetProductId,productId,cosineSim,neighItem))
					if cosineSim > 0.0:
						outList.append(((targetProductId,(cosineSim,neighItem))))
						# outList.append(((targetProductId,(cosineSim,productId,neighItem))))
		# return cosineSimList
		return outList

	"""
	Function to generate users to be predicted for target Items
	Filters users who have not rated atleast 2 items in neighborhood
	Returns (targetProductId, list of users and their rating in other items)
	"""
	def generateUsers(record):
		targetProductId = record[0]
		targetProductUserDict = recommendationItemsBCList.value[0][1]
		for targetItems in recommendationItemsBCList.value:
			if targetItems[0] == targetProductId:
				targetProductUserDict = targetItems[1]
				break
		neighItemsRatings = record[1]
		neighItems = []
		userDict = {}
		neighItems = list(neighItemsRatings)
		if len(neighItems)>50:
			neighItems.sort(reverse=True)
			# return neighItems
			neighItems =  neighItems[:50]
		for user in userList.value:
			userDict[user] = []
		for cosineSim,neighItem in neighItems:
			for user,ratings in neighItem.items():
				userDict.get(user).append((cosineSim,ratings[0]))
		##
		userDictKeys = list(userDict.keys())

		for user in userDictKeys:
			if (user in targetProductUserDict) or (len(userDict[user])<2):
				del(userDict[user])
		return ((targetProductId,userDict))

	"""
	Function to predict user rating for target items.
	Return (targetProductId, predicted ratings for user)
	"""

	def predictUserRating(record):
		targetProductId = record[0]
		predictedRatings = []
		neighItemRatingsDict = record[1]
		for user,ratings in neighItemRatingsDict.items():
			cosineSimSum = 0.0
			targetItemUserRating = 0.0
			predictedUser = user
			for cosineSimRating in ratings:
				cosineSimSum = cosineSimSum + cosineSimRating[0]
				targetItemUserRating = targetItemUserRating + (cosineSimRating[0]*cosineSimRating[1])
			predictedRating = targetItemUserRating/cosineSimSum
			predictedRatings.append((predictedUser,predictedRating))
		return((targetProductId,predictedRatings))

	"""
	Generate a list of users and their rating for the input items
	Returns (targetItemId, list of users along with their ratings 
	who have rated this targetItems)
	"""
	def getRecommedationItemsBCList(record):
		dictItem = {}
		productId = record[0]
		productUserReviews = record[1]
		if productId in recommendationItems.value:
			for userReview in productUserReviews:
				dictItem[userReview[0]] = (userReview[1],userReview[2])
			return ([(productId,dictItem)])
		return []
	"""
	Function to generate the final utility matrix for input items
	Predicted ratings for new users along with ratings of users who 
	have already rated input items 
	"""
	def generateFinalUtilityMatrix(record):
		key = record[0]
		value = dict(record[1])
		for targetId, usersDict in recommendationItemsBCList.value:
			if targetId == key:
				for user,ratings in usersDict.items():
					value[user] = ratings[0]

		#replacing None for users who's rating was not predicted
		for user in userList.value:
			if user not in value:
				value[user] = None
		out = zip(value.keys(),value.values())
		return (key,list(out))

	recommendationItems = sc.broadcast(sys.argv[2])
	distinctUsers = sc.broadcast(25)
	distinctItems = sc.broadcast(5)
	#step a.
	latestReviewRDD = rdd_dict.flatMap(getReviewTimeStamp).reduceByKey(lambda x,y:getLatestReviewReduceByKey(x,y)).map(lambda record: record[1][1])
	#step b.
	filterItemsOnUserCountRDD = latestReviewRDD.map(groupByProduct).flatMap(lambda record: record).groupByKey().filter(filterItemsOnUserCount)
	#step c.
	filterUsersOnReviewCountRDD = filterItemsOnUserCountRDD.map(groupByUser).\
		flatMap(lambda record:record).groupByKey().filter(filterUserOnReviewCount).map(lambda record: record[0])

	userList = sc.broadcast(filterUsersOnReviewCountRDD.collect())
	utilityMatrix = filterItemsOnUserCountRDD.map(generateUtilityMatrix)
	utilityMatrix.persist()
	recommendationItemsBCList = sc.broadcast(utilityMatrix.flatMap(getRecommedationItemsBCList).collect())
	precitedUserRating = utilityMatrix.flatMap(getCosineSimilarity).groupByKey().map(generateUsers).map(predictUserRating)
	finalUtiltiyMatrix = precitedUserRating.map(generateFinalUtilityMatrix)
	print()
	print("###########################################################################################################")
	print("###########################################################################################################")
	print("##########Completed utility matrix (including predictions for each user) for the input products############")
	print(finalUtiltiyMatrix.collect())
	print("###########################################################################################################")
	print("###########################################################################################################")
	print()

	# finalUtiltiyMatrix.saveAsTextFile('hdfs:/output')
	print("--- Time taken for file %s is %s seconds ---" % (sys.argv[1], (time.time() - start_time)))
