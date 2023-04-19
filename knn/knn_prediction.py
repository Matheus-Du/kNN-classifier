# need to represent each test doc as a vector using the same ltc scheme as the training docs
# then, need to find the 'k' closest training docs based on euclidean distance
import os
import sys
import json
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import math
from prettytable import PrettyTable


class Document():
	def __init__(self, name, category, terms):
		self.name = name
		self.category = category
		self.terms = terms 
		self.vector = {} # format: {term: tf-idf}
		self.nearestCategory = None # for the test docs only

	def __str__(self):
		return str(self.name) + " " + str(self.category) + " " + str(self.vector)


def getTerms(text):
	"""
		Returns the tokens in the given text and updates the term list.

		Parameters:
			text (string): the text to be tokenized
	"""

	stopWords = set(stopwords.words('english'))
	tokens = word_tokenize(text)
	filteredTokens = []

	for token in tokens:
		# iterate through each token in the text and append non-stop words to the list of tokens
		token = re.sub(r'[^\w\s]', '', token)
		if token not in stopWords and token != '':
			filteredTokens.append(token.lower())

	return filteredTokens


def main(args):
	"""
	Main function of the NB classifier module.

	Parameters:
	args (list): the relative path to the model file, the relative path to the test file, and the 'k' value for kNN
	"""

	# check that the correct number of arguments were passed
	try:
		assert len(args) == 3
	except AssertionError:
		print("Usage: python knn_prediction.py <preprocessing_file> <test_file> <k>")
		exit(1)

	# open the model file
	asbPath = os.path.abspath(__file__)
	relPath = args[0]
	path = os.path.join(os.path.dirname(asbPath), "." + relPath)
	try:
		modelFile = open(path, "r")
	# if the file can't be found, try again using the raw relative path
	except IOError:
		try:
			path = os.path.join(os.path.dirname(asbPath), relPath)
			modelFile = open(relPath, "r")
		# if the file can't be found, print an error message and exit
		except IOError:
			print("file path: " + path)
			print("Error: training file not found")
			exit(1)
	# load the training data
	nltk.download('stopwords')
	nltk.download('punkt')

	trainingDocuments = []
	trainingVocabulary = {} # format: {term: df}
	trainingCategories = []
	docCount = 0
	# iterate through each document in the model and assign it to the appropriate data structure
	for line in modelFile:
		data = line.split('\t')
		for i in range(len(data)):
			data[i] = data[i].strip()
		
		if data[0] == "idf":
			# if the line contains a term's idf, add it to the vocabulary
			trainingVocabulary[data[1]] = float(data[2])
		elif data[0] == "vector":
			category = data[1]
			# if the category isn't in the categories list for the training data, add it
			if category not in trainingCategories:
				trainingCategories.append(category)
			# if the line contains a document's vector, decode the vector back into a list of floats
			vector = {}
			vectorStr = data[2].split('|')
			# isolate the term and its tf-idf value
			for item in vectorStr:
				if item != '':
					termDimension = item.split(':')
					vector.update({termDimension[0]: float(termDimension[1])})
			# create a new document object using the category and decoded vector and add it to the list of training documents
			document = Document(docCount, data[1], None)
			document.vector = vector
			trainingDocuments.append(document)
			docCount += 1

	# open the test file
	asbPath = os.path.abspath(__file__)
	relPath = args[1]
	path = os.path.join(os.path.dirname(asbPath), "." + relPath)
	try:
		testFile = open(path, "r")
	# if the file can't be found, try again using the raw relative path
	except IOError:
		try:
			path = os.path.join(os.path.dirname(asbPath), relPath)
			testFile = open(relPath, "r")
		# if the file can't be found, print an error message and exit
		except IOError:
			print("file path: " + path)
			print("Error: test file not found")
			exit(1)
	# load the test data
	test = json.load(testFile)
	testFile.close()

	testDocuments = []
	testVocabulary = {} # format: {term: df}
	docCount = 0
	testCategories = []
	# iterate through each document in the test data and create a new document object for each doc
	for doc in test:
		# check for a valid category
		try:
			category = doc["category"]
		except KeyError:
			print("Error: document has no category")
			exit(1)

		# add the category to the list of categories if it isn't already in the list
		if category not in testCategories:
			testCategories.append(category)
		
		document = Document(docCount, category, {})
		terms = []
		# iterate through each zone in the document and get the tokens in the text
		for zone in doc:
			if zone != "category":
				# get the tokens in the text
				terms.extend(getTerms(doc[zone]))
			
		# iterate through each term in the document and add it to the vocabulary
		for term in list(set(terms)):
			# if the term isn't in the 
			# if the term is in the vocabulary, increment its df
			if term in testVocabulary:
				testVocabulary[term] += 1
			# if the term is already in the vocabulary, check if it's in the training vocabulary and add it to the test vocabulary if it is
			else:
				if term in trainingVocabulary:
					testVocabulary[term] = 1
		# iterate through each token in the document and add it to the document's term list
		for token in terms:
			# if the token is already in the document's term list, increment its tf
			if token in document.terms:
				document.terms[token] += 1
			# if the token isn't in the document's term list, check if it's in the test vocabulary and add it to the document's term list if it is
			else:
				if token in testVocabulary:
					document.terms[token] = 1

		testDocuments.append(document)
		docCount += 1

	# replace the df values in the test vocabulary with the idf values
	for term in testVocabulary.keys():
		testVocabulary[term] = math.log(docCount/testVocabulary[term])
	
	# iterate through each test document and compute its tf-idf vector
	for document in testDocuments:
		tfIdfs = []
		# iterate through each term in the test document
		for term in document.terms.keys():
			# compute the tf-idf value for the term and add it to the vector
			tf = 1 + math.log(document.terms[term])
			idf = testVocabulary[term]
			tfIdfs.append([term, tf * idf])
		# perform cosine normalization on the vector
		weight = 0
		for tfIdf in tfIdfs:
			weight += tfIdf[1]**2
		weight = 1/math.sqrt(weight)
		# update the document's vector with the term and normalized vector
		for term in tfIdfs:
			document.vector.update({term[0]: term[1] * weight})

	# iterate through each test document and find its k nearest neighbors
	k = int(args[2])
	categoryPredictions = {} # format: {categoryName: [TP, FP, FN, TN]}
	for category in list(set(testCategories) & set(trainingCategories)):
		categoryPredictions[category] = [0, 0, 0, 0]

	for document in testDocuments:
		# create a list of tuples containing the distance between the test document and each training document and the training document's category
		distances = []
		terms = []
		for trainingDocument in trainingDocuments:
			# get all terms shared by the test document and the training document
			terms = list(set(document.vector.keys()) & set(trainingDocument.vector.keys()))
			# compute the euclidean distance between the two documents
			distance = 0
			for term in terms:
				distance += (document.vector[term] - trainingDocument.vector[term])**2
			distance = math.sqrt(distance)
			# if the length of distances is less than k, add the distance and document to the list
			if len(distances) < k:
				distances.append((distance, trainingDocument))
			# if the length of distances is equal to k, check if the distance is less than the distance of the furthest neighbor
			else:
				# pop the furthest neighbor off the heap
				distances.sort(key=lambda x: x[0])
				if distance > distances[k-1][0]:
					distances[k-1] = (-distance, trainingDocument)

		# compute the score for each category using the equation from p299 in the textbook
		categories = list(set(testCategories) & set(trainingCategories))
		topCategory = [None, -math.inf]
		for category in categories:
			score = 0
			# iterate through each neighbour and 
			for distance in distances:
				if distance[1].category == category:
					# if the category of the training doc and the current category are the same, get the cosine similarity between the two document's vectors
					terms = list(set(document.vector.keys()) & set(distance[1].vector.keys()))
					if len(terms) > 0:
						sumxx, sumxy, sumyy = 0, 0, 0
						for term in terms:
							x = document.vector[term]
							y = distance[1].vector[term]
							sumxx += x*x
							sumyy += y*y
							sumxy += x*y
						score += sumxy/math.sqrt(sumxx*sumyy)
				
			if score > topCategory[1]:
				# if the category being replaced is not None, increment the FN value
				if topCategory[0] != None:
					categoryPredictions[topCategory[0]][3] += 1
				# if the score is greater than the current top score, update the top category
				topCategory = [category, score]
		
		# if the category prediction is correct, increment the TP value
		if topCategory[0] == document.category:
			categoryPredictions[document.category][0] += 1
		# if the category prediction is incorrect, increment the FP value
		else:
			categoryPredictions[topCategory[0]][1] += 1
			categoryPredictions[document.category][2] += 1

	totalF1 = 0.0
	totalTP = 0.0
	totalFP = 0.0
	totalFN = 0.0
	# print the results for each category
	table = PrettyTable(['Category', 'TP', 'FP', 'FN', 'TN', 'Precision', 'Recall', 'F1'])
	for category in categoryPredictions.keys():
		# calculate the precision, recall, and F1 score for each category
		TP = categoryPredictions[category][0]
		totalTP += TP
		FP = categoryPredictions[category][1]
		totalFP += FP
		FN = categoryPredictions[category][2]
		totalFN += FN
		TN = categoryPredictions[category][3]
		precision = (TP + 1) / (TP + FP + 1)
		recall = (TP + 1) / (TP + FN + 1)
		F1 = 2 * (precision * recall) / (precision + recall)
		totalF1 += F1
		table.add_row([category, TP, FP, FN, TN, precision, recall, F1])
	print(table)
	# print micro-averaged and macro-averaged F1 scores
	print("Micro-averaged F1: " + str(totalF1/len(categoryPredictions)))
	macroF1 = totalTP + 1.0
	macroF1 = macroF1 / (totalTP + (1/2)*(totalFP + totalFN) + 1)
	print("Macro-averaged F1: " + str(macroF1))


if __name__ == "__main__":
	try:
		main(sys.argv[1:])
	except Exception as e:
		print("Error: " + e)
		exit(1)
		