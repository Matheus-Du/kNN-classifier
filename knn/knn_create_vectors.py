import os
import sys
import json
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import math


class Document():
	def __init__(self, name, category, terms):
		self.name = name
		self.category = category
		self.terms = terms
		self.vector = []

	def __str__(self):
		return self.name


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
	args (list): both the relative path to the training file and the path to the file where the model will be written
	"""

	# check that the correct number of arguments were passed
	try:
		assert len(args) == 2
	except AssertionError:
		print("Usage: python knn_create_vectors.py <training_file> <output_file_name>")
		exit(1)

	# open the training file
	asbPath = os.path.abspath(__file__)
	relPath = args[0]
	path = os.path.join(os.path.dirname(asbPath), "." + relPath)
	try:
		trainingFile = open(path, "r")
	# if the file can't be found, try again using the raw relative path
	except IOError:
		try:
			path = os.path.join(os.path.dirname(asbPath), relPath)
			trainingFile = open(relPath, "r")
		# if the file can't be found, print an error message and exit
		except IOError:
			print("file path: " + path)
			print("Error: training file not found")
			exit(1)
	# load the training data
	trainingData = json.load(trainingFile)
	trainingFile.close()
	nltk.download('stopwords')
	nltk.download('punkt')

	# iterate through each document in the training data and create a Document object for each
	documents = []
	vocabulary = {} # format: {term: df}
	docCount = 0
	for doc in trainingData:
		# check for a valid category
		try:
			category = doc["category"]
		except KeyError:
			print("Error: document has no category")
			exit(1)

		document = Document(docCount, category, {})
		terms = []
		# iterate through each zone in the document and get the tokens in the text
		for zone in doc:
			if zone != "category":
				# get the tokens in the text
				terms.extend(getTerms(doc[zone]))
			
		# iterate through each term in the document and add it to the vocabulary
		for term in list(set(terms)):
			# if the term isn't in the vocabulary, add it
			if term in vocabulary:
				vocabulary[term] += 1
			# if the term is already in the vocabulary, increment its df
			else:
				vocabulary[term] = 1
		# iterate through each token in the document and add it to the document's term list
		for token in terms:
			# if the token is already in the document's term list, increment its tf
			if token in document.terms:
				document.terms[token] += 1
			# if the token isn't in the document's term list, add it
			else:
				document.terms[token] = 1

		documents.append(document)
		docCount += 1

	# replace the df values for all terms in the vocabulary with their idf values
	for term in vocabulary.keys():
		vocabulary[term] = math.log(docCount/vocabulary[term])

	# iterate through each document and vectorize it using the tf-idf values of its terms
	for document in documents:
		tfIdfs = []
		for term in document.terms.keys():
			tf = 1 + math.log(document.terms[term])
			idf = vocabulary[term]
			tfIdfs.append([term, tf*idf])
		# perform cosine normalization on the tf-idf weights
		weight = 0
		for tfIdf in tfIdfs:
			weight += tfIdf[1]**2
		weight = 1/math.sqrt(weight)
		# set the document's vector to the term + the normalized tf-idf weights
		for term in tfIdfs:
			document.vector.append([term[0], term[1]*weight])

	# write the model to the specified file
	relPath = args[1]
	path = os.path.join(os.path.dirname(asbPath), "." + relPath)
	try:
		modelFile = open(path, "w")
	# if the file can't be found, try again using the raw relative path
	except IOError:
		try:
			path = os.path.join(os.path.dirname(asbPath), relPath)
			modelFile = open(relPath, "w")
		# if the file can't be found, print an error message and exit
		except IOError:
			print("file path: " + path)
			print("Error: model file not found")
			exit(1)

	for term in vocabulary.keys():
		modelFile.write("idf\t{}\t{}\n".format(term, vocabulary[term]))
	for document in documents:
		# convert the vector to a string by concatenating each element with a period
		vectorStr = ""
		for vector in document.vector:
			vectorStr += vector[0] + ":" + str(vector[1]) + "|"
		modelFile.write("vector\t{}\t{}\n".format(document.category, vectorStr))
	modelFile.close()


if __name__ == '__main__':
	try:
		main(sys.argv[1:])
	except Exception as e:
		print("Error: " + str(e))
		exit(1)
