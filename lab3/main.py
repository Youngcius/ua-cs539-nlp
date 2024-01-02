import re
import math
from collections import defaultdict, Counter


class VecDense:

    def tokenizeDoc(self, oneDoc: str):
        """This method tokenizes a a string.

        :param oneDoc: a string.
        :return: a tokenized sting.
        """
        sanitizedStr = re.sub(r'[^a-zA-Z0-9 ]', '', oneDoc)
        tokens = sanitizedStr.lower().split(" ")
        return tokens

    def getVecLength(self, vecIn: list):
        """This method computes the length of a vector.

        :param vecIn: a list representing a vector, one element per dimension.
        :return: the length of the vector.
        """
        # return np.linalg.norm(vecIn)
        return math.sqrt(sum([a ** 2 for a in vecIn]))

    def normalizeVec(self, vecIn: list):
        """This method normalizes a vector to unit length.

        :param vecIn:  a list representing a vector, one element per dimension.
        :return: a list representing a vector, that has been normalized to unit length.
        """
        # return (np.atleast_1d(vecIn) / np.linalg.norm(vecIn)).tolist()
        return [a / self.getVecLength(vecIn) for a in vecIn]

    def dotProductVec(self, vecInA: list, vecInB: list):
        """This method takes the dot product of two vectors.

        :param vecInA, vecInB: two lists representing vectors,
            one element per dimension.
        :return: the dot product.
        """
        # return np.dot(vecInA, vecInB)
        return sum([a * b for a, b in zip(vecInA, vecInB)])

    def cosine(self, vecInA: list, vecInB: list):
        """This method obtains the cosine between two vectors
            (which is nominally the dot product of two vectors of unit length).

        :param vecInA, vecInB: two lists representing vectors, one element per dimension.
        :return: the cosine.
        """
        # return np.dot(vecInA, vecInB) / np.linalg.norm(vecInA) / np.linalg.norm(vecInB)
        return self.dotProductVec(vecInA, vecInB) / self.getVecLength(vecInA) / self.getVecLength(vecInB)

    def computeCentroidVector(self, tokensIn: list, vecDict: dict):
        """This method calculates the centroid vector from a list of
            tokens. The centroid vector is the "average"
            vector of a list of tokens.
        #NOTE:  Special considerations:
            - all tokens should be converted to lower case.
            - if a vector isn't in the dictionary, it
                shouldn't be a part of the average.

        :param tokensIn: a list of tokens.
        :param vecDict: the vector library is a dictionary, 'vecDict',
            whose keys are tokens, and values are lists representing vectors.
        :return: the centroid vector, represented as a list.
        """
        vectors = []
        for t in tokensIn:
            if t.lower() in vecDict.keys():
                vectors.append(vecDict[t.lower()])

        num = len(vectors)
        if vectors:
            # return np.mean(vectors, axis=0).tolist()
            res = []
            for a in zip(*vectors):
                res.append(sum(a) / num)
            return res
        else:
            return [0] * list(vecDict.items())[0][1]


class VecSparseTFIDF:
    """This class calculates TF-IDF vector
    """

    def tokenizeDoc(self, oneDoc: str):
        """This method tokenizes a text.

        :param oneDoc: a string of text.
        :return: a tokenized text.
        """
        sanitizedStr = re.sub(r'[^a-zA-Z0-9 ]', '', oneDoc)
        tokens = sanitizedStr.lower().split(" ")
        return tokens

    def getTermFreq(self, oneDoc: str):
        """This method obtains term frequency.

        :param oneDoc: a string of text, called a document. e.g. "the cat saw the hat"
        :return: a defaultdict representing the term frequency
            counts in that document. Keys are tokens,
            values are counts.
        #NOTE: The input document should be tokenized using tokenizeDoc method.
        """
        termFreq = defaultdict(lambda: 0)
        tokens = self.tokenizeDoc(oneDoc)
        for t in tokens:
            termFreq[t] += 1
        return termFreq

    def getDocFreqs(self, allDocs: list):
        """This method obtains document frequencies.

        :param allDocs: a list of strings.  Each string is one document.
        :return: a default dictionary representing the document frequency
            counts across all documents. Keys are tokens,
            values are counts.
        """
        docFreq = defaultdict(lambda: 0)
        for doc in allDocs:
            # tokens = np.unique(self.tokenizeDoc(doc))
            tokens = sorted(list(Counter(self.tokenizeDoc(doc)).keys()))
            for t in tokens:
                docFreq[t] += 1
        return docFreq

    def makeTFIDFVec(self, oneDoc: str, docFreqs: defaultdict, numDocs: int):
        """This method creates a TF-IDF vector for a given document.

        :param oneDoc: a string representing one document.
        :param docFreqs: a default dictionary representing the document
            frequency counts.  Keys are tokens, values are counts.
        :param numDocs: the total number of documents in the collection.
        :return: a default dictionary representing the tf-idf vector.
            Keys are tokens, values are counts.
        #NOTE: There are many ways to calculate tf-idf vectors.
           Term frequency should be the count of the words
            in a given document.
           Document frequency should be calculated with add-one
            smoothing, as log10((numDocs+1) / (docFreqOfToken + 1)).
        """
        vecOut = defaultdict(lambda: 0)

        termFreq = self.getTermFreq(oneDoc)

        tokens = self.tokenizeDoc(oneDoc)
        for t in tokens:
            # vecOut[t] = (1 + math.log10(termFreq[t])) * math.log10((numDocs + 1) / (docFreqs[t] + 1))

            vecOut[t] = termFreq[t] * math.log10((numDocs + 1) / (docFreqs[t] + 1))

        return vecOut

    def getVecLengthSparse(self, vecIn: defaultdict):
        """This method computes the length of a sparse vector.

        :param vecIn: a default dictionary representing a sparse
            vector. keys are tokens, values are counts, default = 0.
        :return:the length of the vector.
        """
        counts = list(vecIn.values())
        # return np.linalg.norm(counts)
        return math.sqrt(sum([c ** 2 for c in counts]))

    def normalizeVecSparse(self, vecIn: defaultdict):
        """This method normalizes a sparse vector to unit length.

        :param vecIn: a default dictionary representing a sparse vector.
          keys are tokens, values are counts, default = 0.
        :return: a list representing a vector, that has been
            normalized to unit length.
        """
        vecOut = defaultdict(lambda: 0)
        length = self.getVecLengthSparse(vecIn)
        for k, v in vecIn.items():
            vecOut[k] = vecIn[k] / length
        return vecOut

    def dotProductVecSparse(self, vecInA: defaultdict, vecInB: defaultdict):
        """This method takes the dot product of two sparse vectors.

        :param vecInA, vecInB:two default dictionaries representing sparse
            vectors.  keys are tokens, values are counts, default = 0.
        :return: the dot product.
        """
        # keys = np.unique(list(vecInA.keys()) + list(vecInB.keys()))
        keys = sorted(list(Counter(list(vecInA.keys()) + list(vecInB.keys())).keys()))
        res = 0
        for k in keys:
            res += vecInA[k] * vecInB[k]
        return res

    def cosineSparse(self, vecInA: defaultdict, vecInB: defaultdict):
        """This method obtains the cosine between two vectors (which
            is nominally the dot product of two vectors of unit length).

        :param vecInA, vecInB: two default dictionaries representing sparse
            vectors.  keys are tokens, values are counts, default = 0.
        :return: the cosine.
        """
        len1 = self.getVecLengthSparse(vecInA)
        len2 = self.getVecLengthSparse(vecInB)
        return self.dotProductVecSparse(vecInA, vecInB) / len1 / len2


def loadVectors(filename: str):
    """This function loads word vectors from the file.

    :param filename:  the filename of the vectors
        (e.g. glove.subset.50d.txt)
    :return: a dictionary, key: token, value: list of numbers loaded
        from the file.
    """
    print("Loading word vectors from file (" + str(filename) + ")")

    with open('glove.subset.50d.txt', 'r') as f:
        lines = f.readlines()
    linesSplitted = [l.split() for l in lines]
    wordVecs = {l[0]: list(map(float, l[1:])) for l in linesSplitted}

    print("Loaded " + str(len(wordVecs)) + " word vectors.")
    return wordVecs


def doQuestionAnsweringCentroidDense(questions: list, wordVecs: dict):
    """This function performs a baseline question answering task.
    This function implements a cosine similarity baseline question
    answering system for multiple choice questions. For each question,
    compute the centroid vector of the question text.
    Then, for each answer candidate, compute the cosine between the
    question text centroid vector, and that answer choice's centroid
    vector. Pick the answer choice that has the highest cosine
    similarity as the model's chosen answer. If the answer is correct,
     increment numCorrect. Return the total numCorrect.

    :param questions: a list of multiple questions, as included in the
        appropriate test.
    :param wordVecs: a dictionary of word vectors, loaded from
        'loadVectors()'.
    :return: the number of questions answered correctly.
    """
    # {
    #     "question": "Which tools are used to determine the boiling point of water?",
    #     "choices": ["hot plate and stopwatch", "thermometer and balance", "hot plate and thermometer",
    #                 "balance and graduated cylinder"],
    #     "correctIdx": 2,}

    vecDense = VecDense()

    # Evaluate QA performance of cosine model on questions
    numCorrect = 0

    for q in questions:
        centVecQuestion = vecDense.computeCentroidVector(tokensIn=vecDense.tokenizeDoc(q['question']), vecDict=wordVecs)
        centVecChoices = [vecDense.computeCentroidVector(tokensIn=vecDense.tokenizeDoc(choice), vecDict=wordVecs) for
                          choice in q['choices']]
        similarities = [vecDense.cosine(centVecQuestion, vec) for vec in centVecChoices]
        # print(q['correctIdx'], np.argmax(similarities), similarities)
        # if np.argmax(similarities) == q['correctIdx']:
        if similarities.index(max(similarities)) == q['correctIdx']:
            numCorrect += 1

    return numCorrect
