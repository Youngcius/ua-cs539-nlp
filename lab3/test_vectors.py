import pytest
from collections import defaultdict

from main import VecDense, VecSparseTFIDF, loadVectors, doQuestionAnsweringCentroidDense


# @pytest.fixture(autouse=True)
# Check the length calculation for dense vectors is correct
def test_vecDenseLength():
    vecDense = VecDense()

    testVec1 = [0.5, 0.4, 0.3, 0.2]

    length = vecDense.getVecLength(testVec1)
    # print(length)
    assert (length == pytest.approx(0.734, abs=0.01))


# Check that the vector normalization is working
def test_vecDenseNormalize():
    vecDense = VecDense()

    testVec1 = [0.5, 0.4, 0.3, 0.2]
    normVec = vecDense.normalizeVec(testVec1)
    length = vecDense.getVecLength(normVec)

    assert (length == pytest.approx(1.0, abs=0.01))


def test_vecDenseDotProduct():
    vecDense = VecDense()

    testVec1 = [0.5, 0.4, 0.3]
    testVec2 = [0.3, 0.2, 0.1]
    dot = vecDense.dotProductVec(testVec1, testVec2)

    assert (dot == pytest.approx(0.26, abs=0.01))


def test_vecDenseCosine():
    vecDense = VecDense()

    vecDict = {
        'cat': [0.5, 0.4, 0.3, 0.1],
        'dog': [0.6, 0.4, 0.2, 0.9],
        'apple': [0.3, 0.3, 0.3, 0.5],
        'banana': [0.5, 0.5, 0.3, 0.5],
    }

    cosineCatDog = vecDense.cosine(vecDict['cat'], vecDict['dog'])
    cosineAppleBanana = vecDense.cosine(vecDict['apple'], vecDict['banana'])
    cosineCatApple = vecDense.cosine(vecDict['cat'], vecDict['apple'])
    cosineDogDog = vecDense.cosine(vecDict['dog'], vecDict['dog'])
    print(cosineCatDog)
    print(cosineAppleBanana)
    print(cosineCatApple)
    print(cosineDogDog)

    assert (cosineCatDog == pytest.approx(0.73, abs=0.01))
    assert (cosineAppleBanana == pytest.approx(0.97, abs=0.01))
    assert (cosineCatApple == pytest.approx(0.79, abs=0.01))
    assert (cosineDogDog == pytest.approx(1.00, abs=0.01))


def test_vecDenseCentroid():
    vecDense = VecDense()

    vecDict = {
        'cat': [0.5, 0.4, 0.3, 0.1],
        'dog': [0.6, 0.4, 0.2, 0.9],
        'apple': [0.3, 0.3, 0.3, 0.5],
        'banana': [0.5, 0.5, 0.3, 0.5],
    }

    sent1 = "the cat saw the dog with the apple"
    centroidVec = vecDense.computeCentroidVector(vecDense.tokenizeDoc(sent1), vecDict)
    print(centroidVec)

    assert (centroidVec[0] == pytest.approx(0.466, abs=0.01))
    assert (centroidVec[1] == pytest.approx(0.366, abs=0.01))
    assert (centroidVec[2] == pytest.approx(0.266, abs=0.01))
    assert (centroidVec[3] == pytest.approx(0.500, abs=0.01))


#
#   Sparse Vectors
#
def test_vecSparseLength():
    vecSparse = VecSparseTFIDF()

    vec1 = defaultdict(lambda: 0)
    vec1['cat'] = 1
    vec1['dog'] = 1
    vec1['a'] = 2

    length = vecSparse.getVecLengthSparse(vec1)
    print(length)

    assert (length == pytest.approx(2.45, abs=0.01))


def test_vecSparseNormalize():
    vecSparse = VecSparseTFIDF()

    vec1 = defaultdict(lambda: 0)
    vec1['cat'] = 1
    vec1['dog'] = 1
    vec1['a'] = 2

    vec1Norm = vecSparse.normalizeVecSparse(vec1)
    length = vecSparse.getVecLengthSparse(vec1Norm)
    print(length)

    assert (length == pytest.approx(1.0, abs=0.01))


def test_vecSparseDotProduct():
    vecSparse = VecSparseTFIDF()

    vec1 = defaultdict(lambda: 0)
    vec1['cat'] = 1
    vec1['dog'] = 1
    vec1['a'] = 3

    vec2 = defaultdict(lambda: 0)
    vec2['dog'] = 1
    vec2['tree'] = 1
    vec2['a'] = 2
    vec2['and'] = 2

    dot = vecSparse.dotProductVecSparse(vec1, vec2)
    print(dot)

    assert (dot == pytest.approx(7.0, abs=0.01))


def test_vecSparseCosine():
    vecSparse = VecSparseTFIDF()

    vec1 = defaultdict(lambda: 0)
    vec1['cat'] = 1
    vec1['dog'] = 1
    vec1['a'] = 3

    vec2 = defaultdict(lambda: 0)
    vec2['dog'] = 1
    vec2['tree'] = 1
    vec2['a'] = 2
    vec2['and'] = 2

    cos = vecSparse.cosineSparse(vec1, vec2)
    print(cos)

    assert (cos == pytest.approx(0.67, abs=0.01))


def test_vecSparseTermFreq():
    vecSparse = VecSparseTFIDF()

    doc1 = "this is a test and a great test it is"
    doc2 = "I like cats"
    doc3 = "cats are great but I also like dogs"
    allDocs = [doc1, doc2, doc3]

    vec1 = vecSparse.getTermFreq(doc1)
    print(vec1)

    assert (vec1['this'] == 1)
    assert (vec1['is'] == 2)
    assert (vec1['a'] == 2)
    assert (vec1['test'] == 2)
    assert (vec1['and'] == 1)
    assert (vec1['great'] == 1)
    assert (vec1['it'] == 1)

    vec2 = vecSparse.getTermFreq(doc2)
    print(vec2)

    assert (vec2['i'] == 1)
    assert (vec2['like'] == 1)
    assert (vec2['cats'] == 1)


def test_vecSparseDocFreq():
    vecSparse = VecSparseTFIDF()

    doc1 = "this is a test and a great test it is"
    doc2 = "I like cats"
    doc3 = "cats are great but I also like dogs"
    allDocs = [doc1, doc2, doc3]

    docFreqs = vecSparse.getDocFreqs(allDocs)
    print(docFreqs)

    assert (docFreqs['test'] == 1)
    assert (docFreqs['and'] == 1)
    assert (docFreqs['it'] == 1)
    assert (docFreqs['a'] == 1)
    assert (docFreqs['great'] == 2)
    assert (docFreqs['is'] == 1)
    assert (docFreqs['this'] == 1)
    assert (docFreqs['cats'] == 2)
    assert (docFreqs['i'] == 2)
    assert (docFreqs['like'] == 2)
    assert (docFreqs['are'] == 1)
    assert (docFreqs['dogs'] == 1)
    assert (docFreqs['also'] == 1)
    assert (docFreqs['but'] == 1)


def test_vecSparseTFIDF():
    vecSparse = VecSparseTFIDF()

    doc1 = "this is a test and a great test it is"
    doc2 = "I like cats"
    doc3 = "cats are great but I also like dogs"
    allDocs = [doc1, doc2, doc3]

    tfidfVec1 = vecSparse.makeTFIDFVec(doc1, vecSparse.getDocFreqs(allDocs), len(allDocs))
    print(tfidfVec1)

    # Orginal -- equation given in docstring ( log10(numDocs+1/docFreqOfToken+1) )
    # assert(tfidfVec1['this']    == pytest.approx(0.69, abs=0.01))
    # assert(tfidfVec1['is']      == pytest.approx(1.39, abs=0.01))
    # assert(tfidfVec1['a']       == pytest.approx(1.39, abs=0.01))
    # assert(tfidfVec1['test']    == pytest.approx(1.39, abs=0.01))
    # assert(tfidfVec1['and']     == pytest.approx(0.69, abs=0.01))
    # assert(tfidfVec1['great']   == pytest.approx(0.65, abs=0.01))
    # assert(tfidfVec1['it']      == pytest.approx(0.69, abs=0.01))

    # Corrected -- equation given in lecture/textbook ( log10((numDocs+1)/(docFreqOfToken+1)) )
    assert (tfidfVec1['this'] == pytest.approx(0.30, abs=0.01))
    assert (tfidfVec1['is'] == pytest.approx(0.60, abs=0.01))
    assert (tfidfVec1['a'] == pytest.approx(0.60, abs=0.01))
    assert (tfidfVec1['test'] == pytest.approx(0.60, abs=0.01))
    assert (tfidfVec1['and'] == pytest.approx(0.30, abs=0.01))
    assert (tfidfVec1['great'] == pytest.approx(0.125, abs=0.01))
    assert (tfidfVec1['it'] == pytest.approx(0.30, abs=0.01))


def test_vecSparseCosine_2():
    vecSparse = VecSparseTFIDF()

    doc1 = "this is a test and a great test it is"
    doc2 = "I like cats"
    doc3 = "cats are great but I also like dogs"
    allDocs = [doc1, doc2, doc3]

    print(vecSparse.cosineSparse(vecSparse.getTermFreq(doc1), vecSparse.getTermFreq(doc2)))
    print(vecSparse.cosineSparse(vecSparse.getTermFreq(doc1), vecSparse.getTermFreq(doc3)))
    print(vecSparse.cosineSparse(vecSparse.getTermFreq(doc2), vecSparse.getTermFreq(doc3)))
    print(vecSparse.cosineSparse(vecSparse.getTermFreq(doc3), vecSparse.getTermFreq(doc3)))

    assert (vecSparse.cosineSparse(vecSparse.getTermFreq(doc1), vecSparse.getTermFreq(doc2)) == pytest.approx(0.00,
                                                                                                              abs=0.01))
    assert (vecSparse.cosineSparse(vecSparse.getTermFreq(doc1), vecSparse.getTermFreq(doc3)) == pytest.approx(0.09,
                                                                                                              abs=0.01))
    assert (vecSparse.cosineSparse(vecSparse.getTermFreq(doc2), vecSparse.getTermFreq(doc3)) == pytest.approx(0.61,
                                                                                                              abs=0.01))
    assert (vecSparse.cosineSparse(vecSparse.getTermFreq(doc3), vecSparse.getTermFreq(doc3)) == pytest.approx(1.00,
                                                                                                              abs=0.01))


def test_vecSparseCosineTFIDF():
    vecSparse = VecSparseTFIDF()

    doc1 = "this is a test and a great test it is"
    doc2 = "I like cats"
    doc3 = "cats are great but I also like dogs"
    allDocs = [doc1, doc2, doc3]

    vec1TFIDF = vecSparse.makeTFIDFVec(doc1, vecSparse.getDocFreqs(allDocs), len(allDocs))
    vec2TFIDF = vecSparse.makeTFIDFVec(doc2, vecSparse.getDocFreqs(allDocs), len(allDocs))
    vec3TFIDF = vecSparse.makeTFIDFVec(doc3, vecSparse.getDocFreqs(allDocs), len(allDocs))

    print(vecSparse.cosineSparse(vec1TFIDF, vec2TFIDF))
    print(vecSparse.cosineSparse(vec1TFIDF, vec3TFIDF))
    print(vecSparse.cosineSparse(vec2TFIDF, vec3TFIDF))
    print(vecSparse.cosineSparse(vec3TFIDF, vec3TFIDF))

    # Orginal -- equation given in docstring ( log10(numDocs+1/docFreqOfToken+1) )
    # assert( vecSparse.cosineSparse( vec1TFIDF, vec2TFIDF ) == pytest.approx(0.00, abs=0.01) )
    # assert( vecSparse.cosineSparse( vec1TFIDF, vec3TFIDF ) == pytest.approx(0.08, abs=0.01) )
    # assert( vecSparse.cosineSparse( vec2TFIDF, vec3TFIDF ) == pytest.approx(0.59, abs=0.01) )
    # assert( vecSparse.cosineSparse( vec3TFIDF, vec3TFIDF ) == pytest.approx(1.00, abs=0.01) )

    # Corrected -- equation given in lecture/textbook ( log10((numDocs+1)/(docFreqOfToken+1)) )
    assert (vecSparse.cosineSparse(vec1TFIDF, vec2TFIDF) == pytest.approx(0.00, abs=0.01))
    assert (vecSparse.cosineSparse(vec1TFIDF, vec3TFIDF) == pytest.approx(0.02, abs=0.01))
    assert (vecSparse.cosineSparse(vec2TFIDF, vec3TFIDF) == pytest.approx(0.33, abs=0.01))
    assert (vecSparse.cosineSparse(vec3TFIDF, vec3TFIDF) == pytest.approx(1.00, abs=0.01))


def test_vecQuestionAnswering():
    vecDense = VecDense()
    wordVecs = loadVectors("glove.subset.50d.txt")

    #
    # Question corpus
    #
    questions = []

    q1 = {
        "question": "An ice cube placed in sunlight melts quickly. Which BEST explains this event?",
        "choices": ["The Sun is far away.", "The Sun makes heat.", "The ice cube is a solid. ",
                    "The ice cube looks clear."],
        "correctIdx": 1,
    }
    questions.append(q1)

    q2 = {
        "question": "As an airplane climbs in the sky, the pilot notices ice crystals forming on the windshield. This happens because",
        "choices": ["blizzards occur more often at higher altitudes.",
                    "friction with the atmosphere causes ice to develop.",
                    "water evaporates faster at higher altitudes, resulting in ice crystals.",
                    "moisture on the outside of the plane freezes due to colder air at higher altitudes."],
        "correctIdx": 3,
    }
    questions.append(q2)

    q3 = {
        "question": "Some objects conduct electricity. Which object is the best conductor of electricity?",
        "choices": ["a metal fork", "a wood spoon", "a plastic comb", "a rubber eraser"],
        "correctIdx": 0,
    }
    questions.append(q3)

    q4 = {
        "question": "Stars are organized into patterns called constellations. One constellation is named Leo. Which statement best explains why Leo appears in different areas of the sky throughout the year?",
        "choices": ["Earth revolves around the sun.", "The sun revolves around Earth.",
                    "The constellations revolve around Earth.", "Earth revolves around the constellations."],
        "correctIdx": 0,
    }
    questions.append(q4)

    q5 = {
        "question": "Why is some food dehydrated for astronauts to eat in space?",
        "choices": ["to make the food less messy to eat", "to preserve the food for extended flights",
                    "to increase the food's nutritional content", "to keep the food from floating off the plate"],
        "correctIdx": 1,
    }
    questions.append(q5)

    q6 = {
        "question": "Which tools are used to determine the boiling point of water?",
        "choices": ["hot plate and stopwatch", "thermometer and balance", "hot plate and thermometer",
                    "balance and graduated cylinder"],
        "correctIdx": 2,
    }
    questions.append(q6)

    # Evaluate QA performance of cosine model on these questions
    numCorrect = doQuestionAnsweringCentroidDense(questions, wordVecs)
    print("numCorrect: " + str(numCorrect))

    # Check that at least 3 out of 6 are correct
    assert (numCorrect >= 3)
