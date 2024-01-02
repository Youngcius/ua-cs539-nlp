import numpy as np
from scipy import sparse
from typing import Iterator, Iterable, Tuple, Text, Union
from sklearn.feature_extraction import text
from sklearn import preprocessing
from sklearn import linear_model

NDArray = Union[np.ndarray, sparse.spmatrix]


def read_smsspam(smsspam_path: str) -> Iterator[Tuple[Text, Text]]:
    """Generates (label, text) tuples from the lines in an SMSSpam file.

    SMSSpam files contain one message per line. Each line is composed of a labe l
    (ham or spam), a tab character, and the text of the SMS. Here are some
    examples:

      spam	85233 FREE>Ringtone!Reply REAL
      ham	I can take you at like noon
      ham	Where is it. Is there any opening for mca.

    :param smsspam_path: The path of an SMSSpam file, formatted as above.
    :return: An iterator over (label, text) tuples.
    """
    with open(smsspam_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return iter([(l.split()[0], ' '.join(l.split()[1:])) for l in lines])


class TextToFeatures:
    def __init__(self, texts: Iterable[Text]):
        """Initializes an object for converting texts to features.

        During initialization, the provided training texts are analyzed to
        determine the vocabulary, i.e., all feature values that the converter
        will support. Each such feature value will be associated with a unique
        integer index that may later be accessed via the .index() method.

        It is up to the implementer exactly what features to produce from a
        text, but the features will always include some single words and some
        multi-word expressions (e.g., "need" and "to you").

        :param texts: The training texts.
        """
        self.vectorizer = text.CountVectorizer(ngram_range=(1, 2))
        self.vectorizer.fit(texts)

    def index(self, feature: Text) -> int:
        """Returns the index in the vocabulary of the given feature value.

        :param feature: A feature
        :return: The unique integer index associated with the feature.
        """
        return self.vectorizer.get_feature_names_out().tolist().index(feature)

    def __call__(self, texts: Iterable[Text]) -> NDArray:
        """Creates a feature matrix from a sequence of texts.

        Each row of the matrix corresponds to one of the input texts. The value
        at index j of row i is the value in the ith text of the feature
        associated with the unique integer j.

        It is up to the implementer what the value of a feature that is present
        in a text should be, though a common choice is 1. Features that are
        absent from a text will have the value 0.

        :param texts: A sequence of texts.
        :return: A matrix, with one row of feature values for each text.
        """
        return self.vectorizer.transform(texts)


class TextToLabels:
    def __init__(self, labels: Iterable[Text]):
        """Initializes an object for converting texts to labels.

        During initialization, the provided training labels are analyzed to
        determine the vocabulary, i.e., all labels that the converter will
        support. Each such label will be associated with a unique integer index
        that may later be accessed via the .index() method.

        :param labels: The training labels.
        """
        self.encoder = preprocessing.LabelEncoder()
        self.encoder.fit(labels)

    def index(self, label: Text) -> int:
        """Returns the index in the vocabulary of the given label.

        :param label: A label
        :return: The unique integer index associated with the label.
        """
        return self.encoder.classes_.tolist().index(label)

    def __call__(self, labels: Iterable[Text]) -> NDArray:
        """Creates a label vector from a sequence of labels.

        Each entry in the vector corresponds to one of the input labels. The
        value at index j is the unique integer associated with the jth label.

        :param labels: A sequence of labels.
        :return: A vector, with one entry for each label.
        """
        return self.encoder.transform(labels)


class Classifier:
    def __init__(self, use_tfidf=False):
        """Initalizes a logistic regression classifier.
        """
        self.logistic = linear_model.LogisticRegression(max_iter=3000, solver='saga', C=0.9)
        self.idf = None
        self.use_tfidf = use_tfidf

    def train(self, features: NDArray, labels: NDArray) -> None:
        """Trains the classifier using the given training examples.

        :param features: A feature matrix, where each row represents a text.
        Such matrices will typically be generated via TextToFeatures.
        :param labels: A label vector, where each entry represents a label.
        Such vectors will typically be generated via TextToLabels.
        """
        if self.use_tfidf:
            self.logistic.fit(self.to_tdidf(features), labels)
        else:
            self.logistic.fit(features, labels)

    def predict(self, features: NDArray) -> NDArray:
        """Makes predictions for each of the given examples.

        :param features: A feature matrix, where each row represents a text.
        Such matrices will typically be generated via TextToFeatures.
        :return: A prediction vector, where each entry represents a label.
        """
        if self.use_tfidf:
            return self.logistic.predict(self.to_tdidf(features))
        else:
            return self.logistic.predict(features)

    def to_tdidf(self, features: NDArray) -> NDArray:
        """Convert features matrix into TF-iDF matrix

        :param features: document features matrix, i.e., doc-iterm frequency matrix
        :return: doc-iterm TF-iDF matrix
        """
        num_doc, num_iterm = features.shape
        tf = features.copy()  # 2-D array
        tf[tf > 0] = 1 + np.log10(tf[tf > 0])
        if self.idf is None:
            self.idf = np.array([features[:, j].count_nonzero() for j in range(num_iterm)])
            self.idf = np.log10(num_doc / self.idf).reshape(1, -1)
        tfidf = tf / (1 / self.idf)
        return sparse.csr_matrix(tfidf)
