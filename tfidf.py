import string

import nltk
import numpy
import scipy
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


# nltk.download()
# https://pythonprogramming.net/part-of-speech-tagging-nltk-tutorial/

# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4')
class LemmaTokenizer:
    def __init__(self):
        self.wnl = WordNetLemmatizer()
        self.pos_tags = ['NN',
                         'NNS',
                         'NNP',
                         'NNPS', ]

    def __call__(self, articles: str) -> list:
        return [self.wnl.lemmatize(i[0]) for i in nltk.pos_tag(word_tokenize(articles))
                if i[0].lower() not in string.punctuation and i[0] not in stopwords.words('english')
                and i[1] in self.pos_tags]


class TfidfKeywords:
    def get_keywords(self, text: str, number: int = 10) -> bool:
        """
        This function checks if the text matches the keywords.
        :param text: Text for check.
        :param keywords: Key words for check.
        :return: Checked or not.
        """
        cv = CountVectorizer(tokenizer=LemmaTokenizer())
        word_count_vector = cv.fit_transform(text.split())

        tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
        tfidf_transformer.fit(word_count_vector)

        feature_names = cv.get_feature_names_out()
        # Generate tf-idf for the given document.
        tf_idf_vector = tfidf_transformer.transform(cv.transform([text]))
        sorted_items = self._sort_coo(tf_idf_vector.tocoo())
        found_keywords = self._extract_top_keywords(feature_names=feature_names, sorted_items=sorted_items,
                                                    number=number)

        print("\nKeywords")
        for elem in found_keywords:
            print(elem, found_keywords[elem])

        return found_keywords

    def _sort_coo(self, coo_matrix: scipy.sparse.coo.coo_matrix) -> list:
        """
        Sort the tf-idf vectors by descending order of scores.
        :param coo_matrix: A sparse matrix in COOrdinate format.
        :return: List of words and their scores.
        """
        tuples = zip(coo_matrix.col, coo_matrix.data)
        return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

    def _extract_top_keywords(self, feature_names: numpy.ndarray, sorted_items: list[tuple[int, int]],
                              number: int = 10) -> dict:
        """
        Get the feature names and tf-idf score of top n items.

        :param number: Top number of keywords.
        :param feature_names: List of features name.
        :param sorted_items: Sorted items by func sort_coo.
        :return:
        """
        # use only top n items from vector
        sorted_items = sorted_items[:number]

        score_vals = []
        feature_vals = []

        for i, score in sorted_items:
            # keep track of feature name and its corresponding score
            score_vals.append(round(score, 3))
            feature_vals.append(feature_names[i])

        # create a tuples of feature,score
        results = {}
        for idx in range(len(feature_vals)):
            results[feature_vals[idx]] = score_vals[idx]
        return results

    def _get_keywords(self, text: str, number: int = 10) -> bool:
        """
        This function checks if the text matches the keywords.
        :param text: Text for check.
        :param keywords: Key words for check.
        :return: Checked or not.
        """
        cv = CountVectorizer(tokenizer=LemmaTokenizer())
        word_count_vector = cv.fit_transform(text.split())

        tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
        tfidf_transformer.fit(word_count_vector)

        feature_names = cv.get_feature_names_out()
        # Generate tf-idf for the given document.
        tf_idf_vector = tfidf_transformer.transform(cv.transform([text]))
        sorted_items = self._sort_coo(tf_idf_vector.tocoo())
        found_keywords = self._extract_top_keywords(feature_names=feature_names, sorted_items=sorted_items,
                                                    number=number)
        print("\nKeywords")
        for elem in found_keywords:
            print(elem, found_keywords[elem])

        return found_keywords
