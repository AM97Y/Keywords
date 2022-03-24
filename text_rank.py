from collections import OrderedDict

import numpy as np
import spacy

# https://www.machinelearningmastery.ru/textrank-for-keyword-extraction-by-python-c0bae21bcec0/
# python -m spacy download en_core_web_sm
nlp = spacy.load('en_core_web_sm')


class TextRankKeyword:
    """
    Extract keywords from text.
    """

    def __init__(self):
        self.d = 0.85  # damping coefficient, usually is .85
        self.min_diff = 1e-5  # convergence threshold
        self.steps = 10  # iteration steps
        self.node_weight = None  # save keywords and its weight

    def get_keywords(self, text: str,
                     candidate_pos: list[str] = ['NOUN', 'PROPN'],
                     window_size: int = 4, lower: bool = True, number: int = 10) -> list[str]:
        """
        Print top number keywords
        :param text:
        :param lower:
        :param window_size:
        :param candidate_pos:
        :param number: List of top keywords.
        :return:
        """
        self._analyze(text, candidate_pos=candidate_pos, window_size=window_size, lower=lower)

        result = []
        node_weight = OrderedDict(sorted(self.node_weight.items(), key=lambda t: t[1], reverse=True))
        for i, (key, value) in enumerate(node_weight.items()):
            result.append(key)
            if i > number:
                return result
        return result

    def _sentence_segment(self, doc, candidate_pos, lower):
        """
        Store those words only in cadidate_pos. Filter sentences.
        :param doc: Text for analize.
        :param candidate_pos: Part of speach for chose words.
        :param lower: Lower word case or not.
        :return: Sentences.
        """
        sentences = []
        for sent in doc.sents:
            selected_words = []
            for token in sent:
                if token.pos_ in candidate_pos and token.is_stop is False:
                    if lower is True:
                        selected_words.append(token.text.lower())
                    else:
                        selected_words.append(token.text)
            sentences.append(selected_words)
        return sentences

    def _get_vocab(self, sentences):
        """
        Build vocabulary.
        :param sentences: Text.
        :return: Vocab.
        """
        vocab = OrderedDict()
        i = 0
        for sentence in sentences:
            for word in sentence:
                if word not in vocab:
                    vocab[word] = i
                    i += 1
        return vocab

    def _get_token_pairs(self, window_size, sentences):
        """
        Build token_pairs from windows in sentences.
        :param window_size:
        :param sentences:
        :return:
        """
        token_pairs = list()
        for sentence in sentences:
            for i, word in enumerate(sentence):
                for j in range(i + 1, i + window_size):
                    if j >= len(sentence):
                        break
                    pair = (word, sentence[j])
                    if pair not in token_pairs:
                        token_pairs.append(pair)
        return token_pairs

    def _symmetrize(self, a):
        """
        Get symmeric matrix.
        :param a: Matrix.
        :return: Symmeric matrix
        """
        return a + a.T - np.diag(a.diagonal())

    def _get_matrix(self, vocab, token_pairs):
        """
        Get normalized matrix.
        :param vocab:
        :param token_pairs:
        :return:
        """
        # Build matrix
        vocab_size = len(vocab)
        g = np.zeros((vocab_size, vocab_size), dtype='float')
        for word1, word2 in token_pairs:
            i, j = vocab[word1], vocab[word2]
            g[i][j] = 1

        g = self._symmetrize(g)

        # Normalize matrix by column
        norm = np.sum(g, axis=0)
        g_norm = np.divide(g, norm, where=norm != 0)  # this is ignore the 0 element in norm

        return g_norm

    def _analyze(self, text,
                 candidate_pos,
                 window_size, lower):
        """
        Main function to analyze text.
        :param text:
        :param candidate_pos:
        :param window_size:
        :param lower:
        :return:
        """

        # Pare text by spaCy.
        doc = nlp(text)

        sentences = self._sentence_segment(doc, candidate_pos, lower)  # list of list of words
        vocab = self._get_vocab(sentences)
        token_pairs = self._get_token_pairs(window_size, sentences)
        g = self._get_matrix(vocab, token_pairs)

        # Initionlization for weight (pagerank value).
        pagerank_weights = np.array([1] * len(vocab))

        # Iteration
        previous_pagerank_weight = 0
        for epoch in range(self.steps):
            pagerank_weights = (1 - self.d) + self.d * np.dot(g, pagerank_weights)
            if abs(previous_pagerank_weight - sum(pagerank_weights)) < self.min_diff:
                break
            else:
                previous_pagerank_weight = sum(pagerank_weights)

        # Get weight for each node
        node_weight = dict()
        for word, index in vocab.items():
            node_weight[word] = pagerank_weights[index]

        self.node_weight = node_weight
