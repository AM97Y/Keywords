import re

from text_rank import TextRankKeyword
from tfidf import TfidfKeywords


def check_keywords_compliance(text: str, keywords: set[str], algoritm: str = 'TfiDF') -> bool:
    """
    This function checks if the text matches the keywords.
    :param algoritm:
    :param text: Text for check.
    :param keywords: Key words for check.
    :return: Checked or not.
    """
    found_keywords = []
    if algoritm == 'TfiDF':
        tfidf = TfidfKeywords()
        found_keywords = tfidf.get_keywords(text=text, number=len(keywords) * 3)
    else:
        tr = TextRankKeyword()
        found_keywords = tr.get_keywords(text, candidate_pos=['NOUN', 'PROPN'], window_size=4,
                                         lower=True, number=len(keywords) * 3)

    return _check_th_keywords(keywords=keywords, found_keywords=found_keywords, th=0.5)


def _check_th_keywords(keywords: set[str], found_keywords: dict, th: float = 0.5) -> bool:
    """

    :param keywords:
    :param found_keywords:
    :param th:
    :return:
    """
    count = sum([1. for key in keywords if key in found_keywords])

    if count / len(keywords) >= th:
        return True
    else:
        return False


def _remove_urls(text: str) -> str:
    return re.sub(r"http://\S+|https://\S+", "", text)


if __name__ == '__main__':
    text = 'In this paper, we present YAKE!, a novel feature-based system for multi-lingual keyword extraction from ' \
           'single documents, which supports texts of different sizes, domains or languages. Unlike most systems, ' \
           'YAKE! does not rely on dictionaries or thesauri, neither it is trained against any corpora. Instead, ' \
           'we follow an unsupervised approach which builds upon features extracted from the text, making it thus ' \
           'applicable to documents written in many different languages without the need for external knowledge. This ' \
           'can be beneficial for a large number of tasks and a plethora of situations where the access to training ' \
           'corpora is either limited or restricted. In this demo, we offer an easy to use, interactive session, ' \
           'where users from both academia and industry can try our system, either by using a sample document or by ' \
           'introducing their own text. As an add-on, we compare our extracted keywords against the output produced ' \
           'by the IBM Natural Language Understanding (IBM NLU) and Rake system. YAKE! demo is available at ' \
           'http://bit.ly/YakeDemoECIR2018. A python implementation of YAKE! is also available at PyPi repository (' \
           'https://pypi. python.org/pypi/yake/). '
    text = '''The Wandering Earth, described as China’s first big-budget science fiction thriller, quietly made it 
    onto screens at AMC theaters in North America this weekend, and it shows a new side of Chinese filmmaking — one 
    focused toward futuristic spectacles rather than China’s traditionally grand, massive historical epics. At the 
    same time, The Wandering Earth feels like a throwback to a few familiar eras of American filmmaking. While the 
    film’s cast, setting, and tone are all Chinese, longtime science fiction fans are going to see a lot on the 
    screen that reminds them of other movies, for better or worse. '''

    text = "A new Japanese film premiered yesterday"
    text = 'In this paper, we present YAKE!, a novel feature-based system for multi-lingual keyword extraction from ' \
           'single documents, which supports texts of different sizes, domains or languages. Unlike most systems, ' \
           'YAKE! does not rely on dictionaries or thesauri, neither it is trained against any corpora. Instead, ' \
           'we follow an unsupervised approach which builds upon features extracted from the text, making it thus ' \
           'applicable to documents written in many different languages without the need for external knowledge. This ' \
           'can be beneficial for a large number of tasks and a plethora of situations where the access to training ' \
           'corpora is either limited or restricted. In this demo, we offer an easy to use, interactive session, ' \
           'where users from both academia and industry can try our system, either by using a sample document or by ' \
           'introducing their own text. As an add-on, we compare our extracted keywords against the output produced ' \
           'by the IBM Natural Language Understanding (IBM NLU) and Rake system. YAKE! demo is available at ' \
           'http://bit.ly/YakeDemoECIR2018. A python implementation of YAKE! is also available at PyPi repository (' \
           'https://pypi. python.org/pypi/yake/). '
    # TEXT2 = "Most viewers left positive reviews about the movie"
    # TEXT3 = "Soon the masterpiece will be available in all cinemas"
    text = _remove_urls(text=text)

    key_words = check_keywords_compliance(text=text, keywords=('text', 'system', 'python'))
    print(key_words)

    key_words = check_keywords_compliance(text=text, keywords=('text', 'system', 'python'), algoritm='TextRank')
    print(key_words)
