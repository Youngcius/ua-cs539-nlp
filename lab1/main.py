from typing import List, Set, Dict, Tuple, Optional, Text
from operator import itemgetter

import spacy
from collections import Counter
from rich.console import Console

console = Console()

# load  the small english core web model from spacy.
nlp = spacy.load('en_core_web_sm')


def tokenize_text(text: Text) -> List[Text]:
    """This functions tokenize a text by iterating
            over its tokens by using spacy English tokenizer.
        =============
        Params:
            text
        Return:
            A list of tokenized items. ["token1, "token2"...etc]
    """
    return [t.text for t in nlp(text)]


def recognize_name_entity(text: Text) -> List[Text]:
    """This function recognizes name entities in a text by using Spacy
        English tokenizer.
        #NOTE: Please read about converting span (span1, span2..etc)
                into list of strings ["span1", "span2"...etc].
        =============
        Params:
            text
        Return:
            A list of tokenized name entities. ex: ['entity1', 'entity2'...etc]
    """
    return [ent.text for ent in nlp(text).ents]


def get_entity_labels(text: Text) -> List[Tuple[Text, Text]]:
    """This function obtains the labels of entities.
        #NOTE: Please read about the types of label methods in spacy. You should
                return a string label rather than an integer label.
        =============
        Params:
            text
        :Return:
            A list of tuples  of entities and its labels.
            ex: [("entity1", "label1"), ("entity2", "label2")...etc]
    """
    return [(ent.text, ent.label_) for ent in nlp(text).ents]


def get_lemmas(text: Text) -> List[Tuple[Text, Text]]:
    """This function finds lemmas in a text by using Spacy
        English tokenizer. It must return a token and its lemma.
        #NOTE: Please read about the types of lemma methods in spacy. Your should
                return a string lemma rather than an integer lemma.
        =============
        :Params:
            text
        :Return:
            A list of of tuples of tokens and their stem.
            ex: [('tokens', 'token'), ...etc]
    """
    return [(token.text, token.lemma_) for token in nlp(text)]


def get_POS_tags(text: Text) -> List[Tuple[Text, ...]]:
    """This function obtains with its the associated POS and tags of each token
        in a text by using Spacy tags.  It must return the verb itself,
        part of of speech (POS), and the associated tag.
        #NOTE: Read Spacy POS and associated tags.
    =============
        Params:
            text
        Return:
            A list of tuples of strings. ex: [('get', 'VERB', 'VB'),...etc]
    """
    return [(token.text, token.pos_, token.tag_) for token in nlp(text)]


def pos_frequency(text: Text) -> List[Tuple[Text, int]]:
    """This function returns frequency counts of part of speech (POS)
        in a text. It must return the POS and its frequency.
        #NOTE: Refer to counting in Spacy, count the frequencies of the given attributes, make a list of the dictionary of the POS
               and counts, then sort the list. Also, make sure to sort your output by the key.
        =============
        Params:
            text
        Return:
            A sorted list  of tuples of strings and integers sorted by the key. ex: [('ADV', 2),...etc]
    """

    pos_vals = [token.pos_ for token in nlp(text)]
    return sorted(Counter(pos_vals).items(), key=lambda item: item[0])


def parse_dependency(text: Text) -> List[Tuple[Text, ...]]:
    """This function parse a single sentence.
    =============
        Params:
            text
        Return:
            A list of tuples of strings. Ex: [('is', 'ROOT'), ...etc]
    """
    return [(token.text, token.dep_) for token in nlp(text)]


def count_dependency(text: Text) -> List[Tuple[Text, int]]:
    """This function extracts the dependencies of sentences and their frequencies in a text.
        It must return a parsed dependency and its frequency.
        #NOTE: Refer to counting in Spacy, count the frequencies of the given attributes, make a  list of the dictionary of the DEP
               and counts, then sort the list. Also, make sure to sort your output by the key.
        ============
        Params:
            text
        Return:
            A sorted list of tuples of strings and integers sorted by the key. Ex: [('ROOT', 1),...etc]
    """

    dep_vals = [token.dep_ for token in nlp(text)]
    return sorted(Counter(dep_vals).items(), key=lambda item: item[0])


if __name__ == '__main__':
    console.rule('Demonstrating tokenize_text()')
    sentence = "Apple, the computer company that started in a California garage in 1976, is now worth $3 trillion.Several other companies have market values of over $1 trillion,including Google parent Alphabet ($1.95 trillion) and Amazon ($1.68 trillion)."
    print(sentence)
    console.print(tokenize_text(sentence))

    console.rule('Demonstrating recognize_name_entity()')
    sentence = "Tesla chief executive Elon Musk donated a total of 5,044,000 shares in the world’s most valuable automaker to a charity from Nov. 19 to Nov. 29 last year, its filing with U.S. Securities and Exchange Commission (SEC) showed on Monday. On the other hand, Apple CEO Tim Cook last week donated more than $5 million in Apple stock to an unnamed charity, according to an SEC filing shared today."
    print(sentence)
    console.print(recognize_name_entity(sentence))

    console.rule('Demonstrating get_entity_labels()')
    sentence = "The Space Force Agency accounts for about 2.5% of total Defense Department spending. The $2.2 billion increase sought for 2022 represents a significant boost for the smallest branch of the armed forces established 18 months ago.The Pentagon said the $2.2 billion in additional funding sought for the Space Force includes new investments in space systems and much of this funding was transferred from the Air Force, Navy and Army."
    print(sentence)
    console.print(get_entity_labels(sentence))

    console.rule('Demonstrating get_lemmas()')
    sentences = "It claims the 30% cut Google takes from digital purchases on its app store is excessive and unfair. The case follows a similar one launched on behalf of iPhone users in May. Google said it competed 'vigorously and fairly' for developers and consumers and its fees were 'comparable to our competitors'. Most Android phones came pre-loaded with more than one app store. And 97% of developers paid no service fee because their apps were free."
    print(sentences)
    console.print(get_lemmas(sentences))

    console.rule('Demonstrating get_POS_tags()')
    sentence = "Tyson Foods will ease mask rules at some meat processing plants as some U.S. states are ending mask mandates. The tennis star Novak Djokovic said he was prepared to miss the French Open, Wimbledon and other tournaments if he was required to get a Covid vaccine to compete."
    print(sentence)
    console.print(get_POS_tags(sentence))

    console.rule('Demonstrating pos_frequency()')
    sentence = "Governor Doug Ducey’s plan to graduate more nurses in Arizona — and help alleviate a shortage impacting hospitals here and across the country — could have 300 new nurses on the job by 2030. The Governor plans to invest $25.7 million in a public-private partnership with Creighton University to expand the Accelerated Nursing Academy. Combined with $15.7 million from the university, the funds represent a major commitment to training the next generation of Arizona nurses."
    print(sentence)
    console.print(pos_frequency(sentence))

    console.rule('Demonstrating parse_dependency()')
    sentence = "I shared an update last week on the status of our affiliation with the University of Arizona Global Campus and the initiation of needed formal planning toward acquiring UAGC and coordinating the operations of our two universities. Provost Liesl Folks and I have asked Senior Vice Provost Gail Burd to lead this effort, and I am grateful she has accepted this role. Dr. Burd has extensive experience guiding the University’s accreditation processes and she is deeply committed  to our educational mission. Faculty and staff, please look for additional information soon from the Office of the Provost. Documents related to this process have been posted in this Box folder."
    print(sentence)
    console.print(parse_dependency(sentence))

    console.rule('Demonstrating count_dependency()')
    sentence = "It has been a busy two weeks since my last highlights email, with lots of good news for the University. As many of you have seen, Lisa Rulney and I were very pleased Thursday to announce the appointment of Paula Balafas as the next Assistant Vice President and Chief of Police for the University of Arizona. Paula will be an important part of the University’s leadership, and I look forward to her positive impact for our students and the entire University community."
    print(sentence)
    console.print(count_dependency(sentence))
