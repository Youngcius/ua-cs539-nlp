import main


def test_tokenize_text():
    sentence =  "Apple, the computer company that started in a California garage \
                in 1976, is now worth $3 trillion.Several other companies have market values \
                of over $1 trillion,including Google parent Alphabet ($1.95 trillion) \
                and Amazon ($1.68 trillion)."
    expected_tokens = [
         'Apple',
         ',',
         'the',
         'computer',
         'company',
         'that',
         'started',
         'in',
         'a',
         'California',
         'garage',
         '                ',
         'in',
         '1976',
         ',',
         'is',
         'now',
         'worth',
         '$',
         '3',
         'trillion',
         '.',
         'Several',
         'other',
         'companies',
         'have',
         'market',
         'values',
         '                ',
         'of',
         'over',
         '$',
         '1',
         'trillion',
         ',',
         'including',
         'Google',
         'parent',
         'Alphabet',
         '(',
         '$',
         '1.95',
         'trillion',
         ')',
         '                ',
         'and',
         'Amazon',
         '(',
         '$',
         '1.68',
         'trillion',
         ')',
         '.'
    ]
    assert main.tokenize_text(sentence) == expected_tokens


def test_recognize_name_entity():
    sentence = "Tesla chief executive Elon Musk donated a total \
            of 5,044,000 shares in the world’s most valuable automaker \
            to a charity from Nov. 19 to Nov. 29 last year, its filing \
            with U.S. Securities and Exchange Commission (SEC) showed on \
            Monday. On the other hand, Apple CEO Tim Cook last week donated \
            more than $5 million in Apple stock to an unnamed charity, \
            according to an SEC filing shared today."
    expected_tokens = [
         'Elon Musk',
         '5,044,000',
         'Nov. 19 to Nov. 29 last year',
         'U.S. Securities and Exchange Commission',
         'SEC',
         'Monday',
         'Apple',
         'Tim Cook',
         'last week',
         'more than $5 million',
         'Apple',
         'SEC',
         'today'
    ]
    assert main.recognize_name_entity(sentence) == expected_tokens


def test_get_entity_labels():
    sentence = "The Space Force Agency accounts for about 2.5% of \
                total Defense Department spending. The $2.2 billion \
                increase sought for 2022 represents a significant \
                boost for the smallest branch of the armed forces \
                established 18 months ago.The Pentagon said the \
                $2.2 billion in additional funding sought for the \
                Space Force includes new investments in space systems \
                and much of this funding was transferred from the Air Force, \
                Navy and Army."
    expected_tokens = [
         ('The Space Force Agency', 'ORG'),
         ('about 2.5%', 'PERCENT'),
         ('Defense Department', 'ORG'),
         ('$2.2 billion', 'MONEY'),
         ('2022', 'DATE'),
         ('18 months ago', 'DATE'),
         ('Pentagon', 'ORG'),
         ('$2.2 billion', 'MONEY'),
         ('the                 Space Force', 'ORG'),
         ('the Air Force', 'ORG'),
         ('Navy', 'ORG'),
         ('Army', 'ORG')
    ]
    assert main.get_entity_labels(sentence) == expected_tokens


def test_get_lemmas():
    sentence = "It claims the 30% cut Google takes from digital purchases \
                on its app store is excessive and unfair. The case follows \
                a similar one launched on behalf of iPhone users in May. \
                Google said it competed 'vigorously and fairly' for developers \
                and consumers and its fees were 'comparabl to our competitors'. \
                Most Android phones came pre-loaded with more than one app store. \
                And 97% of developers paid no service fee because their apps were free."
    expected_output = [
         ('It', 'it'),
         ('claims', 'claim'),
         ('the', 'the'),
         ('30', '30'),
         ('%', '%'),
         ('cut', 'cut'),
         ('Google', 'Google'),
         ('takes', 'take'),
         ('from', 'from'),
         ('digital', 'digital'),
         ('purchases', 'purchase'),
         ('                ', '                '),
         ('on', 'on'),
         ('its', 'its'),
         ('app', 'app'),
         ('store', 'store'),
         ('is', 'be'),
         ('excessive', 'excessive'),
         ('and', 'and'),
         ('unfair', 'unfair'),
         ('.', '.'),
         ('The', 'the'),
         ('case', 'case'),
         ('follows', 'follow'),
         ('                ', '                '),
         ('a', 'a'),
         ('similar', 'similar'),
         ('one', 'one'),
         ('launched', 'launch'),
         ('on', 'on'),
         ('behalf', 'behalf'),
         ('of', 'of'),
         ('iPhone', 'iPhone'),
         ('users', 'user'),
         ('in', 'in'),
         ('May.', 'May.'),
         ('                ', '                '),
         ('Google', 'Google'),
         ('said', 'say'),
         ('it', 'it'),
         ('competed', 'compete'),
         ("'", "'"),
         ('vigorously', 'vigorously'),
         ('and', 'and'),
         ('fairly', 'fairly'),
         ("'", "'"),
         ('for', 'for'),
         ('developers', 'developer'),
         ('                ', '                '),
         ('and', 'and'),
         ('consumers', 'consumer'),
         ('and', 'and'),
         ('its', 'its'),
         ('fees', 'fee'),
         ('were', 'be'),
         ("'", "'"),
         ('comparabl', 'comparabl'),
         ('to', 'to'),
         ('our', 'our'),
         ('competitors', 'competitor'),
         ("'", "'"),
         ('.', '.'),
         ('                ', '                '),
         ('Most', 'Most'),
         ('Android', 'Android'),
         ('phones', 'phone'),
         ('came', 'come'),
         ('pre', 'pre'),
         ('-', '-'),
         ('loaded', 'loaded'),
         ('with', 'with'),
         ('more', 'more'),
         ('than', 'than'),
         ('one', 'one'),
         ('app', 'app'),
         ('store', 'store'),
         ('.', '.'),
         ('                ', '                '),
         ('And', 'and'),
         ('97', '97'),
         ('%', '%'),
         ('of', 'of'),
         ('developers', 'developer'),
         ('paid', 'pay'),
         ('no', 'no'),
         ('service', 'service'),
         ('fee', 'fee'),
         ('because', 'because'),
         ('their', 'their'),
         ('apps', 'app'),
         ('were', 'be'),
         ('free', 'free'),
         ('.', '.')
                      ]
    assert main.get_lemmas(sentence) == expected_output

    
def test_get_POS_tags():
    sentence = "Tyson Foods will ease mask rules at some meat processing \
                plants as some U.S. states are ending mask mandates. \
                The tennis star Novak Djokovic said he was prepared \
                to miss the French Open, Wimbledon and other tournaments \
                if he was required to get a Covid vaccine to compete."
    expected_output = [
        ('Tyson', 'PROPN', 'NNP'),
        ('Foods', 'PROPN', 'NNPS'),
        ('will', 'AUX', 'MD'),
        ('ease', 'VERB', 'VB'),
        ('mask', 'NOUN', 'NN'),
        ('rules', 'NOUN', 'NNS'),
        ('at', 'ADP', 'IN'),
        ('some', 'DET', 'DT'),
        ('meat', 'NOUN', 'NN'),
        ('processing', 'NOUN', 'NN'),
        ('                ', 'SPACE', '_SP'),
        ('plants', 'NOUN', 'NNS'),
        ('as', 'SCONJ', 'IN'),
        ('some', 'DET', 'DT'),
        ('U.S.', 'PROPN', 'NNP'),
        ('states', 'NOUN', 'NNS'),
        ('are', 'AUX', 'VBP'),
        ('ending', 'VERB', 'VBG'),
        ('mask', 'NOUN', 'NN'),
        ('mandates', 'NOUN', 'NNS'),
        ('.', 'PUNCT', '.'),
        ('                ', 'SPACE', '_SP'),
        ('The', 'DET', 'DT'),
        ('tennis', 'NOUN', 'NN'),
        ('star', 'NOUN', 'NN'),
        ('Novak', 'PROPN', 'NNP'),
        ('Djokovic', 'PROPN', 'NNP'),
        ('said', 'VERB', 'VBD'),
        ('he', 'PRON', 'PRP'),
        ('was', 'AUX', 'VBD'),
        ('prepared', 'VERB', 'VBN'),
        ('                ', 'SPACE', '_SP'),
        ('to', 'PART', 'TO'),
        ('miss', 'VERB', 'VB'),
        ('the', 'DET', 'DT'),
        ('French', 'PROPN', 'NNP'),
        ('Open', 'PROPN', 'NNP'),
        (',', 'PUNCT', ','),
        ('Wimbledon', 'PROPN', 'NNP'),
        ('and', 'CCONJ', 'CC'),
        ('other', 'ADJ', 'JJ'),
        ('tournaments', 'NOUN', 'NNS'),
        ('                ', 'SPACE', '_SP'),
        ('if', 'SCONJ', 'IN'),
        ('he', 'PRON', 'PRP'),
        ('was', 'AUX', 'VBD'),
        ('required', 'VERB', 'VBN'),
        ('to', 'PART', 'TO'),
        ('get', 'VERB', 'VB'),
        ('a', 'DET', 'DT'),
        ('Covid', 'PROPN', 'NNP'),
        ('vaccine', 'NOUN', 'NN'),
        ('to', 'PART', 'TO'),
        ('compete', 'VERB', 'VB'),
        ('.', 'PUNCT', '.')
    ]
    assert main.get_POS_tags(sentence) == expected_output

    
def test_pos_frequency():
    sentence = "Governor Doug Ducey’s plan to graduate more nurses in \
                Arizona — and help alleviate a shortage impacting hospitals \
                here and across the country — could have 300 new nurses on \
                the job by 2030. The Governor plans to invest $25.7 million \
                in a public-private partnership with Creighton University to \
                expand the Accelerated Nursing Academy. Combined with $15.7 million \
                from the university, the funds represent a major commitment \
                to training the next generation of Arizona nurses."
    expected_output = [
        ('ADJ', 6),
        ('ADP', 10),
        ('ADV', 1),
        ('AUX', 1),
        ('CCONJ', 2),
        ('DET', 10),
        ('NOUN', 13),
        ('NUM', 6),
        ('PART', 4),
        ('PROPN', 11),
        ('PUNCT', 7),
        ('SPACE', 7),
        ('SYM', 2),
        ('VERB', 11)
    ]
    assert main.pos_frequency(sentence) == expected_output


def test_parse_dependency():
    sentence = "I shared an update last week on the status of our \
                affiliation with the University of Arizona Global Campus \
                and the initiation of needed formal planning toward acquiring \
                UAGC and coordinating the operations of our two universities. \
                Provost Liesl Folks and I have asked Senior Vice Provost Gail \
                Burd to lead this effort, and I am grateful she has accepted \
                this role. Dr. Burd has extensive experience guiding the \
                University’s accreditation processes and she is deeply \
                committed  to our educational mission. Faculty and staff, \
                please look for additional information soon from the Office \
                of the Provost. Documents related to this process have been \
                posted in this Box folder."
    expected_output = [
        ('I', 'nsubj'),
        ('shared', 'ROOT'),
        ('an', 'det'),
        ('update', 'dobj'),
        ('last', 'amod'),
        ('week', 'npadvmod'),
        ('on', 'prep'),
        ('the', 'det'),
        ('status', 'pobj'),
        ('of', 'prep'),
        ('our', 'poss'),
        ('                ', 'dep'),
        ('affiliation', 'pobj'),
        ('with', 'prep'),
        ('the', 'det'),
        ('University', 'nmod'),
        ('of', 'prep'),
        ('Arizona', 'pobj'),
        ('Global', 'compound'),
        ('Campus', 'compound'),
        ('                ', 'dep'),
        ('and', 'cc'),
        ('the', 'det'),
        ('initiation', 'conj'),
        ('of', 'prep'),
        ('needed', 'amod'),
        ('formal', 'amod'),
        ('planning', 'pobj'),
        ('toward', 'prep'),
        ('acquiring', 'pcomp'),
        ('                ', 'dep'),
        ('UAGC', 'dobj'),
        ('and', 'cc'),
        ('coordinating', 'conj'),
        ('the', 'det'),
        ('operations', 'dobj'),
        ('of', 'prep'),
        ('our', 'poss'),
        ('two', 'nummod'),
        ('universities', 'pobj'),
        ('.', 'punct'),
        ('                ', 'dep'),
        ('Provost', 'compound'),
        ('Liesl', 'compound'),
        ('Folks', 'nsubj'),
        ('and', 'cc'),
        ('I', 'conj'),
        ('have', 'aux'),
        ('asked', 'ROOT'),
        ('Senior', 'compound'),
        ('Vice', 'compound'),
        ('Provost', 'compound'),
        ('Gail', 'dobj'),
        ('                ', 'dep'),
        ('Burd', 'dobj'),
        ('to', 'aux'),
        ('lead', 'xcomp'),
        ('this', 'det'),
        ('effort', 'dobj'),
        (',', 'punct'),
        ('and', 'cc'),
        ('I', 'nsubj'),
        ('am', 'conj'),
        ('grateful', 'acomp'),
        ('she', 'nsubj'),
        ('has', 'aux'),
        ('accepted', 'ccomp'),
        ('                ', 'dep'),
        ('this', 'det'),
        ('role', 'dobj'),
        ('.', 'punct'),
        ('Dr.', 'compound'),
        ('Burd', 'nsubj'),
        ('has', 'ROOT'),
        ('extensive', 'amod'),
        ('experience', 'dobj'),
        ('guiding', 'acl'),
        ('the', 'det'),
        ('                ', 'dep'),
        ('University', 'nmod'),
        ('’s', 'case'),
        ('accreditation', 'compound'),
        ('processes', 'dobj'),
        ('and', 'cc'),
        ('she', 'nsubj'),
        ('is', 'conj'),
        ('deeply', 'advmod'),
        ('                ', 'dep'),
        ('committed', 'acomp'),
        (' ', 'dep'),
        ('to', 'prep'),
        ('our', 'poss'),
        ('educational', 'amod'),
        ('mission', 'pobj'),
        ('.', 'punct'),
        ('Faculty', 'nsubj'),
        ('and', 'cc'),
        ('staff', 'conj'),
        (',', 'punct'),
        ('                ', 'dep'),
        ('please', 'intj'),
        ('look', 'ROOT'),
        ('for', 'prep'),
        ('additional', 'amod'),
        ('information', 'pobj'),
        ('soon', 'advmod'),
        ('from', 'prep'),
        ('the', 'det'),
        ('Office', 'compound'),
        ('                ', 'dep'),
        ('of', 'prep'),
        ('the', 'det'),
        ('Provost', 'pobj'),
        ('.', 'punct'),
        ('Documents', 'nsubjpass'),
        ('related', 'acl'),
        ('to', 'prep'),
        ('this', 'det'),
        ('process', 'pobj'),
        ('have', 'aux'),
        ('been', 'auxpass'),
        ('                ', 'dep'),
        ('posted', 'ROOT'),
        ('in', 'prep'),
        ('this', 'det'),
        ('Box', 'compound'),
        ('folder', 'pobj'),
        ('.', 'punct')
    ]
    assert main.parse_dependency(sentence) == expected_output

    
def test_count_dependency():
    sentence =  "It has been a busy two weeks since my last highlights email, \
                with lots of good news for the University. As many of you have seen, \
                Lisa Rulney and I were very pleased Thursday to announce the \
                appointment of Paula Balafas as the next Assistant Vice President \
                and Chief of Police for the University of Arizona. Paula will be an \
                important part of the University’s leadership, and I look forward to \
                her positive impact for our students and the entire University community."  
    expected_output = [
          ('ROOT', 3),
         ('acomp', 1),
         ('advcl', 2),
         ('advmod', 2),
         ('amod', 7),
         ('attr', 2),
         ('aux', 5),
         ('case', 1),
         ('cc', 4),
         ('compound', 6),
         ('conj', 4),
         ('dep', 6),
         ('det', 8),
         ('dobj', 2),
         ('mark', 1),
         ('npadvmod', 1),
         ('nsubj', 5),
         ('nummod', 1),
         ('pobj', 12),
         ('poss', 4),
         ('prep', 12),
         ('punct', 6)
    ]
    assert main.count_dependency(sentence) == expected_output

    
