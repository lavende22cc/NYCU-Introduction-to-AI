import math
from collections import Counter, defaultdict, OrderedDict

from typing import List

import nltk
import numpy as np
from nltk.tokenize.toktok import ToktokTokenizer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.naive_bayes import GaussianNB
from tqdm import tqdm


class Ngram:
    def __init__(self, config, n=2):
        self.tokenizer = ToktokTokenizer()
        self.n = n
        self.model = None
        self.config = config

    def tokenize(self, sentence):
        '''
        E.g.,
            sentence: 'Here dog.'
            tokenized sentence: ['Here', 'dog', '.']
        '''
        return self.tokenizer.tokenize(sentence)

    def get_ngram(self, corpus_tokenize: List[List[str]]):
        '''
        Compute the co-occurrence of each pair.
        '''
        # begin your code (Part 1)
        features = None
        mode = None
        total = 0
        self.uni_dict = defaultdict(int)
        self.bi_dict = defaultdict(int)
        self.tri_dict = defaultdict(int)
        uni_list = []
        bi_list = []
        tri_list = []
        # self.corpusbj6 corpus_tokenize
        V_count = set() # a set to count how many unique vocabulary in corpus , and it will apply to 'V' value of laplace smoothing

        for line in corpus_tokenize:
            for i in range(len(line)):
                V_count.add(line[i])
                total += 1
                self.uni_dict[line[i]] += 1
                uni_list.append(line[i])
                if(i + 1 < len(line)):
                    self.bi_dict[(line[i] , line[i+1])] += 1
                    bi_list.append((line[i] , line[i+1]))
                if(i + 2 < len(line)):
                    self.tri_dict[(line[i] , line[i+1] , line[i+2])] += 1
                    tri_list.append((line[i],line[i+1],line[i+2]))

        self.V = len(V_count)
        self.total = total

        if self.n == 1:     #get unigram model
            model = defaultdict(int)
            for x in uni_list:
                model[x] += 1

            features = OrderedDict(sorted(self.uni_dict.items(), key=lambda item: -item[1])) # build a OrderDict by dict

        if self.n == 2:     #get bigram model
            model = defaultdict(lambda: defaultdict(int))
            self.bi_dict = defaultdict(int)
            for x,y in bi_list:
                model[x][y] += 1
            
            features = OrderedDict(sorted(self.bi_dict.items(), key=lambda item: -item[1])) # build a OrderDict by dict

        if self.n == 3:     #get trigram model
            model = defaultdict(lambda: defaultdict(int))
            self.tri_dict = defaultdict(int)
            for x,y,z in tri_list:
                model[(x,y)][z] += 1

            self.features = OrderedDict(sorted(self.tri_dict.items(), key=lambda item: -item[1])) # build a OrderDict by dict

        return model , features
    
    def train(self, df):
        '''
        Train n-gram model.
        '''
        corpus = [['[CLS]'] + self.tokenize(document) for document in df['review']]     # [CLS] represents start of sequence
        
        # You may need to change the outputs, but you need to keep self.model at least.
        self.model, self.features = self.get_ngram(corpus)

    def compute_perplexity(self, df_test) -> float:
        '''
        Compute the perplexity of n-gram model.
        Perplexity = 2^(-entropy)
        '''
        if self.model is None:
            raise NotImplementedError("Train your model first")

        corpus = [['[CLS]'] + self.tokenize(document) for document in df_test['review']]
        
        '''
        begin your code (Part 2)
        M indicate number of grams
        l indicate the entropy
        A,B will aplly to count perplexity
        '''
        l, M, A, B = 0, 0, 0, 0

        if self.n == 1:
            for line in corpus:
                M += len(line)
                for i in range(len(line)):
                    # get no gram
                    gram = line[i]

                    # count A/B
                    A = self.uni_dict[gram] + 1
                    B = self.total
                    l += math.log2(A/B)

        if self.n == 2:
           for line in corpus:
                M += len(line)-1
                for i in range(len(line)-1):
                    # get no gram
                    gram = (line[i],line[i+1])

                    # count A/B
                    A = self.bi_dict[gram] + 1
                    B = self.uni_dict[line[i]] + len(self.uni_dict.keys())
                    l += math.log2(A/B)


        if self.n == 3:
            for line in corpus:
                M += max(0,len(line)-2)
                for i in range(len(line)-2):
                    # get no gram
                    gram = (line[i],line[i+1],line[i+2])

                    # count A/B
                    A = self.tri_dict[gram] + 1
                    B = self.bi_dict[(line[i],line[i+1])] + len(self.bi_dict.keys())
                    l += math.log2(A/B)
        l /= M
        perplexity = pow(2,-l)

        # end your code

        return perplexity

    def train_sentiment(self, df_train, df_test):
        '''
        Use the most n patterns as features for training Naive Bayes.
        It is optional to follow the hint we provided, but need to name as the same.

        Parameters:
            train_corpus_embedding: array-like of shape (n_samples_train, n_features)
            test_corpus_embedding: array-like of shape (n_samples_train, n_features)
        
        E.g.,
            Assume the features are [(I saw), (saw a), (an apple)],
            the embedding of the tokenized sentence ['[CLS]', 'I', 'saw', 'a', 'saw', 'saw', 'a', 'saw', '.'] will be
            [1, 2, 0]
            since the bi-gram of the sentence contains
            [([CLS] I), (I saw), (saw a), (a saw), (saw saw), (saw a), (a saw), (saw .)]
            The number of (I saw) is 1, the number of (saw a) is 2, and the number of (an apple) is 0.
        '''
        # begin your code (Part 3)

        # step 1. select the most feature_num patterns as features, you can adjust feature_num for better score!
        feature_num = 500

        # step 2. convert each sentence in both training data and testing data to embedding.
        # Note that you should name "train_corpus_embedding" and "test_corpus_embedding" for feeding the model.

        # end your code

        # feed converted embeddings to Naive Bayes
        nb_model = GaussianNB()
        nb_model.fit(train_corpus_embedding, df_train['sentiment'])
        y_predicted = nb_model.predict(test_corpus_embedding)
        precision, recall, f1, support = precision_recall_fscore_support(df_test['sentiment'], y_predicted, average='macro', zero_division=1)
        precision = round(precision, 4)
        recall = round(recall, 4)
        f1 = round(f1, 4)
        print(f"F1 score: {f1}, Precision: {precision}, Recall: {recall}")


if __name__ == '__main__':
    '''
    Here is TA's answer of part 1 for reference only.
    {'a': 0.5, 'saw: 0.25, '.': 0.25}

    Explanation:
    (saw -> a): 2
    (saw -> saw): 1
    (saw -> .): 1
    So the probability of the following word of 'saw' should be 1 normalized by 2+1+1.

    P(I | [CLS]) = 1
    P(saw | I) = 1; count(saw | I) / count(I)
    P(a | saw) = 0.5
    P(saw | a) = 1.0
    P(saw | saw) = 0.25
    P(. | saw) = 0.25
    '''

    # unit test
    test_sentence = {'review': ['I saw a saw saw a saw.']}
    model = Ngram(2)
    model.train(test_sentence)
    print(model.model['saw'])
    print("Perplexity: {}".format(model.compute_perplexity(test_sentence)))
