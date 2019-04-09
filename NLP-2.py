from nltk.corpus import names
import nltk
import random


def gender_features(word):
    return {'last_letter': word[-1]}


labeled_names = ([(name, 'male') for name in names.words('male.txt')] + [(name, 'female') for name in names.words('female.txt')])
random.shuffle(labeled_names)

featuresets = [(gender_features(n), gender) for (n, gender) in labeled_names]
train_set, test_set = featuresets[500:], featuresets[:500]
classifier = nltk.NaiveBayesClassifier.train(train_set)

print(classifier.classify(gender_features('Sherlock')))

#print(len(names.words()))
'''
#labeled_names = ([(name,'male') for name in name.words('male.txt')]+ [(name,'female') for name in name.words('female.txt')])

def featuresOfName(name):
    return name[-1]

featuresOfName("Sherlock")
'''
