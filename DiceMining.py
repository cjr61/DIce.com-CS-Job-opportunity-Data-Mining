#this is the main file for the dice.com project
import csv
import nltk
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.chunk import conlltags2tree, tree2conlltags
import pandas as pd
from collections import Counter
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline

df = pd.read_csv('modifieddataset2.csv')

#print(df.isnull().sum())
#df = df.dropna()
#df = df[df.skills !='Null']
#print( df[df['skills'].str.contains('Null')])
#df = df[df.skills !='null']
#print ('Data with null or Null values')
#print( df[df['skills'].str.contains('null')])
#df.to_csv('modifieddataset2.csv',index=False)


#print( df[df['skills'].str.contains('See')])

address = df["joblocation_address"].tolist()
rawDescription = df["jobdescription"].tolist()
description = ["".join(map(str, lst)) for lst in rawDescription]
states=[]
cities=[]
i=1
for x in address:
    i=i+1
    try:
        city,state=x.split(',')
    except ValueError:
        print "skip"
    states.append(state)
    cities.append(city)
    # if i==50:
    #   break
statesFrequency = dict(Counter(states))
citiesFrequency = dict(Counter(cities))
print ('states frequency')
print (statesFrequency)
print ('cities frequency')
print (citiesFrequency) 






# BELOW IS TOKENS & TAGGING






tagged_sentences = nltk.corpus.treebank.tagged_sents()


def features(sentence, index):
    """ sentence: [w1, w2, ...], index: the index of the word """
    return {
        'word': sentence[index],
        'is_first': index == 0,
        'is_last': index == len(sentence) - 1,
        'is_capitalized': sentence[index][0].upper() == sentence[index][0],
        'is_all_caps': sentence[index].upper() == sentence[index],
        'is_all_lower': sentence[index].lower() == sentence[index],
        'prefix-1': sentence[index][0],
        'prefix-2': sentence[index][:2],
        'prefix-3': sentence[index][:3],
        'suffix-1': sentence[index][-1],
        'suffix-2': sentence[index][-2:],
        'suffix-3': sentence[index][-3:],
        'prev_word': '' if index == 0 else sentence[index - 1],
        'next_word': '' if index == len(sentence) - 1 else sentence[index + 1],
        'has_hyphen': '-' in sentence[index],
        'is_numeric': sentence[index].isdigit(),
        'capitals_inside': sentence[index][1:].lower() != sentence[index][1:]
    }

def untag(tagged_sentence):
    return [w for w, t in tagged_sentence]



# Split the dataset for training and testing
cutoff = int(.75 * len(tagged_sentences))
training_sentences = tagged_sentences[:cutoff]
test_sentences = tagged_sentences[cutoff:]

print len(training_sentences)  # 2935
print len(test_sentences)  # 979


def transform_to_dataset(tagged_sentences):
    X, y = [], []

    for tagged in tagged_sentences:
        for index in range(len(tagged)):
            X.append(features(untag(tagged), index))
            y.append(tagged[index][1])

    return X, y


def pos_tag(sentence):
    tagged_sentence = []
    tags = clf.predict([features(sentence, index) for index in range(len(sentence))])
    return zip(sentence, tags)


X, y = transform_to_dataset(training_sentences)

clf = Pipeline([
    ('vectorizer', DictVectorizer(sparse=False)),
    ('classifier', DecisionTreeClassifier(criterion='entropy'))
])

clf.fit(X[:10000],
        y[:10000])  # Use only the first 10K samples if you're running it multiple times. It takes a fair bit :)

print 'Training completed'

X_test, y_test = transform_to_dataset(test_sentences)

print "Accuracy:", clf.score(X_test, y_test)

# print pos_tag(word_tokenize(testSentence))
# print description[1]
# temp = description[1].decode("utf8")
# taggedData = pos_tag(word_tokenize(temp))
# taggedNouns = [s for s in taggedData if s[1] == 'NN' or s[1] == 'NNP']
# print taggedNouns


#TODO: switch for different languages, counter for locations and their skills in a nxm grid

techSkills = ["c++", "c", "python", "sql", "nosql", "database", "java", "javascript", "php", "swift", "ios", "ruby",
              "mining", "big data", "agile", "android", "css", "html"]
outCSV = open("States_Skills.csv", "w+")
untaggedSkills = []
allSkills = []
joined = []

outCSV.write("0,")
for i in range(0, len(techSkills)):
    if i < len(techSkills)-1:
        outCSV.write(techSkills[i]+",")
    else:
        outCSV.write(techSkills[i])
outCSV.write("\n")

for desc in description:
    tokens = nltk.word_tokenize(desc.decode('utf-8', errors='ignore'))
    tagged = nltk.pos_tag(tokens)
    nouns = [word for word,pos in tagged if (pos == 'NN' or pos == 'NNP')]
    downcased = [x.lower() for x in nouns]
    joined.append(downcased)


# print joined
skillCount = []
stateCounter = 0
for desc in joined:
    for noun in desc:
        outCSV.write(str(states[stateCounter])+',')
        for skill in techSkills:
            # skillCount.append(noun.count(skill))
            outCSV.write(str(noun.count(skill))+",")
        outCSV.write("\n")
        if stateCounter < len(states)-1:
            stateCounter += 1
        else:
            break

print skillCount
#Todo: output csv in manner of skills "x, y, z" to replace jobdescription


outCSV.close()