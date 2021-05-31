
import numpy as np
import json


class DataManager(object):
    def __init__(self, path, testfile):
        self.data = {}
        for name in ["train", "dev", "test"]:
            self.data[name] = []
            filename = testfile if name == "test" else name
            with open(path + (filename+".json")) as fl:
                for line in fl.readlines():
                    self.data[name].append(json.loads(line))

        # make words dict
        wordsdic = {}
        for name in ["train", "dev"]:
            datas = self.data[name]
            for item in datas:
                for word in item['sentext'].strip().split():
                    word = word.lower()
                    if word in wordsdic:
                        wordsdic[word] = wordsdic[word] + 1
                    else:
                        wordsdic[word] = 1
        wordssorted = sorted(wordsdic.items(), key=lambda d: (d[1], d[0]), reverse=True)
        self.words = {}
        for i in range(len(wordssorted)):
            self.words[wordssorted[i][0]] = i
        wordcount = len(wordssorted)

        # word to idx
        for name in ["train", "test", "dev"]:
            for item in self.data[name]:
                item['text'] = []
                for word in item['sentext'].strip().split():
                    if name == "test" and word.lower() not in self.words:
                        item['text'].append(wordcount)
                    else:
                        item['text'].append(self.words[word.lower()])

        # idx to vec
        self.vector = np.random.rand(len(self.words) + 1, 300) * 0.1
        with open(path + ("vector.txt")) as fl:
            for line in fl.readlines():
                vec = line.strip().split()
                word = vec[0].lower()
                vec = list(map(float, vec[1:]))
                if word in self.words:
                    self.vector[self.words[word]] = np.asarray(vec)
        self.vector = np.asarray(self.vector)

        # get event count
        self.eventcnt = {}
        self.events = []
        for name in ['train', 'dev']:
            for item in self.data[name]:
                for event in item['events']:
                    t = event['etext']
                    if t not in self.events:
                        self.events.append(t)
                        self.eventcnt[t] = 1
                    else:
                        self.eventcnt[t] += 1
        self.event_count = len(self.events)
        for name in ['train', 'test', 'dev']:
            for item in self.data[name]:
                for event in item['events']:
                    event['type'] = self.events.index(event['etext']) + 1
        print(self.events)
        print(self.eventcnt)

