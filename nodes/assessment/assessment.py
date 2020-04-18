import sklearn.metrics as m
import json
import sys
import numpy
import sys
if sys.version_info >= (3, 0):
        from inspect import getfullargspec
else:
        from inspect import getargspec

#read configurations
config = json.loads(input())
if sys.version_info >= (3, 0):
        while True:
                #read request
                data = json.loads(input())

                get_score = getattr(m, config['score'])
                kwargs = {}
                if 'average' in getfullargspec(get_score).args:
                        kwargs['average'] = 'micro'
                if 'beta' in getfullargspec(get_score).args:
                        kwargs['beta'] = 1
                score = get_score(data['real'], data['predicted'], **kwargs)
                if type(score) is numpy.ndarray:                        
			score = json.dumps(score.tolist())
                print(score)
else:
        while True:
                #read request
                data = json.loads(input())

                get_score = getattr(m, config['score'])
                kwargs = {}
                if 'average' in getargspec(get_score).args:
                        kwargs['average'] = 'micro'
                if 'beta' in getargspec(get_score).args:
                        kwargs['beta'] = 1
		score = get_score(data['real'], data['predicted'], **kwargs)
                if type(score) is numpy.ndarray:
                        score = json.dumps(score.tolist())
                print(score)
