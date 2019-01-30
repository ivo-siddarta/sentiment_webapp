from django.shortcuts import render
from django.http import HttpResponse
import pickle
import os
from sentimentwebapp import settings
import keras
import numpy as np
from keras import backend
from keras.preprocessing.sequence import pad_sequences
import re

model_name_dict = {'rnn': 'RNN-LSTM', 'logistic': 'Logistic Regression', 'ada': 'Ada Boost', 'gradient': 'Gradient Booster', 'xg': 'XG Boost'}

DIR = 'bliblireview/'
VECTORIZER_FILENAME = 'vectorizer.vec'
TOKENIZER_FILENAME = 'rnn.token'
MODELS_NAME = ['logistic', 'ada', 'gradient', 'xg', 'rnn']
TOKENIZER_PATH = os.path.join(settings.BASE_DIR, DIR + TOKENIZER_FILENAME)
VEC_PATH = os.path.join(settings.BASE_DIR, DIR + VECTORIZER_FILENAME)
MODELS_PATH = [os.path.join(settings.BASE_DIR, DIR + model + '.model') for model in MODELS_NAME]
RNN_PATH = MODELS_PATH[4]

def load_pickle(filename):
	return pickle.load(open(filename, 'rb'))

tokenizer = load_pickle(TOKENIZER_PATH)
vectorizer = load_pickle(VEC_PATH)
models = [load_pickle(model_path) if 'rnn' not in model_path else 'rnn' for model_path in MODELS_PATH]

def filter_message(messages):
    return list(map(lambda x: re.sub(r"(.)(\1{1})(\1*)", r"\1\1", x).lower(), messages))

def index(request):
	preds = {'RNN-LSTM':3, 'Logistic Regression':3 , 'Ada Boost':3, 'Gradient Booster':3, 'XG Boost':3}
	original_msg = ['']
	if request.GET.get('predict-btn'):
		backend.clear_session()
		original_msg = [request.GET.get('review')]
		if original_msg[0]:
			msg = filter_message(original_msg)
			for model, model_name in zip(models, MODELS_NAME):
				name = model_name_dict[model_name]
				if name == 'RNN-LSTM':
					model = load_pickle(RNN_PATH)
					preds[name] = predict_rnn(msg, model)
				else:
					preds[name] = predict_ml(msg, model)
	return render(request, 'bliblireview/index.html', {'MODELS': preds, 'prediction':original_msg[0]})

def predict_rnn(msg, model):
	transformed_msg = tokenizer.texts_to_sequences(msg)
	transformed_msg = pad_sequences(transformed_msg, maxlen=60, dtype='int32', value=0)
	pred = model.predict(transformed_msg, batch_size=1, verbose = 2)
	return np.argmax(pred)

def predict_ml(msg, model):
	transformed_msg = vectorizer.transform(msg)
	pred = model.predict(transformed_msg)
	return pred[0]