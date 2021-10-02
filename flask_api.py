import numpy as np
import pandas as pd
from flask_api_utils import QuestionGrader, get_index
from threading import Thread
from flask import Flask, jsonify, request, json
import os

app = Flask(__name__)

# global grader object    
app.grader = QuestionGrader(
    knn_rec_size=100,
    rec_size=10,
)

def init_grader():
    global app
    app.grader.init_state(
        os.environ['EMB_PATH_KNRM'],
        os.environ['MLP_PATH'],
        os.environ['VOCAB_PATH'],
        os.environ['EMB_PATH_GLOVE'],
        get_index()
    )
    pass

# intializing grader
thread = Thread(target=init_grader)
thread.start()

@app.route('/ping')
def setup_model():
    if app.grader.state_intialized:
        return jsonify(status='ok')
    else:
        return jsonify(status='not loaded')

@app.route('/query', methods=['POST'])
def grade():
    if app.grader is None:
        return jsonify(status="Grader not intialized")

    if not app.grader.index_initialized:
        return jsonify(status="FAISS is not initialized!")

    inp = request.json

    if 'queries' not in inp:
        return jsonify(status="Wrong key")

    queries = inp.get('queries')
    lang_check, suggestions = app.grader.process_query(queries)

    return jsonify(lang_check=lang_check, suggestions=suggestions)