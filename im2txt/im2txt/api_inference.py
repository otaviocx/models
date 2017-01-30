# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Generate captions for images using default beam search parameters."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import requests
import re

import tensorflow as tf

import json

from im2txt import configuration
from im2txt import inference_wrapper
from im2txt.inference_utils import caption_generator
from im2txt.inference_utils import vocabulary

from flask import Flask, request, redirect, url_for
#from flask_restful import Resource, Api
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename
import time

import sys


#FLAGS = tf.flags.FLAGS

#tf.flags.DEFINE_string("checkpoint_path", "",
#                       "Model checkpoint file or directory containing a "
#                       "model checkpoint file.")
#tf.flags.DEFINE_string("vocab_file", "", "Text file containing the vocabulary.")
#tf.flags.DEFINE_string("input_files", "",
#                       "File pattern or comma-separated list of file patterns "
#                       "of image files.")

# --checkpoint_path=/opt/DeepLearning/checkpoint2000000/model.ckpt-2000000 --vocab_file=/opt/DeepLearning/word_counts.txt

CHECKPOINT_PATH = "/opt/checkpoint2000000/model.ckpt-2000000"
VOCAB_FILE = "/opt/checkpoint2000000/word_counts.txt"

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "/opt/images"
CORS(app)

def translate(text, toLang):
  token = requests.post('https://api.cognitive.microsoft.com/sts/v1.0/issueToken', headers={'Ocp-Apim-Subscription-Key': '35e034f427f74c44a1c39445ceea31c4'})
  tradu = requests.get('https://api.microsofttranslator.com/v2/http.svc/Translate?appid=&text='+text+'&from=en-US&to='+toLang, headers={'Authorization': 'Bearer '+token.text})
  m = re.search('>(.*)<', tradu.text)
  #return tradu.text
  return m.group(1)

@app.route("/description", methods=["POST"])
def upload_file():
  old_stdout = sys.stdout
  sys.stdout = file('im2txt.log', 'w')
  upFile = request.files['file']
  filename = secure_filename(upFile.filename)
  filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
  upFile.save(filepath)
  result = process_image(filepath)
  sys.stdout.close()
  sys.stdout = old_stdout
  #jsonResult = json.dumps(result[bname]);
  textPt = translate(result[0]['description'], "pt-BR")
  docBody = '<script type="text/javascript">'
  docBody += 'var voz = new Audio("http://api.voicerss.org/?key=926a78262e6d44f2ae4cc278d7869813&hl=en-us&src='+result[0]['description']+'");'
  docBody += 'voz.play();'
  docBody += 'voz.onended = function() {'
  docBody +=   'var vozPt = new Audio("http://api.voicerss.org/?key=926a78262e6d44f2ae4cc278d7869813&hl=pt-br&src='+textPt+'");'
  docBody +=   'vozPt.play();'
  docBody += '};'
  docBody += "</script>"
  docBody += result[0]['description']+"<br/>"
  docBody += textPt+"<br/>"
  return docBody;

@app.route("/description.json", methods=["POST"])
def upload_file_json():
  timeInit = time.time()
  old_stdout = sys.stdout
  upFile = request.files['file']
  filename = str(time.time())+secure_filename(upFile.filename)
  filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
  sys.stdout = file(filepath+'.log', 'w')
  upFile.save(filepath)
  result = process_image(filepath)
  resp = result[0]
  lang = request.args['lang']
  text = resp['description']

  if lang != 'en' and lang != 'en-us':
    text = translate(text, lang)

  resp['description'] = text
  resp['audio'] = "http://api.voicerss.org/?key=926a78262e6d44f2ae4cc278d7869813&hl="+lang+"&src="+resp['description']
  resp['time'] = time.time() - timeInit
  jsonResult = json.dumps(resp)
  print(jsonResult)
  sys.stdout.close()
  sys.stdout = old_stdout
  return jsonResult;

def process_image(filepath):
  
  # Build the inference graph.
  g = tf.Graph()
  with g.as_default():
    model = inference_wrapper.InferenceWrapper()
    restore_fn = model.build_graph_from_config(configuration.ModelConfig(),
                                               CHECKPOINT_PATH)
  g.finalize()

  # Create the vocabulary.
  vocab = vocabulary.Vocabulary(VOCAB_FILE)

  filenames = [filepath]
  #for file_pattern in FLAGS.input_files.split(","):
  #  filenames.extend(tf.gfile.Glob(file_pattern))
  tf.logging.info("Running caption generation on %d files matching %s",
                  len(filenames), filepath)

  with tf.Session(graph=g, config=tf.ConfigProto(intra_op_parallelism_threads=4)) as sess:
    # Load the model from checkpoint.
    restore_fn(sess)

    # Prepare the caption generator. Here we are implicitly using the default
    # beam search parameters. See caption_generator.py for a description of the
    # available beam search parameters.
    generator = caption_generator.CaptionGenerator(model, vocab)
    
    result = {}
    for filename in filenames:
      with tf.gfile.GFile(filename, "r") as f:
        image = f.read()
      captions = generator.beam_search(sess, image)
      bname = os.path.basename(filename)
      print(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
      print("Captions for image %s:" % bname)
      result[bname] = []
      for i, caption in enumerate(captions):
        # Ignore begin and end words.
        sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
        sentence = " ".join(sentence)
	resJson = {"description": sentence, "logprob": math.exp(caption.logprob)}
        result[bname].append(resJson)
        print("  %d) %s (p=%f)" % (i, sentence, math.exp(caption.logprob)))

    return result[bname]


if __name__ == "__main__":
    app.run(host='0.0.0.0')
#if __name__ == "__main__":
#  tf.app.run()
