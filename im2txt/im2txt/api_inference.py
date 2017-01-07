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

#FLAGS = tf.flags.FLAGS

#tf.flags.DEFINE_string("checkpoint_path", "",
#                       "Model checkpoint file or directory containing a "
#                       "model checkpoint file.")
#tf.flags.DEFINE_string("vocab_file", "", "Text file containing the vocabulary.")
#tf.flags.DEFINE_string("input_files", "",
#                       "File pattern or comma-separated list of file patterns "
#                       "of image files.")

# --checkpoint_path=/opt/DeepLearning/checkpoint2000000/model.ckpt-2000000 --vocab_file=/opt/DeepLearning/word_counts.txt

CHECKPOINT_PATH = "/opt/DeepLearning/checkpoint2000000/model.ckpt-2000000"
VOCAB_FILE = "/opt/DeepLearning/word_counts.txt"

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "/opt/DeepLearning/images"
CORS(app)
@app.route("/description", methods=["POST"])
def upload_file():
  file = request.files['file']
  filename = secure_filename(file.filename)
  filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
  file.save(filepath)
  
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

  with tf.Session(graph=g) as sess:
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
      print("Captions for image %s:" % bname)
      result[bname] = []
      for i, caption in enumerate(captions):
        # Ignore begin and end words.
        sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
        sentence = " ".join(sentence)
	resJson = {"description": sentence, "logprob": math.exp(caption.logprob)}
        result[bname].append(resJson)
        print("  %d) %s (p=%f)" % (i, sentence, math.exp(caption.logprob)))
    return json.dumps(result)

if __name__ == "__main__":
    app.run()
#if __name__ == "__main__":
#  tf.app.run()
