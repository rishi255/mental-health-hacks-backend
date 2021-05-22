# -*- coding: utf-8 -*-
import logging
import pickle
import sys
import uuid

import numpy as np
import pexpect  # dependency of ParlAI
from flask import Flask, request, session
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences


class ChatBot:
    def __init__(self):
        self.child = pexpect.spawn(
            "parlai interactive -t blended_skill_talk -mf zoo:blender/blender_90M/model",
            timeout=None,
        )
        # self.child.expect("Enter Your Message:")
        self.personality = self.child.before.decode("utf-8", "ignore").split(
            "[context]"
        )[1]

    def get_response(self):
        response = (self.child.before.split(b"1m"))[1].split(b"\x1b")
        return response[0].decode("utf-8")

    def send_request(self, req):
        self.child.sendline(req)
        # self.child.expect("Enter Your Message:")


subreddits = [
    "Depression",
    "CPTSD",
    "Anxiety",
    "BorderlinePDisorder",
    "OCD",
    "Agoraphobia",
    "schizophrenia",
    "domesticviolence",
]

keyid = 0
maxlen = 1000
alpha = 0.125
num_words = 100000
oov_token = "<UNK>"
pad_type = "post"
trunc_type = "post"


tokenizer = None
with open("tokenizerM.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)

model = keras.models.load_model("SDetector")
print("Model loaded!")


def predict(msg):
    test_sequences = tokenizer.texts_to_sequences([msg])
    test_padded = pad_sequences(
        test_sequences, padding=pad_type, truncating=trunc_type, maxlen=maxlen
    )
    ans = model.predict(test_padded)
    return ans[0]


app = Flask(__name__)
app.secret_key = "abc"
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


@app.route("/")
def hello():
    return "Hello!!!!!! World!!!!!!!!"


@app.route("/init")
def home():
    session_id = str(uuid.uuid4())  # Guarantees a unique session ID
    session[session_id] = {
        "msg_history": "",
        "id": session_id,
        "question": 0,
        "bot": ChatBot(),
    }
    return str(session_id)


@app.route("/bot")
def bot():
    sess_id = request.values.get("id", "").lower()
    incoming_msg = request.values.get("msg", "").lower()
    print(incoming_msg)
    # text = incoming_msg
    msg_history = session[sess_id]["msg_history"]

    if incoming_msg == "clear":
        msg_history = ""
        incoming_msg = "hi"
    if (len(msg_history) + len(incoming_msg)) <= maxlen:
        msg_history += ". " + incoming_msg
    else:
        msg_history = incoming_msg
    session[sess_id]["msg_history"] = msg_history
    if incoming_msg == "done":
        session.pop(session[sess_id][id])
        return "Thank you for using the app!"

    scores = predict(msg_history)
    most_pred = subreddits[np.argmax(scores)]
    return_msg = ""

    session[sess_id]["question"] += 1
    if session[sess_id]["question"] != 0 and session[sess_id]["question"] % 5 == 0:
        return_msg = f"You are showing symptoms of {most_pred} according to our analysis. If you are satisfied with it you can write 'done'. Else write 'okay' to continue."
        return return_msg

    else:
        session[sess_id]["bot"].send_request(incoming_msg)
        return_msg = session[sess_id]["bot"].get_response()
        return return_msg


if __name__ == "__main__":
    app.run(debug=True)
