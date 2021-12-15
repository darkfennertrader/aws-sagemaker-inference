import re
import sys
import logging
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from test_cuda import test_cuda

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


JSON_CONTENT_TYPE = "application/json"

# checking that cuda is available inside container
test_cuda()

MODEL_PATH = "/opt/ml/model/"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)


def get_device():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return device


def text_cleaning(text_to_clean):
    text_to_clean = re.sub(r"[.?!]+", " ", text_to_clean)
    text_to_clean = " ".join(text_to_clean.split())
    return text_to_clean.lower().strip()


def model_fn(model_dir):
    device = get_device()
    model = (
        AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).eval().to(device)
    )
    return model


def input_fn(serialized_input_data, content_type=JSON_CONTENT_TYPE):
    # print()
    # print("inside input_fn")
    if content_type == JSON_CONTENT_TYPE:
        input_data = json.loads(serialized_input_data)
        device = get_device()
        # query_cleaned = text_cleaning(input_data["query"])
        # sentences_list_cleaned = text_cleaning(input_data["sentences_list"])
        query = input_data["query"]
        sentences_list = input_data["sentences_list"]
        # print("*********   within input_fn   ************")
        # print(query)
        # print(sentences_list)

        # format query input
        query_list = [query] * len(sentences_list)

        features_encoded = tokenizer(
            query_list,
            sentences_list,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(device)

        return input_data["sentences_list"], features_encoded

    else:
        raise Exception("Requested unsupported ContentType in Accept: " + content_type)
        return


def predict_fn(input_data, model):
    # print()
    # print("inside predict_fn")
    # print(f"Got input Data: {input_data}")

    sentences_list, features = input_data
    # print("*********   within predict_fn   ************")
    # print(sentences_list)
    # print(features)

    with torch.no_grad():
        scores = model(**features).logits

    best_sentence = sentences_list[torch.argmax(scores)]

    return best_sentence


def output_fn(prediction_output, accept=JSON_CONTENT_TYPE):
    # print()
    # print("within output_fn")
    # print(f"prediction output: {prediction_output}")
    # print(type(prediction_output))
    if accept == JSON_CONTENT_TYPE:
        prediction_output = {"best_sentence": str(prediction_output)}
        return json.dumps(prediction_output), accept

    raise Exception("Requested unsupported ContentType in Accept: " + accept)
