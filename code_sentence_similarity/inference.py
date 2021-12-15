import re
import sys
import logging
import json
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
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


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    # extracting the last_hidden_state tensor
    token_embeddings = model_output[
        0
    ]  # First element of model_output contains all token embeddings
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def model_fn(model_dir):
    device = get_device()
    model = AutoModel.from_pretrained(MODEL_PATH).eval().to(device)
    return model


def input_fn(serialized_input_data, content_type=JSON_CONTENT_TYPE):
    # print()
    # print("inside input_fn")
    if content_type == JSON_CONTENT_TYPE:
        input_data = json.loads(serialized_input_data)
        device = get_device()
        user_input_cleaned = text_cleaning(input_data["user_input"])
        true_sentence_cleaned = text_cleaning(input_data["true_sentence"])

        encoded_user_input = tokenizer.encode_plus(
            user_input_cleaned,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(device)

        encoded_true_sentence = tokenizer.encode_plus(
            true_sentence_cleaned,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(device)

        # we make a little bit of text cleaning

        return encoded_user_input, encoded_true_sentence

    else:
        raise Exception("Requested unsupported ContentType in Accept: " + content_type)
        return


def predict_fn(input_data, model):
    # print()
    # print("inside predict_fn")
    # print(f"Got input Data: {input_data}")

    encoded_user_input, encoded_true_sentence = input_data

    with torch.no_grad():
        user_embedding = model(**encoded_user_input)
        sentence_embedding = model(**encoded_true_sentence)

    pooled_user_embedding = mean_pooling(
        user_embedding, encoded_user_input["attention_mask"]
    )
    pooled_sentence_embedding = mean_pooling(
        sentence_embedding, encoded_true_sentence["attention_mask"]
    )

    user_mean_pooled = pooled_user_embedding.cpu().detach().numpy()
    sentence_mean_pooled = pooled_sentence_embedding.cpu().detach().numpy()

    similarity = cosine_similarity(user_mean_pooled, sentence_mean_pooled)

    return similarity


def output_fn(prediction_output, accept=JSON_CONTENT_TYPE):
    # print()
    # print("inside output_fn")
    # print(f"prediction output: {prediction_output}")
    # print(type(prediction_output))
    if accept == JSON_CONTENT_TYPE:
        prediction_output = {"similarity": str(prediction_output.item())}
        return json.dumps(prediction_output), accept

    raise Exception("Requested unsupported ContentType in Accept: " + accept)
