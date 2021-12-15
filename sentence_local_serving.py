import time
import sagemaker
from sagemaker.local import LocalSession
from sagemaker.pytorch.model import PyTorchModel

testing_phrases = [
    "I buy clothes through the internet",
    "the little girl took the recepit to the bus",
    "In the early winter I didn't miss the warm weather",
    "I definitely thought the water at the beach was salty",
    "What do you do for a living?",
    "I'm going to love this application",
    "Golden Group is an italian famous company",
    "the English test was a piece of cake",
    "I don't really like going to bars anymore",
    "So we're going to London, then Munich, then we will fly out of Athens, right?",
    "I've been trying to figure this out for ages",
    "Where do you live?",
    "Italy is the most beautiful country in the world",
    "What's it like living in Rome?",
    "Thank you so much I really appreciate your cooking dinner",
    "I'm really sorry for the mess. I was not expecting anyone today",
    "Are you going to the grocery store today?",
    "I'm sorry but I didn't understand anything of what you said",
    "I am looking for snow boots. Can you help me?",
    "Can I have this delivered next Tuesday?",
]

user_sentence = "I love going to the sea"
ground_truth = "Going to the sea is the most beautiful thing in the world"


def main():
    sagemaker_session = LocalSession()
    sagemaker_session.config = {"local": {"local_code": True}}

    # For local training a dummy role will be sufficient
    role = "arn:aws:iam::111111111111:role/service-role/AmazonSageMaker-ExecutionRole-20200101T000001"

    print("Deploying local mode endpoint")
    print(
        "Note: if launching for the first time in local mode, container image download might take a few minutes to complete."
    )

    pytorch_model = PyTorchModel(
        model_data="./model/sentence-similarity/model.tar.gz",
        role=role,
        framework_version="1.8.1",
        source_dir="code_sentence_similarity",
        py_version="py3",
        entry_point="inference.py",
    )

    predictor = pytorch_model.deploy(
        initial_instance_count=1, instance_type="local_gpu"
    )

    predictor.serializer = sagemaker.serializers.JSONSerializer()
    predictor.deserializer = sagemaker.deserializers.JSONDeserializer()

    result = predictor.predict({"user_input": user_sentence, "true_sentence": ground_truth})
    print(f"result: {result['similarity']}")

    # predictor.delete_endpoint(predictor.endpoint)


if __name__ == "__main__":
    start = time.time()
    main()
    print(f"\nExecution Time : {(time.time() -start):.3f} ms")
