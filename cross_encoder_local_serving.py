import time
import sagemaker
from sagemaker.local import LocalSession
from sagemaker.pytorch.model import PyTorchModel

query = "How many people live in Berlin?"
sentences_list = [
    "Berlin has a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.",
    "New York City is famous for the Metropolitan Museum of Art.",
    "Berlin is a beautiful city",
    "London has a population of around 8 million registered inhabitants",
    "people who live Berlin are very unfriendly",
]


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
        model_data="./model/cross-encoder/model.tar.gz",
        role=role,
        framework_version="1.8.1",
        source_dir="code_cross_encoder",
        py_version="py3",
        entry_point="inference.py",
    )

    predictor = pytorch_model.deploy(
        initial_instance_count=1, instance_type="local_gpu"
    )

    predictor.serializer = sagemaker.serializers.JSONSerializer()
    predictor.deserializer = sagemaker.deserializers.JSONDeserializer()

    return predictor

    # result = predictor.predict({"query": query, "sentences_list": sentences_list})
    # print(f"result: {result['best_sentence']}")

    # predictor.delete_endpoint(predictor.endpoint)


if __name__ == "__main__":

    predictor = main()

    inference_time = []
    for _ in range(10):
        start = time.time()
        result = predictor.predict({"query": query, "sentences_list": sentences_list})
        inference_time.append((time.time() - start))

    print(f"result: {result['best_sentence']}")
    print(f"\nExecution Time : {((time.time() -start)/len(inference_time)):.3f} ms")
