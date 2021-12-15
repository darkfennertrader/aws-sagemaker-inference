import time
import sagemaker
from sagemaker.local import LocalSession
from sagemaker.pytorch.model import PyTorchModel


text_to_summarize = "When Paul Jobs was mustered out of the Coast Guard after World War II, he made a wager with his crewmates. They had arrived in San Francisco, where their ship was decommissioned, and Paul bet that he would find himself a wife within two weeks. He was a taut, tattooed engine mechanic, six feet tall, with a passing resemblance to James Dean. But it wasn’t his looks that got him a date with Clara Hagopian, a sweet-humored daughter of Armenian immigrants. It was the fact that he and his friends had a car, unlike the group she had originally planned to go out with that evening. Ten days later, in March 1946, Paul got engaged to Clara and won his wager. It would turn out to be a happy marriage, one that lasted until death parted them more than forty years later."


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
        model_data="./model/summarizer/model.tar.gz",
        role=role,
        framework_version="1.8.1",
        source_dir="code",
        py_version="py3",
        entry_point="inference.py",
    )

    predictor = pytorch_model.deploy(
        initial_instance_count=1, instance_type="local_gpu"
    )

    predictor.serializer = sagemaker.serializers.JSONSerializer()
    predictor.deserializer = sagemaker.deserializers.JSONDeserializer()

    result = predictor.predict(text_to_summarize)
    print(f"result: {result}")

    # predictor.delete_endpoint(predictor.endpoint)


if __name__ == "__main__":
    start = time.time()
    main()
    print(f"\nExecution Time : {(time.time() -start):.3f} ms")
