import os
import pickle

from typing import List, Optional
from helm.common.general import ensure_file_downloaded, ensure_directory_exists

from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Reference,
    TRAIN_SPLIT,
    VALID_SPLIT,
    TEST_SPLIT,
    CORRECT_TAG,
    Input,
    Output,
)


class LislaamErrorCat(Scenario):
    """Given an original text, summary and the location of errors in that summary."""

    name = "lislaam_error_classification"
    description = "Scenario for classifying errors in summaries of documents"
    tags = ["classification", "error-detection"]

    def __init__(self, dataset_name: str):
        """
        Initializes error classification scenario.

        Args:
            dataset_name: String identifier for dataset (e.g., "error-detection-dataset").
        """
        super().__init__()
        self.dataset_name = dataset_name

    def _download_dataset(tag: str, output_path: str):
        data_dir = os.path.join(output_path, "data")
        ensure_directory_exists(data_dir)
        #ensure_file_downloaded(source_url=url, target_path=os.path.join(data_dir, f"{tag}.pk"))

        with open(os.path.join(data_dir, f"{tag}.pk"), "rb") as fin:
            dataset = pickle.load(fin)

        return dataset

    def _load_dataset(self, dataset_name: str, output_path: str):
        if dataset_name == "rag_test_data":
            dataset = self._download_dataset("rag_test_data", output_path)
            document_key = "doc"
            summary_key = "summ"
            error_type = "error_type"
            error_locations = "annotated_span"
        else:
            raise ValueError(f"Dataset {dataset_name} is not supported.")

        return dataset, document_key, summary_key, error_type, error_locations

    def get_instances(self, output_path: str) -> List[Instance]:
        dataset, document_key, summary_key, error_type, error_locations = self._load_dataset(self.dataset_name, output_path)

        splits = {"train": TRAIN_SPLIT, "validation": VALID_SPLIT, "test": TEST_SPLIT}

        instances: List[Instance] = []

        for split_name, split in splits.items():
            for example in dataset[split_name]:
                document: str = example[document_key]
                summary: str = example[summary_key]
                error_type: str = example[error_type]  # "extrinsic" or "intrinsic"
                error_span: str = example[error_locations]

                # Create the prompt
                prompt = f"### ORIGINAL_TEXT: {document}\n ### SUMMARY: {summary}\n ### ERROR_LOCATIONS: {error_span}\n ### Output: "

                # Generate the instance
                instance = Instance(
                    input=Input(text=prompt),
                    references=[Reference(Output(text=error_type), tags=[CORRECT_TAG])],
                    split=split,
                )
                instances.append(instance)

        return instances