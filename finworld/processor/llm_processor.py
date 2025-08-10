import os
import json
import re
from typing import Any, Optional, List, Dict
import asyncio
from huggingface_hub import snapshot_download
import datasets as ds
from datasets import load_dataset
from dotenv import load_dotenv
load_dotenv(verbose=True)

from finworld.registry import PROCESSOR
from finworld.processor.custom import AbstractProcessor
from finworld.processor.salesforce import load_salesforce
from finworld.utils import assemble_project_path, get_tag_name, push_to_hub_folder
from finworld.config import config
from finworld.log import logger
from finworld.processor.score import compute_score

INSTRUCT_EN = (
    r'You FIRST think about the reasoning process as an internal monologue and then provide the final answer. '
    r'The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \boxed{}.'
)
INSTRUCT_ZH = (
    r'你首先将思考过程作为内部独白，然后给出最终答案。推理过程必须用<think> </think>标签括起来。最终答案必须放在\boxed{}中。'
)

def parse(answer: str) -> str:
    answer = str(answer)

    res_str = ""
    try:
        float(answer)
        res_str = answer
    except Exception as e:
        # match `A. balabala B. balabala`
        pattern = r'(?<!\w)([A-F])(?=\s|[.)\,]|$)(?:[.)\,]?\s*)(.*?)(?=[\s,]*[A-F](?:[.)\,]?\s*)|$)'
        matches = re.findall(pattern, answer, re.DOTALL)
        if matches:
            options = {key: value.strip() for key, value in matches}
            option_keys = list(sorted(list(options.keys())))
            res_str = ",".join(option_keys)
        else:
            # match `120`, `120.3`, `120e3`, `120F`
            pattern = r"([+\-]?\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?[A-Za-z]*)"
            matches = re.findall(pattern, answer)
            if matches:
                res_str = matches[0]
            else:
                res_str = answer
    return res_str

def make_map_fn(source, split, save_source):
    def process_fn(example, idx):

        question_raw = None
        solution_raw = None
        question = None
        answer = None
        images = None

        if source == "finance_exam":
            question_raw = example.pop('problem')
            question_raw = "\n".join(question_raw.split('\n\n'))
            question = question_raw + '\n' + INSTRUCT_EN
            solution_raw = example.pop('solution')
            answer = example.pop('answer')
        if source == "cflue":
            question_raw = example.pop('problem')
            question_raw = "\n".join(question_raw.split('\n\n'))
            question = question_raw + '\n' + INSTRUCT_ZH
            solution_raw = example.pop('answer')
            answer = solution_raw
        if source == "flare-finqa":
            question_raw = example.pop('query')
            question_raw = "\n".join(question_raw.split('\n\n'))
            pattern = re.compile(
                r"Context:\s*(.*?)\s*Question:\s*(.*?)\s*Answer:\s*(.*)",
                re.DOTALL
            )
            match = pattern.search(question_raw)
            if match:
                ex_context = match.group(1).strip()
                ex_question = match.group(2).strip()
                ex_answer = match.group(3).strip()
            else:
                raise ValueError("No match found in the question.")
            question = f"Context: {ex_context}\nQuestion: {ex_question}" + '\n' + INSTRUCT_EN
            solution_raw = example.pop('answer')
            answer = solution_raw

        if source == "convfinqa":
            question_raw = example.pop('query')
            question_raw = "\n".join(question_raw.split('\n\n'))
            pattern = re.compile(
                r"Context:\s*(.*?)\s*Question:\s*(.*?)\s*Answer:\s*(.*)",
                re.DOTALL
            )
            match = pattern.search(question_raw)
            if match:
                ex_context = match.group(1).strip()
                ex_question = match.group(2).strip()
                ex_answer = match.group(3).strip()
            else:
                raise ValueError("No match found in the question.")
            question = f"Context: {ex_context}\nQuestion: {ex_question}" + '\n' + INSTRUCT_EN
            solution_raw = example.pop('answer')
            answer = solution_raw

        if source == "fineval":
            question_raw = example.pop('input')
            question_raw = "\n".join(question_raw.split('\n\n'))
            question = question_raw + '\n' + INSTRUCT_ZH
            solution_raw = example.pop('output')
            answer = parse(solution_raw)

        if source == "salesforce":
            question_raw = example.pop('query')
            question_raw = "\n".join(question_raw.split('\n\n'))
            question = question_raw + '\n' + INSTRUCT_EN
            solution_raw = example.pop('answer')
            answer = solution_raw

        answer = answer.strip()

        # check the answer
        pred_answer = "<think>placeholder</think>\\boxed{" + answer + "}"
        score = compute_score(pred_answer, answer)
        if score < 1.0:
            logger.error(f"Score {score} is less than 1.0 for example {idx}. "
                         f"Question: {question}, "
                         f"Predicted Answer: {pred_answer}, "
                         f"Answer: {answer}, "
                         f"Save Source: {save_source}")

        data = {
            "data_source": save_source,
            "prompt": [{
                "role": "user",
                "content": question,
            }],
            "ability": "finance",
            "reward_model": {
                "style": "rule",
                "ground_truth": answer
            },
            "extra_info": {
                'answer': solution_raw,
                "question": question_raw,
            }
        }

        if images is not None:
            data["images"] = images

        return data

    return process_fn

@PROCESSOR.register_module(force=True)
class LLMProcessor(AbstractProcessor):
    def __init__(self,
                 repo_id: Optional[str] = None,
                 repo_type: Optional[str] = None,
                 train_source: List[Dict[str, Any]] = None,
                 test_source: List[Dict[str, Any]] = None,
                 max_concurrent: Optional[int] = 6,
                 **kwargs
                 ):
        super().__init__()

        self.repo_id = repo_id if repo_id is not None else None
        self.repo_name = self.repo_id.split('/')[-1] if self.repo_id else None
        self.repo_type = repo_type if repo_type is not None else "dataset"
        self.train_source = train_source if train_source is not None else []
        self.test_source = test_source if test_source is not None else []
        self.max_concurrent = max_concurrent

        self.exp_path = config.exp_path
        os.makedirs(self.exp_path, exist_ok=True)

    async def process_train_data(self, source: List[Dict[str, Any]], split: str):
        datasets = {}

        for item in source:
            repo_id = item.get('repo_id')
            name = item.get('name')
            meta_type = item.get('meta_type', 'FINANCE')
            lang_type = item.get('lang_type', 'EN')
            type_ = item.get('type', 'GENERAL')

            # download the dataset
            download_path = assemble_project_path(os.path.join(self.exp_path, name))
            logger.info(f"| Downloading dataset {name} from {repo_id}...")
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=download_path
            )
            logger.info(f"| Dataset {name} downloaded to {download_path}.")

            # Load the dataset
            if name == "salesforce":
                dataset = load_salesforce(source=name, data_path=download_path)
                dataset = dataset.map(function=make_map_fn(name, 'train', save_source=name), with_indices=True)
                datasets[name] = dataset
            elif name == "fineval":
                dataset = load_dataset(path=download_path)
                dataset = dataset['train']
                dataset = dataset.map(function=make_map_fn(name, 'train', save_source=name), with_indices=True)
                dataset = dataset.map(
                    lambda x: {"meta_type": meta_type, "type": type_, "lang_type": lang_type})
                datasets[name] = dataset
            elif name == "flare-finqa":
                dataset = load_dataset(path=download_path)
                dataset = ds.concatenate_datasets([dataset['train'], dataset['valid'], dataset['test']])
                dataset = dataset.map(function=make_map_fn(name, 'train', save_source=name), with_indices=True)
                dataset = dataset.map(
                    lambda x: {"meta_type": meta_type, "type": type_, "lang_type": lang_type})
                datasets[name] = dataset
            elif name == "convfinqa":
                dataset = load_dataset(path=download_path)
                dataset = ds.concatenate_datasets([dataset['train'], dataset['valid']])
                dataset = dataset.map(function=make_map_fn(name, 'train', save_source=name), with_indices=True)
                dataset = dataset.map(
                    lambda x: {"meta_type": meta_type, "type": type_, "lang_type": lang_type})
                datasets[name] = dataset
            elif name == "finance_exam":
                dataset = load_dataset('json', data_files=[
                    os.path.join(download_path, 'data.jsonl'),
                ])
                dataset = dataset['train']
                dataset = dataset.map(function=make_map_fn(name, 'train', save_source=name), with_indices=True)
                dataset = dataset.map(lambda x: {"lang_type": lang_type})
                datasets[name] = dataset
            elif name == "cflue":
                dataset = load_dataset(path=download_path)
                dataset = dataset['train']
                dataset = dataset.map(function=make_map_fn(name, 'train', save_source=name), with_indices=True)
                dataset = dataset.map(lambda x: {"lang_type": lang_type})
                datasets[name] = dataset



        # Concatenate all datasets
        if datasets:
            # filter columns
            columns = ["data_source", "meta_type", "type", "lang_type", "prompt", "ability", "reward_model", "extra_info"]

            data_nums = 0

            # save datasets
            output_path = os.path.join(self.exp_path, self.repo_name)
            os.makedirs(output_path, exist_ok=True)

            for name, dataset in datasets.items():

                logger.info(f"| Processing dataset {name} with length: {len(dataset)}...")
                dataset = dataset.remove_columns(
                    [col for col in dataset.column_names if col not in columns]
                )

                # reset id
                dataset = dataset.map(lambda x, idx: {"id": "{:06d}".format(idx)}, with_indices=True)

                logger.info(f"| Training dataset {name} processed with length: {len(dataset)}, saving to {output_path}...")
                dataset.to_parquet(os.path.join(output_path, f'{name}_train.parquet'))

                data_nums += len(dataset)

            logger.info(f"| Total training data processed: {data_nums} examples.")

    async def process_test_data(self, source: List[Dict[str, Any]], split: str):
        datasets = {}

        for item in source:
            repo_id = item.get('repo_id')
            name = item.get('name')
            meta_type = item.get('meta_type', 'FINANCE')
            lang_type = item.get('lang_type', 'EN')
            type_ = item.get('type', 'GENERAL')

            # download the dataset
            download_path = assemble_project_path(os.path.join(self.exp_path, name))
            logger.info(f"| Downloading dataset {name} from {repo_id}...")
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=download_path
            )
            logger.info(f"| Dataset {name} downloaded to {download_path}.")

            # Load the dataset
            if name == "flare-finqa":
                dataset = load_dataset(path=download_path)
                dataset = dataset['test']
                dataset = dataset.map(function=make_map_fn(name, 'test', save_source=name), with_indices=True)
                dataset = dataset.map(
                    lambda x: {"meta_type": meta_type, "type": type_, "lang_type": lang_type})
                datasets[name] = dataset
            elif name == "convfinqa":
                dataset = load_dataset(path=download_path)
                dataset = dataset['test']
                dataset = dataset.map(function=make_map_fn(name, 'test', save_source=name), with_indices=True)
                dataset = dataset.map(
                    lambda x: {"meta_type": meta_type, "type": type_, "lang_type": lang_type})
                datasets[name] = dataset
            elif name == "fineval":
                dataset = load_dataset(path=download_path)
                dataset = dataset['test']
                dataset = dataset.map(function=make_map_fn(name, 'test', save_source=name), with_indices=True)
                dataset = dataset.map(
                    lambda x: {"meta_type": meta_type, "type": type_, "lang_type": lang_type})
                datasets[name] = dataset
            elif name == "cflue":
                dataset = load_dataset(path=download_path)
                dataset = dataset['test']
                dataset = dataset.map(function=make_map_fn(name, 'test', save_source=name), with_indices=True)
                dataset = dataset.map(lambda x: {"lang_type": lang_type})
                datasets[name] = dataset

        # Concatenate all datasets
        if datasets:
            # filter columns
            columns = ["data_source", "meta_type", "type", "lang_type", "prompt", "ability", "reward_model",
                       "extra_info"]

            # save datasets
            output_path = os.path.join(self.exp_path, self.repo_name)
            os.makedirs(output_path, exist_ok=True)

            data_nums = 0

            for name, dataset in datasets.items():
                logger.info(f"| Processing dataset {name} with length: {len(dataset)}...")
                dataset = dataset.remove_columns(
                    [col for col in dataset.column_names if col not in columns]
                )

                # reset id
                dataset = dataset.map(lambda x, idx: {"id": "{:06d}".format(idx)}, with_indices=True)

                logger.info(f"| Testing dataset {name} processed with length: {len(dataset)}, saving to {output_path}...")
                dataset.to_parquet(os.path.join(output_path, f'{name}_test.parquet'))

                data_nums += len(dataset)

            logger.info(f"| Total testing data processed: {data_nums} examples.")

    async def run(self, *args, **kwargs):
        """
        Run the LLMProcessor to process the training and testing data.
        """
        logger.info("| Starting LLMProcessor...")

        # Process training data
        if self.train_source:
            await self.process_train_data(self.train_source, 'train')
        if self.test_source:
            await self.process_test_data(self.test_source, 'test')

        logger.info(f"| Pushing processed data to Hugging Face Hub: {self.repo_id}...")
        push_to_hub_folder(
            hf_token=os.getenv("HF_API_KEY"),
            endpoint=os.getenv("HF_ENDPOINT"),
            repo_id=self.repo_id,
            repo_type=self.repo_type,
            folder_path=os.path.join(self.exp_path, self.repo_name),
        )