# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import re
from mathruler.grader import extract_boxed_content, grade_answer

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


def verify(answer: str, method="strict") -> bool:
    if method == "strict":
        pattern = r"^(?:([A-Z](?:,[A-Z])*)|((?:\d+\.\d+|\.\d+|\d+|[+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)(?:[A-Za-z]+)?))$"
        match = re.fullmatch(pattern, answer)
        if match:
            return True
        else:
            return False
    elif method == "flexible":
        raise NotImplementedError

def extract_answer(solution: str) -> str:
    answer = extract_boxed_content(solution)
    answer = parse(answer)
    return answer

def format_reward(predict_str: str) -> float:
    pattern = re.compile(r'<think>.*</think>.*\\boxed\{.*\}.*', re.DOTALL)
    match_result = re.fullmatch(pattern, predict_str)
    return 1.0 if match_result else 0.0

def acc_reward(predict_str: str, ground_truth: str) -> float:
    return 1.0 if grade_answer(predict_str, ground_truth) else 0.0

def compute_score(predict_str: str, ground_truth: str) -> float:
    format_score = format_reward(predict_str)

    answer = extract_answer(predict_str)
    correct = acc_reward(answer, ground_truth)

    reward = 0.8 * correct + 0.2 * format_score

    return {
        "score": reward,
        "acc": correct,
        "pred": answer,
    }
