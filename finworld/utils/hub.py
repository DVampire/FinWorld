#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

import os
from glob import glob
from dotenv import load_dotenv
load_dotenv(verbose=True)

from huggingface_hub import (
    upload_large_folder,
    HfApi
)

def generate_readme(repo_id: str, folder_path: str) -> str:
    lines = [
        "---",
        "title: README",
        "emoji: 📚",
        "colorFrom: blue",
        "colorTo: green",
        "sdk: null",
        "sdk_version: null",
        "app: null",
        "app_version: null",
        "license: cc-by-4.0",
        "tags: []",
        "thumbnail: null",
        "---",
        f"# Dataset: `{repo_id}`",
        "",
        "## Dataset Overview",
        "This dataset was automatically uploaded via script.",
        ""
    ]

    # Get all files and directories directly under folder_path
    entries = glob(os.path.join(folder_path, "*"))
    files = [f for f in entries if os.path.isfile(f) and not f.endswith(".log")]
    dirs = [d for d in entries if os.path.isdir(d)]

    total_files = len(files)
    file_types = {}
    for file in files:
        ext = os.path.splitext(file)[-1].lower()
        file_types[ext] = file_types.get(ext, 0) + 1

    lines.append(f"Total files: **{total_files}**")
    if file_types:
        lines.append("File types:")
        for ext, count in sorted(file_types.items()):
            lines.append(f"- `{ext or 'unknown'}`: {count}")

    lines += ["", "## Folder structure", "```"]

    for file in sorted(files):
        lines.append(f"- {os.path.basename(file)}")

    for d in sorted(dirs):
        sub_files = glob(os.path.join(d, "**", "*"), recursive=True)
        file_count = len([f for f in sub_files if os.path.isfile(f)])
        lines.append(f"- {os.path.basename(d)}/ ({file_count} files)")

    lines.append("```")
    lines.append("\n---\n*This README was generated automatically.*")
    return "\n".join(lines)

def push_to_hub_folder(
        hf_token: str,
        endpoint: str,
        repo_id: str,
        repo_type: str,
        folder_path: str,
        extra_ignore_patterns=None
    ):
    api = HfApi(token=hf_token, endpoint=endpoint)

    repo_url = api.create_repo(
        repo_id=repo_id,
        repo_type=repo_type,
        private=True,
        exist_ok=True,
        token=hf_token
    )

    if extra_ignore_patterns is None:
        extra_ignore_patterns = []
    ignore_patterns = ["*.log"]
    ignore_patterns.extend(extra_ignore_patterns)

    readme_content = generate_readme(repo_id, folder_path)
    readme_path = os.path.join(folder_path, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)

    os.environ["HF_TOKEN"] = hf_token

    upload_large_folder(
        repo_id=repo_id,
        folder_path=folder_path,
        revision="main",
        ignore_patterns=ignore_patterns,
        repo_type=repo_type,
        num_workers=4
    )