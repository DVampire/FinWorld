workdir = "workdir"
assets_name = None
source = None
data_type = None
level = None
tag = f"llm"
exp_path = f"{workdir}/{tag}"
log_path = "finworld.log"

processor = dict(
    type = "LLMProcessor",
    repo_id = "finreasoner",
    repo_type = "dataset",
    train_source = [
        {
            "repo_id": "Salesforce/FinEval",
            "name": "salesforce",
            "meta_type": "FINANCE",
            "lang_type": "EN",
            "type": "GENERAL",
        },
        {
            "repo_id": "FinGPT/fingpt-fineval",
            "name": "fineval",
            "meta_type": "FINANCE",
            "lang_type": "ZH",
            "type": "EXAM",
        },
        {
            "repo_id": "ChanceFocus/flare-finqa",
            "name": "flare-finqa",
            "meta_type": "FINANCE",
            "lang_type": "EN",
            "type": "QA",
        },
        {
            "repo_id": "ChanceFocus/flare-convfinqa",
            "name": "convfinqa",
            "meta_type": "FINANCE",
            "lang_type": "EN",
            "type": "QA",
        },
        {
            "repo_id": "zwt963/finance_exam",
            "name": "finance_exam",
            "meta_type": "FINANCE",
            "lang_type": "EN",
            "type": "EXAM",
        },
        {
            "repo_id": "zwt963/cflue",
            "name": "cflue",
            "meta_type": "FINANCE",
            "lang_type": "ZH",
            "type": "GENERAL",
        }
    ],
    test_source = [
        {
            "repo_id": "ChanceFocus/flare-finqa",
            "name": "flare-finqa",
            "meta_type": "FINANCE",
            "lang_type": "EN",
            "type": "QA",
        },
        {
            "repo_id": "ChanceFocus/flare-convfinqa",
            "name": "convfinqa",
            "meta_type": "FINANCE",
            "lang_type": "EN",
            "type": "QA",
        },
        {
            "repo_id": "FinGPT/fingpt-fineval",
            "name": "fineval",
            "meta_type": "FINANCE",
            "lang_type": "ZH",
            "type": "EXAM",
        },
        {
            "repo_id": "zwt963/cflue",
            "name": "cflue",
            "meta_type": "FINANCE",
            "lang_type": "ZH",
            "type": "GENERAL",
        }
    ],
    max_concurrent = 6,
)