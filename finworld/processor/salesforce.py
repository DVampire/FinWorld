import re
import os
import datasets as ds
from datasets import load_dataset
from dotenv import load_dotenv
load_dotenv(verbose=True)

def load_salesforce(source, data_path):
    datasets = []

    dataset = load_dataset(path=data_path, name = 'CFA-Challenge')
    dataset = dataset['test']
    def process_fn1(example, idx):
        meta_type = 'CFA'
        lang_type = 'EN'

        query = example['query']
        answer = example['answer']
        source = example['source']

        if 'Mock PM' in source:
            type = 'PM'
        else:
            type = 'EXAM'

        pattern = re.compile(
            r"Scenario:\s*(.*?)\s*Question:\s*(.*?)\s*Answer Choices:\s*(.*)\s*Answer:\s*(.*)",
            re.DOTALL
        )
        match = pattern.search(query)
        if match:
            ex_scenario = match.group(1).strip()
            ex_question = match.group(2).strip()
            ex_answer_choices = match.group(3).strip()
            ex_answer = match.group(4).strip()

            if 'TOPIC: ETHICAL AND PROFESSIONAL STANDARDS' in ex_scenario:
                ex_scenario = ex_scenario.replace('TOPIC: ETHICAL AND PROFESSIONAL STANDARDS', '')
            if 'TOTAL POINT VALUE OF THIS QUESTION SET IS 12 POINTS' in ex_scenario:
                ex_scenario = ex_scenario.replace('TOTAL POINT VALUE OF THIS QUESTION SET IS 12 POINTS', '')
            if ex_scenario[-1] == ';':
                ex_scenario = ex_scenario[:-1]
            ex_scenario = ex_scenario.strip()

            if ex_question[-1] == ';':
                ex_question = ex_question[:-1]

            if ex_answer_choices[-1] == ';':
                ex_answer_choices = ex_answer_choices[:-1]
            new_answer_choices = []
            pattern = r'([A-Z]):\s*(.*?)\s*(?=[A-Z]:|$)'
            matches = re.findall(pattern, ex_answer_choices, re.DOTALL)
            for match in matches:
                key = match[0].strip()
                value = match[1]
                if value.endswith('.,') or value.endswith('..'):
                    value = value[:-1]
                value = value.strip()
                new_answer_choices.append(f"{key}. {value}")
            ex_answer_choices = '\n'.join(new_answer_choices)

            question = (f"Context:\n{ex_scenario}\n"
                        f"Question:\n{ex_question}\n"
                        f"Choices:\n{ex_answer_choices}")

            example['query'] = question
            example['answer'] = answer
            example['meta_type'] = meta_type
            example['type'] = type
            example['lang_type'] = lang_type

            return example

        else:
            raise ValueError("No match found in the question.")

    dataset = dataset.map(function=process_fn1, with_indices=True)
    datasets.append(dataset)

    dataset = load_dataset(path=data_path, name = 'CFA-Easy')
    dataset = dataset['test']
    def process_fn2(example, idx):
        meta_type = 'CFA'
        lang_type = 'EN'
        type = 'EASY'

        query = example['query']
        answer = example['answer']

        pattern = re.compile(
            r"Q:\s*(.*?)\s*CHOICES:\s*(.*?)\s*Answer:\s*(.*)",
            re.DOTALL
        )
        match = pattern.search(query)

        if match:
            ex_question = match.group(1).strip()
            ex_answer_choices = match.group(2).strip()
            ex_answer = match.group(3).strip()

            if ex_question[-1] == ';':
                ex_question = ex_question[:-1]
            if ex_answer_choices[-1] == ';':
                ex_answer_choices = ex_answer_choices[:-1]
            new_answer_choices = []
            pattern = r'([A-Z]):\s*(.*?)\s*(?=[A-Z]:|$)'
            matches = re.findall(pattern, ex_answer_choices, re.DOTALL)
            for match in matches:
                key = match[0].strip()
                value = match[1]
                if value.endswith('.,') or value.endswith('..'):
                    value = value[:-1]
                value = value.strip()
                new_answer_choices.append(f"{key}. {value}")
            ex_answer_choices = '\n'.join(new_answer_choices)

            # question = f"{ex_question}\n{ex_answer_choices}"
            question = (f"Question:\n{ex_question}\n"
                        f"Choices:\n{ex_answer_choices}")

            example['query'] = question
            example['answer'] = answer
            example['meta_type'] = meta_type
            example['type'] = type
            example['lang_type'] = lang_type
            return example
        else:
            raise ValueError("No match found in the question.")
    dataset = dataset.map(function=process_fn2, with_indices=True)
    datasets.append(dataset)

    dataset = load_dataset(path=data_path, name = 'CRA-Bigdata')
    dataset = dataset['test']
    def process_fn3(example, idx):
        meta_type = 'CRA'
        lang_type = 'EN'
        type = 'TREND'

        query = example['query']
        answer = example['answer']

        if 'Answer:' in query:
            query = query.replace('Answer:', '')
        question = query.strip()

        match = re.search(r"(.*?)\s*Context:\s*(.*)$", question, re.DOTALL)
        ex_qusetion = match.group(1).strip()
        ex_context = match.group(2).strip()

        ex_qusetion = ex_qusetion + ' should be classified as:'

        maps = {
            'Rise': 'A',
            'Fall': 'B',
        }

        sorted_maps = sorted(maps.items(), key=lambda x: x[1])
        ex_answer_choices = "\n".join([f"{value}. {key}" for key, value in sorted_maps])

        # question = f"{context}\n{qusetion}"
        question = (f"Context:\n{ex_context}\n"
                    f"Question:\n{ex_qusetion}\n"
                    f"Choices:\n{ex_answer_choices}")

        answer = maps.get(answer)

        example['query'] = question
        example['answer'] = answer
        example['meta_type'] = meta_type
        example['type'] = type
        example['lang_type'] = lang_type

        return example

    dataset = dataset.map(function=process_fn3, with_indices=True)
    datasets.append(dataset)

    dataset = load_dataset(path=data_path, name = 'CRA-CCF')
    dataset = dataset['test']
    def process_fn4(example, idx):
        meta_type = 'CRA'
        lang_type = 'EN'
        type = 'CCF'

        query = example['query']
        answer = example['answer']

        match = re.search(r"(.*?)\s*For instance[,:]\s*(.*?)\s*Text:\s*(.*)\s*Answer:\s*(.*)$", query, re.DOTALL)
        if match:
            ex_context = match.group(1).strip()
            ex_example = match.group(2).strip()
            ex_question = match.group(3).strip()
            ex_answer = match.group(4).strip()

            maps = {
                'yes': 'A',
                'no': 'B',
            }
            sorted_maps = sorted(maps.items(), key=lambda x: x[1])
            ex_answer_choices = "\n".join([f"{value}. {key}" for key, value in sorted_maps])

            ex_question = ex_question + ' should be classified as:'

            question = (f"Context:\n{ex_context}\n"
                        f"Example:\n{ex_example}\n"
                        f"Question:\n{ex_question}\n"
                        f"Choices:\n{ex_answer_choices}")

            answer = maps.get(answer)

            example['query'] = question
            example['answer'] = answer
            example['meta_type'] = meta_type
            example['type'] = type
            example['lang_type'] = lang_type
            return example
        else:
            raise ValueError("No match found in the question.")

    dataset = dataset.map(function=process_fn4, with_indices=True)
    dataset = dataset.map(function=lambda x: {"type": "CCF"})
    datasets.append(dataset)

    dataset = load_dataset(path=data_path, name = 'CRA-CCFraud')
    dataset = dataset['test']
    dataset = dataset.map(function=process_fn4, with_indices=True)
    dataset = dataset.map(function=lambda x: {"type": "CCFRAUD"})
    datasets.append(dataset)

    dataset = load_dataset(path=data_path, name = 'CRA-LendingClub')
    dataset = dataset['test']
    dataset = dataset.map(function=process_fn4, with_indices=True)
    dataset = dataset.map(function=lambda x: {"type": "LENDINGCLUB"})
    datasets.append(dataset)

    dataset = load_dataset(path=data_path, name = 'CRA-Polish')
    dataset = dataset['test']
    dataset = dataset.map(function=process_fn4, with_indices=True)
    dataset = dataset.map(function=lambda x: {"type": "POLISH"})
    datasets.append(dataset)

    dataset = load_dataset(path=data_path, name = 'CRA-ProtoSeguro')
    dataset = dataset['test']
    dataset = dataset.map(function=process_fn4, with_indices=True)
    dataset = dataset.map(function=lambda x: {"type": "PROTOSEGURO"})
    datasets.append(dataset)

    dataset = load_dataset(path=data_path, name = 'CRA-Taiwan')
    dataset = dataset['test']
    dataset = dataset.map(function=process_fn4, with_indices=True)
    dataset = dataset.map(function=lambda x: {"type": "TAIWAN"})
    datasets.append(dataset)

    dataset = load_dataset(path=data_path, name = 'CRA-TravelInsurance')
    dataset = dataset['test']
    dataset = dataset.map(function=process_fn4, with_indices=True)
    dataset = dataset.map(function=lambda x: {"type": "TRAVELINSURANCE"})
    datasets.append(dataset)

    dataset = load_dataset(path=data_path, name = 'FinanceBench')
    dataset = dataset['test']
    dataset = dataset.map(function=process_fn4, with_indices=True)
    dataset = dataset.map(function=lambda x: {"type": "FINANCEBENCH"})
    datasets.append(dataset)

    dataset = load_dataset(path=data_path, name = 'FIQASA')
    dataset = dataset['test']

    def process_fn5(example, idx):
        meta_type = 'FIQASA'
        lang_type = 'EN'
        type = 'EXAM'

        query = example['query']
        answer = example['answer']

        pattern = re.compile(
            r"(.*?)\s*Text:\s*(.*?)\s*Answer:\s*(.*)",
            re.DOTALL
        )
        match = pattern.search(query)
        if match:
            ex_context = match.group(1).strip()
            ex_question = match.group(2).strip()
            ex_answer = match.group(3).strip()

            ex_context = ex_context.replace('Positive', 'positive')
            ex_context = ex_context.replace('Negative', 'negative')
            ex_context = ex_context.replace('Neutral', 'neutral')

            map = {
                'positive': 'A',
                'negative': 'B',
                'neutral': 'C',
            }
            sorted_map = sorted(map.items(), key=lambda x: x[1])
            ex_answer_choices = "\n".join([f"{value}. {key}" for key, value in sorted_map])

            answer = map.get(answer)

            question = (f"Context:\n{ex_question}\n"
                        f"Question:\n{ex_context}\n"
                        f"Choices:\n{ex_answer_choices}")

            example['query'] = question
            example['answer'] = answer
            example['meta_type'] = meta_type
            example['type'] = type
            example['lang_type'] = lang_type

            return example
        else:
            raise ValueError("No match found in the question.")

    dataset = dataset.map(function=process_fn5, with_indices=True)
    dataset = dataset.map(function=lambda x: {"meta_type": "TR", "type": "SENTIMENT"})
    datasets.append(dataset)

    dataset = load_dataset(path=data_path, name = 'Flare-Australian')
    dataset = dataset['test']

    def process_fn6(example, idx):
        meta_type = 'CRA'
        lang_type = 'EN'
        type = 'CCF'

        query = example['query']
        answer = example['answer']

        match = re.search(r"(.*?)\s*For instance[,:]\s*(.*?)\s*Text:\s*(.*)\s*$", query, re.DOTALL)
        if match:
            ex_context = match.group(1).strip()
            ex_example = match.group(2).strip()
            ex_question = match.group(3).strip()

            maps = {
                'good': 'A',
                'bad': 'B',
            }
            sorted_maps = sorted(maps.items(), key=lambda x: x[1])
            ex_answer_choices = "\n".join([f"{value}. {key}" for key, value in sorted_maps])

            ex_question = ex_question + ' should be classified as:'

            question = (f"Context:\n{ex_context}\n"
                        f"Example:\n{ex_example}\n"
                        f"Question:\n{ex_question}\n"
                        f"Choices:\n{ex_answer_choices}")

            answer = maps.get(answer)

            example['query'] = question
            example['answer'] = answer
            example['meta_type'] = meta_type
            example['type'] = type
            example['lang_type'] = lang_type
            return example
        else:
            raise ValueError("No match found in the question.")
    dataset = dataset.map(function=process_fn6, with_indices=True)
    dataset = dataset.map(function=lambda x: {"meta_type":"TR", "type": "STATUS"})
    datasets.append(dataset)

    dataset = load_dataset(path=data_path, name = 'Flare-German')
    dataset = dataset['test']
    dataset = dataset.map(function=process_fn6, with_indices=True)
    dataset = dataset.map(function=lambda x: {"meta_type":"TR", "type": "STATUS"})
    datasets.append(dataset)

    dataset = load_dataset(path=data_path, name = 'FOMC')
    dataset = dataset['test']
    def process_fn7(example, idx):
        meta_type = 'FOMC'
        lang_type = 'EN'
        type = 'EXAM'

        query = example['query']
        answer = example['answer']

        pattern = re.compile(
            r"(.*?)\s*Text:\s*(.*?)\s*Answer:\s*(.*)",
            re.DOTALL
        )
        match = pattern.search(query)
        if match:
            ex_context = match.group(1).strip()
            ex_question = match.group(2).strip()
            ex_answer = match.group(3).strip()

            map = {
                'HAWKISH': 'A',
                'DOVISH': 'B',
                'NEUTRAL': 'C',
            }
            sorted_map = sorted(map.items(), key=lambda x: x[1])
            ex_answer_choices = "\n".join([f"{value}. {key}" for key, value in sorted_map])

            answer = answer.upper()

            answer = map.get(answer)

            question = (f"Context:\n{ex_question}\n"
                        f"Question:\n{ex_context}\n"
                        f"Choices:\n{ex_answer_choices}")

            example['query'] = question
            example['answer'] = answer
            example['meta_type'] = meta_type
            example['type'] = type
            example['lang_type'] = lang_type

            return example
        else:
            raise ValueError("No match found in the question.")


    dataset = dataset.map(function=lambda x: {"meta_type":"POLICY", "type": "STATUS"})
    datasets.append(dataset)

    dataset = load_dataset(path=data_path, name = 'FPB')
    dataset = dataset['test']
    dataset = dataset.map(function=process_fn5, with_indices=True)
    dataset = dataset.map(function=lambda x: {"meta_type":"TR", "type": "SENTIMENT"})
    datasets.append(dataset)

    dataset = load_dataset(path=data_path, name = 'MA')
    dataset = dataset['test']
    def process_fn8(example, idx):
        meta_type = 'MA'
        lang_type = 'EN'
        type = 'EXAM'

        query = example['query']
        answer = example['answer']

        pattern = re.compile(
            r"(.*?)\s*Text:\s*(.*?)\s*Answer:\s*(.*)",
            re.DOTALL
        )
        match = pattern.search(query)
        if match:
            ex_context = match.group(1).strip()
            ex_question = match.group(2).strip()
            ex_answer = match.group(3).strip()

            map = {
                'rumour': 'A',
                'complete': 'B',
            }
            sorted_map = sorted(map.items(), key=lambda x: x[1])
            ex_answer_choices = "\n".join([f"{value}. {key}" for key, value in sorted_map])

            question = (f"Context:\n{ex_question}\n"
                        f"Question:\n{ex_context}\n"
                        f"Choices:\n{ex_answer_choices}")

            answer = map.get(answer)

            example['query'] = question
            example['answer'] = answer
            example['meta_type'] = meta_type
            example['type'] = type
            example['lang_type'] = lang_type

            return example
        else:
            raise ValueError("No match found in the question.")
    dataset = dataset.map(function=process_fn8, with_indices=True)
    dataset = dataset.map(function=lambda x: {"meta_type":"TWEET", "type": "STATUS"})
    datasets.append(dataset)

    dataset = load_dataset(path=data_path, name = 'MLESG')
    dataset = dataset['test']
    def process_fn9(example, idx):
        meta_type = 'MLESG'
        lang_type = 'EN'
        type = 'EXAM'

        query = example['query']
        answer = example['answer']

        pattern = re.compile(
            r"(.*?)\s*Text:\s*(.*?)\s*Answer:\s*(.*)",
            re.DOTALL
        )
        match = pattern.search(query)
        if match:
            ex_context = match.group(1).strip()
            ex_question = match.group(2).strip()
            ex_answer = match.group(3).strip()

            map = {
                'Access to Communications': '01',
                'Biodiversity & Land Use': '02',
                'Packaging Material & Waste': '03',
                'Financing Environmental Impact': '04',
                'Carbon Emissions': '05',
                'Human Capital Development': '06',
                'Ownership & Control': '07',
                'Community Relations': '08',
                'Responsible Investment': '09',
                'Opportunities in Renewable Energy': '10',
                'Consumer Financial Protection': '11',
                'Accounting': '12',
                'Business Ethics': '13',
                'Opportunities in Clean Tech': '14',
                'Toxic Emissions & Waste': '15',
                'Product Carbon Footprint': '16',
                'Opportunities in Green Building': '17',
                'Climate Change Vulnerability': '18',
                'Pay': '19',
                'Water Stress': '20',
                'Supply Chain Labor Standards': '21',
                'Chemical Safety': '22',
                'Board': '23',
                'Opportunities in Nutrition & Health': '24',
                'Access to Health Care': '25',
                'Electronic Waste': '26',
                'Access to Finance': '27',
                'Raw Material Sourcing': '28',
                'Health & Demographic Risk': '29',
                'Labor Management': '30',
                'Controversial Sourcing': '31',
                'Privacy & Data Security': '32',
                'Product Safety & Quality': '33',
            }
            sorted_map = sorted(map.items(), key=lambda x: x[1])
            ex_answer_choices = "\n".join([f"{value}. {key}" for key, value in sorted_map])

            question = (f"Context:\n{ex_question}\n"
                        f"Question:\n{ex_context}\n"
                        f"Choices:\n{ex_answer_choices}")

            answer = map.get(answer)

            example['query'] = question
            example['answer'] = answer
            example['meta_type'] = meta_type
            example['type'] = type
            example['lang_type'] = lang_type

            return example
        else:
            raise ValueError("No match found in the question.")
    dataset = dataset.map(function=process_fn9, with_indices=True)
    dataset = dataset.map(function=lambda x: {"meta_type":"ESG", "type": "STATUS"})
    datasets.append(dataset)

    dataset = load_dataset(path=data_path, name = 'MMLU-finance')
    dataset = dataset['test']
    def process_fn10(example, idx):
        meta_type = 'MMLU'
        lang_type = 'EN'
        type = 'EXAM'

        query = example['query']
        answer = example['answer']

        pattern = re.compile(
            r"(.*?)\s*Question:\s*(.*?)\s*Answer Choices:\s*(.*)\s*Answer:\s*(.*)",
            re.DOTALL
        )
        match = pattern.search(query)
        if match:
            ex_context = match.group(1).strip()
            ex_question = match.group(2).strip()
            ex_answer_choices = match.group(3).strip()
            ex_answer = match.group(4).strip()

            if ex_context[-1] == ';':
                ex_context = ex_context[:-1]
            if ex_question[-1] == ';':
                ex_question = ex_question[:-1]

            if ex_answer_choices[-1] == ';':
                ex_answer_choices = ex_answer_choices[:-1]

            new_answer_choices = []
            pattern = r'([A-Z]):\s*(.*?)\s*(?=[A-Z]:|$)'
            matches = re.findall(pattern, ex_answer_choices, re.DOTALL)

            for match in matches:
                key = match[0].strip()
                value = match[1]
                if value.endswith('.,') or value.endswith('..'):
                    value = value[:-1]
                value = value.strip()
                new_answer_choices.append(f"{key}. {value}")
            ex_answer_choices = '\n'.join(new_answer_choices)

            question = (f"Question:\n{ex_question}\n"
                        f"Choices:\n{ex_answer_choices}")

            example['query'] = question
            example['answer'] = answer
            example['meta_type'] = meta_type
            example['type'] = type
            example['lang_type'] = lang_type

            return example
        else:
            raise ValueError("No match found in the question.")
    dataset = dataset.rename_column('answer', 'answer_num')
    dataset = dataset.rename_column('answer_text', 'answer')
    dataset = dataset.rename_column('question_text', 'query')
    dataset = dataset.map(function=process_fn10, with_indices=True)
    dataset = dataset.map(function=lambda x: {"meta_type":"MARKET", "type": "STATUS"})
    datasets.append(dataset)

    dataset = load_dataset(path=data_path, name = 'SM-ACL')
    dataset = dataset['test']
    dataset = dataset.map(function=process_fn3, with_indices=True)
    dataset = dataset.map(function=lambda x: {"meta_type":"ACLSM", "type": "STATUS"})
    datasets.append(dataset)

    dataset = load_dataset(path=data_path, name = 'SM-CIKM')
    dataset = dataset['test']
    dataset = dataset.map(function=process_fn3, with_indices=True)
    dataset = dataset.map(function=lambda x: {"meta_type":"CIKMSM", "type": "STATUS"})
    datasets.append(dataset)

    columns = ["meta_type", "type", "lang_type", "query", "answer"]
    for index, dataset in enumerate(datasets):
        dataset = dataset.remove_columns([col for col in dataset.column_names if col not in columns])
        datasets[index] = dataset

    datasets = ds.concatenate_datasets(datasets)

    # remove answer is none
    datasets = datasets.filter(lambda x: x['answer'] is not None and x['answer'] != '')
    datasets = datasets.filter(lambda x: x['query'] is not None and x['query'] != '')
    return datasets