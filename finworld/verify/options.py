import re

def parse(answer: str) -> str:
    answer = str(answer)

    res_str = ""
    try:
        float(answer)
        res_str = answer
    except Exception as e:

        answer = answer.strip()

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

def verify(answer: str, method = "strict") -> bool:
    if method == "strict":
        pattern = r"^(?:([A-Z](?:,[A-Z])*)|((?:\d+\.\d+|\.\d+|\d+|[+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)(?:[A-Za-z]+)?))$"
        match = re.fullmatch(pattern, answer)
        if match:
            return True
        else:
            return False
    elif method == "flexible":
        raise NotImplementedError

if __name__ == '__main__':
    bala = "bala"

    cases = []

    option_case = [
        # single option
        "A",
        f"A. {bala}",
        f"A) {bala}",
        f"A.{bala}",
        f"A{bala}",
        f"A {bala}",

        # double options
        "A,B",
        f"A. {bala} B. {bala}",
        f"A) {bala} B) {bala}",
        f"A.{bala} B.{bala}",
        f"A{bala} B{bala}",
        f"A {bala} B {bala}",
        "A and B",
        f"A. {bala} and B. {bala}",
        f"A) {bala} and B) {bala}",
        f"A.{bala} and B.{bala}",
        f"A{bala} and B{bala}",
        f"A {bala} and B {bala}",

        # more than two options
        "A,B,C",
        f"A. {bala} B. {bala} C. {bala}",
        f"A) {bala} B) {bala} C) {bala}",
        f"A.{bala} B.{bala} C.{bala}",
        f"A{bala} B{bala} C{bala}",
        f"A {bala} B {bala} C {bala}",
        "A, B and C",
        f"A. {bala}, B. {bala} and C. {bala}",
        f"A) {bala}, B) {bala} and C) {bala}",
        f"A.{bala}, B.{bala} and C.{bala}",
        f"A{bala}, B{bala} and C{bala}",
        f"A {bala}, B {bala} and C {bala}",
    ]
    
    digital_cases = [
        # integer
        "120",
        "+120",
        "-120",
        
        # float
        "120.3",
        "+120.3",
        "-120.3",
        
        # scientific notation
        "120e3",
        "120e-3",
        "+120e3",
        "+120e-3",
        "-120e3",
        "-120e-3",
        "120.3e3",
        "120.3e-3",
        "+120.3e3",
        "+120.3e-3",
        "-120.3e3",
        "-120.3e-3",
        "120E3",
        "120E-3",
        "+120E3",
        "+120E-3",
        "-120E3",
        "-120E-3",
        "120.3E3",
        "120.3E-3",
        "+120.3E3",
        "+120.3E-3",
        "-120.3E3",
        "-120.3E-3",
        
        # digital with character [A-Za-z]
        "120F",
        "120.3F",
        "+120F",
        "+120.3F",
        "-120F",
        "-120.3F",
        "120e3F",
        "120e-3F",
        "+120e3F",
        "+120e-3F",
        "-120e3F",
        "-120e-3F",
        "120.3e3F",
        "120.3e-3F",
        "+120.3e3F",
        "+120.3e-3F",
        "-120.3e3F",
        "-120.3e-3F",
        "120E3F",
        "120E-3F",
        "+120E3F",
        "+120E-3F",
        "-120E3F",
        "-120E-3F",
        "120.3E3F",
        "120.3E-3F",
        "+120.3E3F",
        "+120.3E-3F",
        "-120.3E3F",
        "-120.3E-3F",
    ]

    gts = []

    cases += option_case
    cases += digital_cases
    mixed_cased = [f"The answer is {case}." for case in cases]
    cases += mixed_cased

    for case in cases:
        res = parse(case)
        gts.append(res)
        print(case, "==>", res, verify(res, method="strict"))

    print(cases)
    print(gts)