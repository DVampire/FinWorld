#!/usr/bin/env python3
"""
Quick‑and‑dirty latency test for a TGI instance running on localhost:8000.
"""

import json
import time
import requests   # pip install requests

ENDPOINT = "http://localhost:8000/generate"

# 1) Build the prompt and generation parameters ------------------------------
prompt = (
    "### System:\n"
    "You are a helpful assistant that provides concise summaries.\n\n"
    "### User:\n"
    "Please summarise the following text briefly:\n\n"
    "Tim Seymour spoke on [CNBC's Fast Money Final Trade](http:\/\/video.cnbc.com\/gallery\/?video=3000375583) about **Apple Inc.** (NASDAQ: [AAPL](\/stock\/aapl#NASDAQ)). He would buy it between $118 and $122.\nJon Najarian thinks that **Skyworks Solutions Inc** (NASDAQ: [SWKS](\/stock\/swks#NASDAQ)) could reach $100 and he wants to buy it.\nKaren Finerman is a buyer of **Golar LNG Limited (USA)** (NASDAQ: [GLNG](\/stock\/glng#NASDAQ)). She expects to see growth in the liquefied natural gas space.\nGuy Adami said that **Freeport-McMoRan Inc** (NYSE: [FCX](\/stock\/fcx#NYSE)) had a nice move higher and he thinks that it is going to continue to trade higher."
)

payload = {
    "inputs": prompt,
}

# 2) Send the request and time it --------------------------------------------
t0 = time.perf_counter()
response = requests.post(ENDPOINT, json=payload, timeout=120)
elapsed = time.perf_counter() - t0

# 3) Show results -------------------------------------------------------------
print(f"HTTP {response.status_code} · {elapsed:.2f}s")
print(json.dumps(response.json(), indent=2, ensure_ascii=False))
