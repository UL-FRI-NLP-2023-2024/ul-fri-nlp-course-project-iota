import json
import os
from collections import Counter

files = [f for f in os.listdir(".") if os.path.isfile(f) and f.endswith(".json") and "phi2" in f]

for file in files:
    with open(file) as f:
        data = json.load(f)

    character = file.split("_")[0]
    print(f"{character}:")
    for key, value in data.items():
        print(f"{key}:")
        counter = Counter(value["houses"])
        print(f"  House: {counter.most_common(1)[0][0]}, ({counter})")
        scores = sorted(value["scores"].items(), key=lambda x: x[1], reverse=True)
        print(f"  Scores: {scores}")
    input()
