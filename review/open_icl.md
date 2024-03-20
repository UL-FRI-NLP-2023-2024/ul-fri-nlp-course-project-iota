---
Review: Jan
---

# OpenICL: An Open-Source Framework for In-context Learning

[Paper](/review/pdfs/OpenICL.pdf)

- The paper introduces OpenICL, an open-source framework for in-context learning.
- In Context Learning (ICL): paradigm with LLMs where the model is trained for a particular task, without actually updating their weights. Instead, the model is shown some tasks and then asked to solve a related task.
- While ICL seems simple at first, doing it well requires some skill. That's why OpenICL was created. For example, we have 10000000 task definitions, the point of OpenICL is to optimally choose 100 of them to show to the model. Giving all of them to the LLM would overwhelm it, so OpenICL supports the following retrieval strategies:
  - Heuristic-based retrieval: BM25, Top-K, etc. (good for textually similar tasks)
  - Random sampling (good for very diverse tasks)
  - Model-based retrieval: use the model itself or a smaller model to retrieve tasks. There also other methods such as filtering by embeddings, doing RAG, Minimum Description Length (MDL) or Entropy-based selection.

**Examples of ICL:**

```s
# Sentiment Analysis
Input: "I had a wonderful time at the cafe. The coffee was great!"
Examples Provided to the Model:
Positive sentiment: "The movie was fantastic! I loved it."
Negative sentiment: "It was a terrible experience. I'm never going back."
Expected Output: Positive sentiment

# Arithmetic reasoning
Input: "If you have five apples and you eat two, how many do you have left?"
Examples Provided to the Model:
"If I have 3 candies and someone gives me 2 more, how many candies do I have?" "You have 5 candies."
"Ten birds were sitting on a tree. Two flew away. How many are left?" "There are 8 birds left."
Expected Output: "You have 3 apples left."
```

https://github.com/Shark-NLP/OpenICL at 496 stars and 28 forks. Last commit 9 months ago.
