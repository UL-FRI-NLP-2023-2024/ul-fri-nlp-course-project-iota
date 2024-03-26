---
Review: Jan
---

# Chatbot is Not All You Need: Information-rich Prompting for More Realistic Responses

[Paper](/review/pdfs/Chatbot%20is%20Not%20All%20You%20Need.pdf)  
[Some code](https://github.com/srafsasm/InfoRichBot)

`interlocutor - a person who takes part in a dialogue or conversation.`

- We can enchance LLM responses by providing them with more information:
  - Five senses
  - Attributes
  - Emotinal state
  - Relationship with interlocutor
  - Memories
- Or putting it otherwise:
  - Information rich prompts - We update the prompt so they provide multi-aspect information about the character.
    - Name, personality trait, personal goal. Senses - Updated after each interaction. Memories - All previous character interactions.
  - Within-prompt self memory management - The models summarizes the history log and maintains it in the prompt.
    - Short term: Conversation log (prompts and responses) with a threshold. After threshold: Summarization -> this is appended to the long term memory.
    - Long term: Memory list
- Prompt engineering:
  - Template based prompts: Provide a structured format that can be easily replicated for similar tasks, ensuring consistency and efficiency in responses.
  - Few shot prompts: Help the model grasp the task's context or desired output style by providing explicit examples, improving the model's ability to generate relevant and accurate responses.
  - Chain of thought (CoT) prompts: Guide the AI to approach problems methodically, improving its problem-solving capabilities and the clarity of its explanations.
