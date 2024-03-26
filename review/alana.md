## task-oriented dialogues are simple

-   the reward is easily defined
-   same with domain-specific dialogues

## open-domain dialogues are more complex

-   goals are hard to define
-   consistant personality, deeper understanding, bigger context

# Architectures

-   modular (pipeline, more manual work)
-   end-to-end
-   ensemble (multiple models)

## Modular

-   speech recognition -> language understanding -> dialogue management -> language generation -> speech synthesis

## End-to-end

-   way less manual work
-   halucination, racist, unethical, etc.

## Ensemble

-   multiple "bots", each specialized on a different domain
-   better level of control
-   more work to combine and add new functionalities

# Evaluation

-   local and global coherence
-   entity-grid model
-   automatic evaluation during development, human evaluation at the end (costly, time consuming)
-   there are many possible responses to a given input, we should not penalize valid responses
-   engagment metrics
-   task-based metrics

## Some automatic metrics

-   F1 score
-   BLEU score
-   METEOR score
-   ROUGE score
-   Greedy Matching
-   Embedding average
-   Vector Extrema
-   BERTScore
-   ...

# Alana Bot Ensemble

-   rule-based bots
-   information-retreival bots
-   miscellaneous

## selection and post processing

-   bot priority list
-   contextual priority
-   ranking function
-   fallback strategies
