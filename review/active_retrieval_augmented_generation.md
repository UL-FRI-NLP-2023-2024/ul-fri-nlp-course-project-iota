# Forward-looking active retrieval (FLARE) augmented generation

Checks for low probability tokens in a response and retrieves relevant information from a knowledge base to change the response. Applicable to any existing LLM, no training.

Retrieval augmented LMs are paired with a retriever. The retrieved knowledge is then combined with the user input and fed to the LM. $y = LM(x, D_q)$, where $x$ is the user input, $D_q$ is the retrieved knowledge, and $y$ is the response.

## Single time retrieval augmented generation

User input $x$ is also the query for the retirever $y = LM(x, D_x)$.

## Active retrieval augmented generation

Interleaves retrieval and generation. Retrieval query is generated from the user input and all generated outputs.

## FLARE

### Retrieval instructions

Generate a "[Search(query)]" when additional information is needed:
`The colors on the flag of Ghana have the following meanings. Red is for [Search(Ghana flag red meaning)] the blood of martyrs, ...`

### Direct

Since we can't fine-tune black box LLMs, the retrieval instructions might not be reliable.

1. Generate a temporary next sentance $\hat{s_t} = LM(x, y_{<t})$
2. If LM is confident, we accept it witout retrieval of additional information
3. If not, we formulate a search query using $\hat{s_t}$.

$$
y_t =
\begin{cases}
\hat{s_t} \qquad\qquad\qquad\text{If all tokens have probability higher than threshold} \\
LM(D_q,x,y_{<t}) \text{Otherwise}
\end{cases}
$$

### Confidence-based

Mask out low probability tokens and retrieve information for them.
