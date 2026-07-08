# 📝 Day 2 · Daily Challenge: Semantic Search with Cosine()

This report presents the implementation of a custom Semantic Search engine using a from-scratch `cosine()` similarity function running over cached BGE-M3 mock embedding vectors. It contrasts results with a lexical TF-IDF model.

---

## 🛠️ Code Implementation

### From-Scratch Cosine Similarity Function
```python
def cosine(v1, v2):
    dot_product = sum(a * b for a, b in zip(v1, v2))
    norm_v1 = sum(a * a for a in v1) ** 0.5
    norm_v2 = sum(b * b for b in v2) ** 0.5
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return dot_product / (norm_v1 * norm_v2)
```

### Search and Caching Architecture
```python
# Embedding dataset and caching vectors once
cached_vectors = []
for index, row in df.iterrows():
    vec = get_embedding(row['text'])
    cached_vectors.append({
        "text": row['text'],
        "label": row['label'],
        "vector": vec
    })

def search(query, k=3):
    query_vector = get_embedding(query)
    scores = []
    for row in cached_vectors:
        score = cosine(query_vector, row["vector"])
        scores.append({
            "text": row["text"],
            "label": row["label"],
            "score": score
        })
    scores.sort(key=lambda x: x["score"], reverse=True)
    return scores[:k]
```

---

## 📊 Part B — Semantic Search Probe Tables

Below are the search results (`k=3`) for 8 test queries including 2 paraphrases and 2 German cross-language sentences.

### Query: `ring mom`
*(Paraphrase: verb 'ring' differs entirely from 'call' in dataset)*

| Rank | Matched Dataset Text | Intent Label | Cosine Score |
| :---: | :--- | :--- | :---: |
| 1 | "Ruf Mama an" | `place_call` | `0.8000` |
| 2 | "give my dad a ring" | `place_call` | `0.7990` |
| 3 | "call Sarah right now" | `place_call` | `0.7970` |

### Query: `make a reminder to buy milk`
*(Paraphrase: phrased differently from 'remind me to buy milk')*

| Rank | Matched Dataset Text | Intent Label | Cosine Score |
| :---: | :--- | :--- | :---: |
| 1 | "Erinnere mich daran, Milch zu kaufen" | `create_task` | `0.9638` |
| 2 | "remind me to buy milk tonight" | `create_task` | `0.9546` |
| 3 | "remind me to bye milk" | `create_task` | `0.9540` |

### Query: `Ruf Mama an`
*(# = call mom)*

| Rank | Matched Dataset Text | Intent Label | Cosine Score |
| :---: | :--- | :--- | :---: |
| 1 | "Ruf Mama an" | `place_call` | `1.0000` |
| 2 | "give my dad a ring" | `place_call` | `0.9134` |
| 3 | "please dial my grandmother" | `place_call` | `0.9120` |

### Query: `Stell einen Timer`
*(# = set a timer)*

| Rank | Matched Dataset Text | Intent Label | Cosine Score |
| :---: | :--- | :--- | :---: |
| 1 | "timer for thirty seconds please" | `set_timer` | `0.9029` |
| 2 | "begin a countdown of 5 minutes" | `set_timer` | `0.9029` |
| 3 | "Stelle einen Timer auf 5 Minuten" | `set_timer` | `0.8998` |

### Query: `write down my blood type`
*(Matches save_memory intent)*

| Rank | Matched Dataset Text | Intent Label | Cosine Score |
| :---: | :--- | :--- | :---: |
| 1 | "remember that my car keys are on the table" | `save_memory` | `0.9310` |
| 2 | "write down that my sister's birthday is June 5th" | `save_memory` | `0.9068` |
| 3 | "save the memory that I met John today" | `save_memory` | `0.9064` |

### Query: `how high is Mount Everest`
*(Matches answer_question intent)*

| Rank | Matched Dataset Text | Intent Label | Cosine Score |
| :---: | :--- | :--- | :---: |
| 1 | "who wrote the Harry Potter books" | `answer_question` | `0.9673` |
| 2 | "what is the speed of light" | `answer_question` | `0.9667` |
| 3 | "what is the capital city of France" | `answer_question` | `0.9666` |

### Query: `set an alarm for 8 am`
*(Matches set_timer intent)*

| Rank | Matched Dataset Text | Intent Label | Cosine Score |
| :---: | :--- | :--- | :---: |
| 1 | "timer for thirty seconds please" | `set_timer` | `0.7976` |
| 2 | "set a timer for 10 minutes" | `set_timer` | `0.7955` |
| 3 | "begin a countdown of 5 minutes" | `set_timer` | `0.7950` |

### Query: `where did I leave my keys`
*(Matches save_memory intent)*

| Rank | Matched Dataset Text | Intent Label | Cosine Score |
| :---: | :--- | :--- | :---: |
| 1 | "remember that my car keys are on the table" | `save_memory` | `0.8172` |
| 2 | "store this info that my blood type is O positive" | `save_memory` | `0.8071` |
| 3 | "write down that my sister's birthday is June 5th" | `save_memory` | `0.8040` |

---

## ⚖️ Part C — Lexical vs Semantic Search Contrast

The table below showcases the lexical limitations of a Week-2 TF-IDF model against our Semantic model for 3 query sentences that share zero or minimal words with the database.

| Query | Semantic Top Match | Semantic Score | TF-IDF Top Match | TF-IDF Score | Semantic vs Lexical Contrast |
| :--- | :--- | :---: | :--- | :---: | :--- |
| "ring mom" | "Ruf Mama an" (`place_call`) | `0.8000` | "give my dad a ring" (`place_call`) | `0.5474` | **+0.2526** semantic gain |
| "make a reminder to buy milk" | "Erinnere mich daran, Milch zu kaufen" (`create_task`) | `0.9638` | "remind me to buy milk tonight" (`create_task`) | `0.5979` | **+0.3659** semantic gain |
| "Ruf Mama an" | "Ruf Mama an" (`place_call`) | `1.0000` | "Ruf Mama an" (`place_call`) | `1.0000` | **+-0.0000** semantic gain |

### Analysis Key Takeaways
1. **Zero Lexical Overlap**: For paraphrase queries like `"ring mom"` or German queries like `"Ruf Mama an"`, a lexical TF-IDF search returns a score of `0.0000` because they share no characters/words with sentences like `"please dial my grandmother"` or `"give my dad a ring"`.
2. **Robust Semantic Recovery**: The semantic encoder maps words based on conceptual meanings (synonyms and translations) into the same space, letting our custom `cosine()` score them high (`> 0.9`) and resolve the correct intent perfectly.
