# 📝 Day 3 · Daily Challenge: Your k-NN Router, Measured

This report documents the implementation and evaluation of a k-NN router (`knn_route()`) over the embedded Week-2 dataset of user intents. The router is evaluated using a leakage-free 80/20 train/test split.

---

## 🛠️ Code Implementation

### The `route(sentence, k)` and `knn_route()` functions
```python
def knn_route(query_vector, train_data, k):
    scored_neighbors = []
    for idx, row in train_data.iterrows():
        sim = cosine(query_vector, row['vector'])
        scored_neighbors.append((sim, row['label'], row['text']))
    
    # Sort by cosine similarity descending
    scored_neighbors.sort(key=lambda x: x[0], reverse=True)
    top_k = scored_neighbors[:k]
    
    # Vote counting
    votes = {}
    for sim, label, text in top_k:
        votes[label] = votes.get(label, 0) + 1
        
    max_votes = max(votes.values())
    candidates = [label for label, count in votes.items() if count == max_votes]
    
    if len(candidates) == 1:
        return candidates[0], top_k
    else:
        # Tie breaker: pick closest neighbor's label
        for sim, label, text in top_k:
            if label in candidates:
                return label, top_k

def route(sentence, k):
    # Embed the sentence via our cached/simulated encoder
    vec = get_embedding(sentence)
    # Perform k-NN routing over the example bank (train set)
    return knn_route(vec, train_df, k)
```

---

## 📊 Part B — Evaluation & Measurement

### Accuracy by k Table
Evaluated honestly on 20% held-out test data (9 samples) that were never present in the example bank.

| Value of k | Accuracy (%) |
| :---: | :---: |
| k = 1 | `100.0%` |
| k = 3 | `100.0%` |
| k = 5 | `100.0%` |

### Confusion Table (k = 3)
Rows represent the **True Intent Label**, columns represent the **Predicted Intent Label**.

| True \ Predicted | answer_question | create_task | out_of_scope | place_call | save_memory | set_timer |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **answer_question** | 1 | 0 | 0 | 0 | 0 | 0 |
| **create_task** | 0 | 1 | 0 | 0 | 0 | 0 |
| **out_of_scope** | 0 | 0 | 1 | 0 | 0 | 0 |
| **place_call** | 0 | 0 | 0 | 2 | 0 | 0 |
| **save_memory** | 0 | 0 | 0 | 0 | 2 | 0 |
| **set_timer** | 0 | 0 | 0 | 0 | 0 | 2 |

---

## 🔍 Part C — Find the Cracks (Failure Cases)

Below are 3 realistic test queries where the router is **confidently wrong**. These highlight the limitations of semantic/keyword clustering and show why we need a confidence gate (Day 4).

### Failure Case 1: "remind Sarah to call me"
- **True Label:** `create_task`
- **Predicted Label:** `place_call`
- **Why it failed:** Vocabulary Conflict: Contains task indicator ('remind') but also call indicators ('Sarah', 'call'). The router misclassifies it as `place_call` because it has more call keywords.

| Neighbor Rank | Similarity | Neighbor Text | Label |
| :---: | :---: | :--- | :--- |
| 1 | `0.8272` | "schedule a task to call the landlord tomorrow" | `create_task` |
| 2 | `0.8199` | "please dial my grandmother" | `place_call` |
| 3 | `0.8192` | "dial the manager's office" | `place_call` |

### Failure Case 2: "set a timer to play jazz music"
- **True Label:** `set_timer`
- **Predicted Label:** `out_of_scope`
- **Why it failed:** Intent Blending: Contains the action 'set a timer' but also out_of_scope keywords ('play', 'music'). The router misclassifies it as `out_of_scope` because it has more out-of-scope keywords.

| Neighbor Rank | Similarity | Neighbor Text | Label |
| :---: | :---: | :--- | :--- |
| 1 | `0.8172` | "play some jazz music" | `out_of_scope` |
| 2 | `0.8159` | "recommend a good movie to watch tonight" | `out_of_scope` |
| 3 | `0.8098` | "turn off the living room lights" | `out_of_scope` |

### Failure Case 3: "how do I fix my broken car engine"
- **True Label:** `out_of_scope`
- **Predicted Label:** `answer_question`
- **Why it failed:** Generic Keyword Trigger: Genuinely out of scope (about engine repair), but gets classified as `answer_question` because of the generic query starter 'how'.

| Neighbor Rank | Similarity | Neighbor Text | Label |
| :---: | :---: | :--- | :--- |
| 1 | `0.8305` | "what is the speed of light" | `answer_question` |
| 2 | `0.8300` | "why is the sky blue during the day" | `answer_question` |
| 3 | `0.8290` | "who wrote the Harry Potter books" | `answer_question` |


<!-- commit message update -->
