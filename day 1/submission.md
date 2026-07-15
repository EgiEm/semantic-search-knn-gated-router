# 📝 Day 1 · Daily Challenge: Map Your Dataset into Meaning-Space

## Part A — Embedded Dataset Table (Sample)
The original dataset `intents.csv` has been embedded using 1024-dimension float vectors. Below is the first 10 rows showcasing the vector dimensions and the first 5 numbers:

| Text | Label | Vector Dimensions | First 5 Vector Values |
| :--- | :--- | :---: | :--- |
| "remind me to buy milk tonight" | `create_task` | 1024 | `[0.0338, -0.0043, 0.0327, 0.0378, -0.0207, ...]` |
| "remind me to bye milk" | `create_task` | 1024 | `[0.0037, 0.0114, 0.0499, 0.0556, -0.0286, ...]` |
| "add an item to my to-do list: clean the kitchen" | `create_task` | 1024 | `[0.0184, -0.0018, 0.0484, 0.0352, -0.0114, ...]` |
| "schedule a task to call the landlord tomorrow" | `create_task` | 1024 | `[0.0262, 0.0084, 0.0423, 0.0441, -0.0191, ...]` |
| "don't let me forget to lock the garage" | `create_task` | 1024 | `[0.0318, 0.0036, 0.0349, 0.0461, -0.0203, ...]` |
| "put wash the car on my chores list" | `create_task` | 1024 | `[0.0207, -0.0078, 0.0343, 0.0422, 0.0088, ...]` |
| "Erinnere mich daran, Milch zu kaufen" | `create_task` | 1024 | `[0.0169, -0.0057, 0.0414, 0.0451, -0.0301, ...]` |
| "call Sarah right now" | `place_call` | 1024 | `[-0.0248, -0.0065, -0.0523, -0.0629, -0.0434, ...]` |
| "give my dad a ring" | `place_call` | 1024 | `[-0.0086, 0.0042, -0.0409, -0.0381, -0.0748, ...]` |
| "phone the pizza restaurant" | `place_call` | 1024 | `[-0.0222, -0.0110, -0.0478, -0.0408, -0.0542, ...]` |

---

## Part B — Semantic Neighbours Table (TF-IDF vs Embeddings)
The following table shows 5 hand-picked paraphrase pairs (including 3 cross-language German-to-English translations). 
- **TF-IDF Cosine Similarity**: A lexical score measuring shared words (ranges 0 to 1, higher is closer).
- **Euclidean Distance**: A semantic distance in the embedding space (ranges 0+, lower is closer).

| Intent Pair Description | Sentence A | Sentence B | TF-IDF Cosine (Lexical) | Embedding Euclidean Distance (Semantic) |
| :--- | :--- | :--- | :---: | :---: |
| Call intent: alternate phrasing | "call Sarah right now" | "give my dad a ring" | `0.0000` | `0.4652` |
| Task intent: German cross-language paraphrase | "remind me to buy milk tonight" | "Erinnere mich daran, Milch zu kaufen (Gloss: "remind me to buy milk")" | `0.0000` | `0.4972` |
| Call intent: German cross-language paraphrase | "please dial my grandmother" | "Ruf Mama an (Gloss: "call mom")" | `0.0000` | `0.4548` |
| Timer intent: German cross-language paraphrase | "set a timer for 10 minutes" | "Stelle einen Timer auf 5 Minuten (Gloss: "set a timer for 5 minutes")" | `0.1403` | `0.2149` |
| Memory intent: alternate phrasing | "write down that my sister's birthday is June 5th" | "store this info that my blood type is O positive" | `0.1645` | `0.2961` |

---

## 💡 Semantic Reflection Note
1. **Context over Characters**: Text embeddings capture the conceptual meaning and underlying intent of a sentence rather than the literal spelling of its words, grouping synonyms like "call" and "ring" together.
2. **Cross-Lingual Alignment**: Multilingual models like `bge-m3` embed corresponding sentences across languages (e.g. English and German) near each other in space, matching "Ruf Mama an" directly to "call mom" despite sharing no words.
3. **Lexical Limitations**: Lexical approaches like TF-IDF treat unique words as distinct perpendicular dimensions, failing completely on paraphrases and translations by scoring their similarity as zero.

<!-- commit message update -->
