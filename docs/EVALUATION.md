# 📐 Hệ Thống Đánh Giá: Video Understanding

## 1. Tổng Quan Evaluation Framework

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   VIDEO UNDERSTANDING EVALUATION FRAMEWORK                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    TASK-LEVEL EVALUATION                               │   │
│  │                                                                      │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │   │
│  │  │  Temporal   │  │   Visual    │  │  Narrative  │              │   │
│  │  │  Grounding  │  │     QA      │  │  Reasoning  │              │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘              │   │
│  │                                                                      │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │   │
│  │  │   Scene     │  │  Character  │  │   Action   │              │   │
│  │  │Description  │  │  Tracking   │  │ Recognition │              │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                     │                                         │
│                                     ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    METRIC LEVEL                                       │   │
│  │                                                                      │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │   │
│  │  │    R@K      │  │    BLEU     │  │   IoU@T     │              │   │
│  │  │ (Retrieval) │  │   CIDEr     │  │ (Temporal)  │              │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘              │   │
│  │                                                                      │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │   │
│  │  │   SPICE     │  │    EM       │  │    F1      │              │   │
│  │  │  (Semantic)  │  │  (Exact)    │  │  (Overlap) │              │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                     │                                         │
│                                     ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    DATASET LEVEL                                       │   │
│  │                                                                      │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │   │
│  │  │  Internal   │  │  External   │  │    Human    │              │   │
│  │  │  Benchmarks │  │   Standard  │  │   Eval      │              │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Evaluation Task Taxonomy

### 2.1 Task Definitions

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      TASK TAXONOMY FOR VIDEO UNDERSTANDING                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Task 1: TEMPORAL GROUNDING                                                 │
│  ───────────────────────────────                                            │
│  Definition: Given a natural language query, locate the corresponding      │
│              temporal segment in the video.                                  │
│  Example: "Find the scene where Jack draws Rose on Titanic"                 │
│  Input: Query text + Video                                                  │
│  Output: [start_time, end_time]                                            │
│  Difficulty: Requires semantic + temporal reasoning                          │
│                                                                              │
│  Task 2: VISUAL QUESTION ANSWERING (VQA)                                    │
│  ─────────────────────────────────────                                      │
│  Definition: Answer questions about visual content in video frames.          │
│  Example: "What color is the dress Rose is wearing?"                       │
│  Input: Video frames + Question                                             │
│  Output: Text answer                                                        │
│  Difficulty: Requires visual understanding + language generation             │
│                                                                              │
│  Task 3: NARRATIVE REASONING                                               │
│  ─────────────────────────────                                             │
│  Definition: Answer questions requiring multi-step causal/temporal reasoning.│
│  Example: "Why does Rose decide to let Jack go?"                           │
│  Input: Video + Narrative query                                             │
│  Output: Causal explanation                                                 │
│  Difficulty: Highest - requires world knowledge + reasoning                   │
│                                                                              │
│  Task 4: SCENE DESCRIPTION GENERATION                                      │
│  ─────────────────────────────────────                                      │
│  Definition: Generate detailed textual description of a video segment.      │
│  Example: "Describe the dinner scene in detail"                             │
│  Input: Video segment                                                       │
│  Output: Rich textual description                                           │
│  Difficulty: Requires comprehensive video understanding                      │
│                                                                              │
│  Task 5: CHARACTER TRACKING                                                 │
│  ────────────────────────                                                    │
│  Definition: Track character appearances and actions throughout video.     │
│  Example: "List all scenes where Rose appears"                              │
│  Input: Character name + Video                                               │
│  Output: List of [scene, start, end, description]                         │
│  Difficulty: Requires person re-identification + temporal reasoning         │
│                                                                              │
│  Task 6: CROSS-VIDEO RETRIEVAL                                            │
│  ──────────────────────────                                                │
│  Definition: Given a query, find the correct video from a collection.     │
│  Example: "Find the movie with the sinking ship scene"                     │
│  Input: Text/Image query + Video collection                                 │
│  Output: Ranked list of videos with timestamps                              │
│  Difficulty: Requires both retrieval and verification                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Metrics Definitions

### 3.1 Retrieval Metrics

```python
# Retrieval Metric Implementations

def recall_at_k(predictions: List[str], ground_truth: str, k: int) -> float:
    """
    R@K: Proportion of queries where ground truth is in top-k predictions.

    Args:
        predictions: Ranked list of predicted video IDs
        ground_truth: Correct video ID
        k: Cutoff rank

    Returns:
        Recall@K score (0.0 to 1.0)
    """
    top_k = predictions[:k]
    return 1.0 if ground_truth in top_k else 0.0


def mean_reciprocal_rank(predictions: List[List[str]], ground_truths: List[str]) -> float:
    """
    MRR: Mean of reciprocal ranks of ground truth in predictions.

    MRR = (1/N) * Σ(1/rank_i)

    where rank_i is the position of the ground truth in predictions.
    """
    rr_sum = 0.0
    for preds, gt in zip(predictions, ground_truths):
        for i, pred in enumerate(preds, 1):
            if pred == gt:
                rr_sum += 1.0 / i
                break
    return rr_sum / len(ground_truths)


def ndcg_at_k(predictions: List[List[str]], ground_truths: List[str], k: int) -> float:
    """
    NDCG@K: Normalized Discounted Cumulative Gain at K.

    Accounts for both relevance and position of predictions.
    """
    dcg = 0.0
    for i, pred in enumerate(predictions[:k], 1):
        rel = 1.0 if pred == ground_truths[i] else 0.0
        dcg += rel / math.log2(i + 1)

    # IDCG (ideal DCG)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(k, len(ground_truths)) + 1))

    return dcg / idcg if idcg > 0 else 0.0
```

### 3.2 Temporal Grounding Metrics

```python
# Temporal Grounding Metric Implementations

def temporal_iou(pred_start: float, pred_end: float,
                 gt_start: float, gt_end: float) -> float:
    """
    IoU for temporal segments.

    IoU = |intersection| / |union|
    """
    intersection = max(0, min(pred_end, gt_end) - max(pred_start, gt_start))
    union = max(pred_end, gt_end) - min(pred_start, gt_start)
    return intersection / union if union > 0 else 0.0


def recall_at_tiou(predictions: List[Tuple[float, float]],
                   ground_truths: List[Tuple[float, float]],
                   thresholds: List[float] = [0.3, 0.5, 0.7]) -> Dict[float, float]:
    """
    R@IoU@T: Recall at different IoU thresholds.

    For each IoU threshold τ, count how many predictions exceed it.
    """
    results = {}
    for tau in thresholds:
        correct = sum(
            1 for pred, gt in zip(predictions, ground_truths)
            if temporal_iou(pred[0], pred[1], gt[0], gt[1]) >= tau
        )
        results[f"R@{tau}"] = correct / len(ground_truths)
    return results


def temporal_accuracy_at_t(predictions: List[Tuple[float, float]],
                           ground_truths: List[Tuple[float, float]],
                           tolerance_seconds: float = 5.0) -> float:
    """
    TRec@T: Temporal accuracy within T seconds tolerance.

    A prediction is correct if |pred_center - gt_center| <= tolerance_seconds
    """
    correct = 0
    for (p_start, p_end), (gt_start, gt_end) in zip(predictions, ground_truths):
        pred_center = (p_start + p_end) / 2
        gt_center = (gt_start + gt_end) / 2
        if abs(pred_center - gt_center) <= tolerance_seconds:
            correct += 1
    return correct / len(predictions)


def temporal_precision_recall(predictions: List[Tuple[float, float]],
                              ground_truths: List[Tuple[float, float]]) -> Tuple[float, float]:
    """
    Temporal Precision/Recall considering temporal overlap.
    """
    precisions = []
    recalls = []

    for pred, gt in zip(predictions, ground_truths):
        iou = temporal_iou(pred[0], pred[1], gt[0], gt[1])

        # Precision: how much of prediction is correct
        pred_duration = pred[1] - pred[0]
        if pred_duration > 0:
            precision = iou  # Simplified: IoU as precision proxy
        else:
            precision = 0.0

        # Recall: how much of ground truth is covered
        gt_duration = gt[1] - gt[0]
        if gt_duration > 0:
            recall = iou  # Simplified: IoU as recall proxy
        else:
            recall = 0.0

        precisions.append(precision)
        recalls.append(recall)

    return sum(precisions) / len(precisions), sum(recalls) / len(recalls)
```

### 3.3 Generation Metrics

```python
# Text Generation Metric Implementations

def bleu_score(prediction: str, reference: str, n: int = 4) -> float:
    """
    BLEU: Bilingual Evaluation Understudy Score.

    Measures n-gram overlap between prediction and reference.

    BLEU = BP * exp(Σ w_n * log precision_n)

    where BP is brevity penalty and w_n are weights (typically 1/n).
    """
    from collections import Counter

    def get_ngrams(tokens, n):
        return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

    prediction_tokens = prediction.lower().split()
    reference_tokens = reference.lower().split()

    precisions = []
    for i in range(1, n+1):
        pred_ngrams = Counter(get_ngrams(prediction_tokens, i))
        ref_ngrams = Counter(get_ngrams(reference_tokens, i))

        matches = sum((pred_ngrams & ref_ngrams).values())
        total = sum(pred_ngrams.values())

        if total == 0:
            precisions.append(0.0)
        else:
            precisions.append(matches / total)

    # Brevity penalty
    bp = 1.0 if len(prediction_tokens) >= len(reference_tokens) \
         else exp(1 - len(reference_tokens) / len(prediction_tokens))

    return bp * exp(sum(log(p + 1e-10) for p in precisions) / n)


def cider_score(predictions: List[str], references: List[str]) -> float:
    """
    CIDEr: Consensus-based Image Description Evaluation.

    Measures TF-IDF weighted n-gram overlap.
    """
    from collections import Counter
    import math

    def to_ngrams(sentence, n):
        return [tuple(sentence[i:i+n]) for i in range(len(sentence)-n+1)]

    def compute_tf_idf(ngrams, corpus_ngrams, doc_length, avg_doc_length):
        tf = Counter(ngrams)
        idf = {}
        for gram in set(ngrams):
            idf[gram] = math.log((len(corpus_ngrams) + 1) /
                                  (1 + sum(1 for doc in corpus_ngrams if gram in doc)))
        return {gram: tf[gram] * idf.get(gram, 0) for gram in tf}

    preds_tokens = [p.lower().split() for p in predictions]
    refs_tokens = [r.lower().split() for r in references]

    cider_scores = []
    for pred, ref in zip(preds_tokens, refs_tokens):
        score = 0.0
        for n in range(1, 5):
            pred_ngrams = to_ngrams(pred, n)
            ref_ngrams = to_ngrams(ref, n)

            # Simplified CIDEr computation
            pred_counts = Counter(pred_ngrams)
            ref_counts = Counter(ref_ngrams)

            overlap = sum((pred_counts & ref_counts).values())
            if sum(pred_counts.values()) > 0 and sum(ref_counts.values()) > 0:
                score += overlap / (sum(pred_counts.values()) + sum(ref_counts.values()))

        cider_scores.append(score / 4)

    return sum(cider_scores) / len(cider_scores)


def spice_score(prediction: str, reference: str) -> float:
    """
    SPICE: Semantic Propositional Image Caption Evaluation.

    Measures overlap of scene graph tuples (objects, attributes, relations).

    Requires parsing to scene graphs first - using simplified version.
    """
    # Simplified: use F1 of named entities and actions
    pred_tokens = set(prediction.lower().split())
    ref_tokens = set(reference.lower().split())

    # Common tokens as proxy for scene graph overlap
    overlap = len(pred_tokens & ref_tokens)
    total = len(pred_tokens | ref_tokens)

    return overlap / total if total > 0 else 0.0


def exact_match(prediction: str, reference: str) -> float:
    """
    EM: Exact Match accuracy.

    1.0 if prediction == reference, 0.0 otherwise.
    """
    return 1.0 if prediction.strip() == reference.strip() else 0.0


def f1_score(prediction: str, reference: str) -> float:
    """
    Token-level F1 between prediction and reference.
    """
    pred_tokens = set(prediction.lower().split())
    ref_tokens = set(reference.lower().split())

    tp = len(pred_tokens & ref_tokens)
    fp = len(pred_tokens - ref_tokens)
    fn = len(ref_tokens - pred_tokens)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision + recall > 0:
        return 2 * precision * recall / (precision + recall)
    return 0.0
```

---

## 4. Benchmark Protocols

### 4.1 Internal Benchmark: Video Understanding Test (VUT-100)

**Tổ chức:** 100 query-video pairs từ MovieNet/MovieGraphs

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    VUT-100 BENCHMARK STRUCTURE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Dataset Split:                                                              │
│  ─────────────                                                               │
│  ├── Train: 60 movies (60% of queries)                                      │
│  ├── Val: 20 movies (20% of queries)                                       │
│  └── Test: 20 movies (20% of queries = 100 queries)                        │
│                                                                              │
│  Query Distribution:                                                         │
│  ──────────────────                                                         │
│  ├── Temporal Grounding: 25 queries                                        │
│  ├── Visual QA: 25 queries                                                 │
│  ├── Narrative Reasoning: 25 queries                                       │
│  └── Scene Description: 25 queries                                         │
│                                                                              │
│  Query Examples:                                                             │
│  ───────────────                                                            │
│  Temporal: "When does Rose first meet Jack?"                                │
│  Visual QA: "Describe the color of the life jackets in the scene"          │
│  Narrative: "Why does the ship split in half?"                             │
│  Scene Desc: "Give a detailed description of the dinner scene"             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 External Benchmark Mappings

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    EXTERNAL BENCHMARK ALIGNMENT                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  External Benchmark     →   Internal Task     →   Primary Metrics         │
│  ─────────────────────────────────────────────────────────────────────     │
│  Charades-STA          →   Temporal Grounding →   R@IoU@0.5, R@IoU@0.7   │
│  DiDeMo                →   Temporal Grounding →   R@1, MRR                 │
│  MSR-VTT-QA           →   Visual QA          →   Accuracy                 │
│  LSMDC                 →   Scene Description  →   CIDEr, SPICE             │
│  CinePile-QA          →   Narrative QA       →   Accuracy, F1             │
│  ActivityNet-QA       →   Visual QA          →   Accuracy                 │
│  MSR-VTT Retrieval    →   Cross-Video Retr.  →   R@5, R@10, MRR           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Evaluation Protocols

### 5.1 Temporal Grounding Evaluation Protocol

```python
# temporal_grounding_eval.py

class TemporalGroundingEvaluator:
    """
    Evaluator for temporal grounding tasks.
    """

    def __init__(self, ground_truth_file: str):
        """
        Load ground truth from file.

        Format:
        {
            "query_id": {
                "video_id": "tt0120338",
                "query": "When does Jack draw Rose?",
                "ground_truth": {
                    "start": 5760.0,  # seconds
                    "end": 5850.0,
                    "description": "The famous drawing scene..."
                }
            }
        }
        """
        with open(ground_truth_file) as f:
            self.ground_truth = json.load(f)

    def evaluate(self, predictions: Dict[str, Dict]) -> Dict[str, float]:
        """
        Evaluate predictions against ground truth.

        Args:
            predictions: {
                "query_id": {
                    "start": 5800.0,
                    "end": 5900.0,
                    "confidence": 0.95
                }
            }

        Returns:
            Dict of metric_name -> score
        """
        results = {
            "num_queries": len(self.ground_truth),
            "iou@0.3": [],
            "iou@0.5": [],
            "iou@0.7": [],
            "trec@5s": [],
            "trec@10s": [],
            "trec@30s": [],
        }

        for query_id, gt in self.ground_truth.items():
            if query_id not in predictions:
                continue

            pred = predictions[query_id]
            iou = temporal_iou(
                pred["start"], pred["end"],
                gt["ground_truth"]["start"], gt["ground_truth"]["end"]
            )

            results["iou@0.3"].append(iou >= 0.3)
            results["iou@0.5"].append(iou >= 0.5)
            results["iou@0.7"].append(iou >= 0.7)

            # Temporal accuracy
            pred_center = (pred["start"] + pred["end"]) / 2
            gt_center = (gt["ground_truth"]["start"] + gt["ground_truth"]["end"]) / 2

            results["trec@5s"].append(abs(pred_center - gt_center) <= 5)
            results["trec@10s"].append(abs(pred_center - gt_center) <= 10)
            results["trec@30s"].append(abs(pred_center - gt_center) <= 30)

        # Aggregate
        return {
            "R@IoU@0.3": mean(results["iou@0.3"]),
            "R@IoU@0.5": mean(results["iou@0.5"]),
            "R@IoU@0.7": mean(results["iou@0.7"]),
            "TRec@5s": mean(results["trec@5s"]),
            "TRec@10s": mean(results["trec@10s"]),
            "TRec@30s": mean(results["trec@30s"]),
            "mIoU": mean([
                temporal_iou(p["start"], p["end"],
                             gt["start"], gt["end"])
                for query_id, (p, gt) in ...  # paired properly
            ])
        }
```

### 5.2 Visual QA Evaluation Protocol

```python
# visual_qa_eval.py

class VisualQAEvaluator:
    """
    Evaluator for visual question answering tasks.
    """

    def __init__(self, ground_truth_file: str):
        with open(ground_truth_file) as f:
            self.ground_truth = json.load(f)

    def evaluate(self, predictions: Dict[str, str]) -> Dict[str, float]:
        """
        Evaluate VQA predictions.

        Supports multiple answer formats:
        - Free-form text (evaluated by F1, SPICE)
        - Multiple choice (evaluated by accuracy)
        - Yes/No (evaluated by accuracy)
        """
        results = {
            "accuracy": [],
            "f1": [],
            "bleu4": [],
        }

        for query_id, gt in self.ground_truth.items():
            if query_id not in predictions:
                continue

            pred = predictions[query_id]

            # Multiple choice
            if gt.get("type") == "mc":
                results["accuracy"].append(
                    1.0 if pred.get("answer") == gt["answer"] else 0.0
                )

            # Free-form text
            else:
                results["accuracy"].append(exact_match(pred, gt["answer"]))
                results["f1"].append(f1_score(pred, gt["answer"]))
                results["bleu4"].append(bleu_score(pred, gt["answer"], 4))

        return {
            "Accuracy": mean(results["accuracy"]),
            "F1": mean(results["f1"]),
            "BLEU-4": mean(results["bleu4"]),
        }
```

### 5.3 Scene Description Evaluation Protocol

```python
# scene_description_eval.py

class SceneDescriptionEvaluator:
    """
    Evaluator for scene description generation.
    """

    def __init__(self, ground_truth_file: str):
        """
        Load multiple reference descriptions per scene.

        Format:
        {
            "scene_id": {
                "references": [
                    "A man and woman dancing on a ship.",
                    "Jack and Rose dancing romantically."
                ],
                "aspects": {
                    "characters": ["man", "woman"],
                    "actions": ["dancing"],
                    "setting": ["ship"]
                }
            }
        }
        """
        with open(ground_truth_file) as f:
            self.ground_truth = json.load(f)

    def evaluate(self, predictions: Dict[str, str]) -> Dict[str, float]:
        """
        Evaluate scene descriptions using multiple metrics.
        """
        all_cider = []
        all_bleu4 = []
        all_spice = []

        for scene_id, gt in self.ground_truth.items():
            if scene_id not in predictions:
                continue

            pred = predictions[scene_id]
            refs = gt["references"]

            # CIDEr (compare against all references, take max)
            cider_scores = [cider_score(pred, ref) for ref in refs]
            all_cider.append(max(cider_scores))

            # BLEU-4 (compare against all references, take max)
            bleu_scores = [bleu_score(pred, ref, 4) for ref in refs]
            all_bleu4.append(max(bleu_scores))

            # SPICE (aspect-based evaluation)
            spice = self._evaluate_aspects(pred, gt.get("aspects", {}))
            all_spice.append(spice)

        return {
            "CIDEr": mean(all_cider),
            "BLEU-4": mean(all_bleu4),
            "SPICE": mean(all_spice),
            "num_evaluated": len(all_cider)
        }

    def _evaluate_aspects(self, prediction: str, aspects: Dict) -> float:
        """
        Evaluate aspect-level metrics.

        Aspects: characters, actions, setting, objects, etc.
        """
        pred_lower = prediction.lower()
        scores = []

        for aspect_type, aspect_values in aspects.items():
            if not aspect_values:
                continue

            matches = sum(1 for val in aspect_values if val.lower() in pred_lower)
            scores.append(matches / len(aspect_values))

        return mean(scores) if scores else 0.0
```

---

## 6. Ablation Studies

### 6.1 Component Ablation Design

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ABLATION STUDY DESIGN                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Experiment: Remove one component at a time, measure impact                 │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │ Component             │ Removal Effect   │ Expected Impact          │    │
│  ├──────────────────────┼─────────────────┼──────────────────────────┤    │
│  │ VLM Analysis          │ No scene desc   │ ↓ Scene understanding     │    │
│  │ Cross-encoder Rerank │ FAISS only      │ ↓ Retrieval accuracy      │    │
│  │ Scene Index (L1)     │ Frame only      │ ↓ Semantic search         │    │
│  │ Whisper/STT          │ No dialogue     │ ↓ Dialog understanding   │    │
│  │ GraphRAG (Neo4j)     │ No entity links │ ↓ Multi-hop reasoning    │    │
│  │ Script Alignment     │ No script ctx   │ ↓ Narrative coherence    │    │
│  │ ICA Strategy         │ Single retrieval│ ↓ Recall                 │    │
│  │ Visual Lexical Bonus │ CLIP only       │ ↓ Text-image match        │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Ablation Experiment Template

```python
# ablation_experiment.py

ABLATION_CONFIGS = {
    "baseline": {
        "vlm_analysis": False,
        "cross_encoder_rerank": False,
        "scene_index": False,
        "whisper": False,
        "graph_rag": False,
        "script_alignment": False,
        "ica": False,
        "visual_lexical_bonus": False,
    },
    "+vlm": {
        "vlm_analysis": True,
        "cross_encoder_rerank": False,
        "scene_index": False,
        "whisper": False,
        "graph_rag": False,
        "script_alignment": False,
        "ica": False,
        "visual_lexical_bonus": False,
    },
    "+vlm+cross": {
        "vlm_analysis": True,
        "cross_encoder_rerank": True,
        "scene_index": False,
        "whisper": False,
        "graph_rag": False,
        "script_alignment": False,
        "ica": False,
        "visual_lexical_bonus": False,
    },
    # ... full grid
    "full": {
        "vlm_analysis": True,
        "cross_encoder_rerank": True,
        "scene_index": True,
        "whisper": True,
        "graph_rag": True,
        "script_alignment": True,
        "ica": True,
        "visual_lexical_bonus": True,
    }
}

def run_ablation():
    """
    Run full ablation study and generate comparison table.
    """
    results = {}

    for config_name, config in ABLATION_CONFIGS.items():
        print(f"Running ablation: {config_name}")
        pipeline = VideoUnderstandingPipeline(config)
        metrics = evaluate(pipeline, test_set)
        results[config_name] = metrics

    # Generate comparison table
    print("\n" + "="*80)
    print("ABLATION STUDY RESULTS")
    print("="*80)
    print(f"{'Config':<30} {'R@IoU@0.5':<15} {'TRec@10s':<15} {'CIDEr':<15}")
    print("-"*80)

    for config_name, metrics in sorted(results.items(), key=lambda x: -x[1]["R@IoU@0.5"]):
        print(f"{config_name:<30} {metrics['R@IoU@0.5']:<15.3f} "
              f"{metrics['TRec@10s']:<15.3f} {metrics['CIDEr']:<15.3f}")
```

---

## 7. Statistical Significance Testing

### 7.1 Paired t-test Protocol

```python
# statistical_significance.py

from scipy import stats
import numpy as np

def compute_statistical_significance(
    baseline_results: List[float],
    experiment_results: List[float],
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Compute statistical significance using paired t-test.

    Args:
        baseline_results: Metric values for baseline system (n=100)
        experiment_results: Metric values for experiment system (n=100)
        alpha: Significance level (default 0.05)

    Returns:
        Dict with t-statistic, p-value, and significance determination
    """
    assert len(baseline_results) == len(experiment_results)

    # Paired t-test
    t_stat, p_value = stats.ttest_rel(experiment_results, baseline_results)

    # Effect size (Cohen's d)
    diff = np.array(experiment_results) - np.array(baseline_results)
    cohens_d = np.mean(diff) / np.std(diff)

    # 95% Confidence Interval
    mean_diff = np.mean(diff)
    se = np.std(diff) / np.sqrt(len(diff))
    ci_lower = mean_diff - 1.96 * se
    ci_upper = mean_diff + 1.96 * se

    return {
        "t_statistic": t_stat,
        "p_value": p_value,
        "significant": p_value < alpha,
        "cohens_d": cohens_d,
        "mean_improvement": mean_diff,
        "ci_95": (ci_lower, ci_upper),
        "effect_size_interpretation": interpret_cohens_d(cohens_d)
    }


def interpret_cohens_d(d: float) -> str:
    """
    Interpret Cohen's d effect size.
    """
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"
```

---

## 8. Human Evaluation Protocol

### 8.1 Human Evaluation Interface Design

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       HUMAN EVALUATION INTERFACE                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Query: "Describe the scene where Jack draws Rose"                          │
│  Ground Truth: "In this scene, Jack is sitting at a table..."               │
│                                                                              │
│  ─────────────────────────────────────────────────────────────────────     │
│                                                                              │
│  Model Output:                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ "Jack is sitting with Rose on a couch in their cabin. He takes       │   │
│  │ a pencil and starts drawing her portrait. Rose smiles as she         │   │
│  │ watches him draw."                                                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Rate the following aspects (1-5 scale):                                     │
│                                                                              │
│  Relevance:      [1] [2] [3] [4] [5]                                      │
│  Fluency:        [1] [2] [3] [4] [5]                                      │
│  Completeness:   [1] [2] [3] [4] [5]                                      │
│  Accuracy:       [1] [2] [3] [4] [5]                                      │
│                                                                              │
│  Does it correctly identify the scene? [ ] Yes  [ ] No  [ ] Partial       │
│                                                                              │
│  Comments: ________________________________________________________________ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 8.2 Inter-Annotator Agreement

```python
# compute_agreement.py

def compute_krippendorffs_alpha(annotations: List[Dict]) -> float:
    """
    Compute Krippendorff's Alpha for inter-annotator agreement.

    Suitable for ordinal ratings (1-5 scales).
    """
    from scipy.stats import krippendorff

    # Convert to matrix: rows = items, cols = annotators
    ratings_matrix = np.array([
        [ann["relevance"] for ann in annotations],
        [ann["fluency"] for ann in annotations],
        [ann["completeness"] for ann in annotations],
        [ann["accuracy"] for ann in annotations],
    ])

    alpha = krippendorff.alpha(reliability_data=ratings_matrix,
                                level_of_measurement='ordinal')
    return alpha


def compute_fleiss_kappa(ratings: np.ndarray, n_categories: int) -> float:
    """
    Compute Fleiss' Kappa for inter-annotator agreement.

    Args:
        ratings: Matrix of shape (n_items, n_annotators)
                 where each entry is a category index
        n_categories: Number of possible categories
    """
    N = ratings.shape[0]  # number of items
    n = ratings.shape[1]  # number of annotators

    # Count agreements per category
    P_i = []
    for i in range(N):
        counts = np.bincount(ratings[i], minlength=n_categories)
        p_i = (1 / (n * (n - 1))) * sum(c * (c - 1) for c in counts)
        P_i.append(p_i)

    P_bar = sum(P_i) / N

    # Category proportions
    p_c = np.bincount(ratings.flatten(), minlength=n_categories) / (N * n)

    # Expected agreement
    P_e = sum(p_c * p_c)

    kappa = (P_bar - P_e) / (1 - P_e) if (1 - P_e) > 0 else 0
    return kappa
```

---

## 9. Evaluation Dashboard

### 9.1 Metrics Summary Table

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      EVALUATION METRICS SUMMARY                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Task                    Primary Metrics      Target     Baseline           │
│  ───────────────────────────────────────────────────────────────────────    │
│  Temporal Grounding     R@IoU@0.5           > 0.45      0.32 (CLIP)        │
│                        TRec@10s            > 0.75      0.58               │
│                        MRR                 > 0.55      0.41               │
│                                                                              │
│  Visual QA              Accuracy            > 0.70      0.52               │
│                        F1                  > 0.65      0.48               │
│                                                                              │
│  Narrative Reasoning     Accuracy            > 0.60      0.40               │
│                        F1                  > 0.55      0.35               │
│                                                                              │
│  Scene Description      CIDEr               > 0.40      0.25               │
│                        SPICE               > 0.35      0.22               │
│                        BLEU-4              > 0.30      0.18               │
│                                                                              │
│  Cross-Video Retrieval  R@5                 > 0.80      0.62               │
│                        MRR                 > 0.60      0.45               │
│                                                                              │
│  Character Tracking     Accuracy            > 0.85      0.70               │
│                        F1                  > 0.80      0.65               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 9.2 Result Logging Schema

```python
# eval_result.py

@dataclass
class EvalResult:
    """Structured evaluation result."""

    # Identification
    experiment_id: str
    timestamp: datetime
    config: Dict[str, Any]

    # Dataset info
    dataset: str  # "VUT-100", "Charades-STA", etc.
    split: str  # "test", "val"
    num_queries: int

    # Metrics
    metrics: Dict[str, float]

    # Statistical info
    std_dev: Dict[str, float]
    ci_95: Dict[str, Tuple[float, float]]
    p_value: float  # vs baseline

    # Failure analysis
    error_cases: List[Dict]  # [{query_id, predicted, expected, error_type}]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)

    def summary_table_row(self) -> str:
        """Format for results table."""
        metrics_str = ", ".join(f"{k}={v:.3f}" for k, v in self.metrics.items())
        return f"| {self.experiment_id:<20} | {self.dataset:<15} | {metrics_str:<40} |"
```

---

## 10. Benchmark Results Format

### 10.1 Standard Results JSON

```json
{
  "benchmark": "VUT-100",
  "version": "1.0",
  "date": "2026-03-19",
  "system": "VideoUnderstandingPipeline",
  "config": {
    "clip_model": "ViT-L/14",
    "vlm_model": "Qwen2-VL-7B",
    "whisper_model": "medium",
    "use_scene_index": true,
    "use_graph_rag": true,
    "use_ica": true
  },
  "results": {
    "temporal_grounding": {
      "R@IoU@0.3": 0.68,
      "R@IoU@0.5": 0.52,
      "R@IoU@0.7": 0.31,
      "TRec@5s": 0.71,
      "TRec@10s": 0.82,
      "TRec@30s": 0.93,
      "MRR": 0.61
    },
    "visual_qa": {
      "Accuracy": 0.74,
      "F1": 0.68,
      "BLEU-4": 0.42
    },
    "narrative_reasoning": {
      "Accuracy": 0.63,
      "F1": 0.58
    },
    "scene_description": {
      "CIDEr": 0.45,
      "SPICE": 0.38,
      "BLEU-4": 0.35
    },
    "cross_video_retrieval": {
      "R@1": 0.58,
      "R@5": 0.84,
      "R@10": 0.91,
      "MRR": 0.67
    }
  },
  "ablation": {
    "-VLM": {"R@IoU@0.5": 0.44, "delta": -0.08},
    "-SceneIndex": {"R@IoU@0.5": 0.46, "delta": -0.06},
    "-GraphRAG": {"R@IoU@0.5": 0.48, "delta": -0.04}
  },
  "errors": [
    {
      "query_id": "titanic_025",
      "error_type": "temporal_mismatch",
      "predicted": [5760, 5850],
      "expected": [5780, 5820],
      "iou": 0.33
    }
  ]
}
```
