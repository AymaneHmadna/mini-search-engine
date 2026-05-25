import numpy as np
from search_engine import SearchEngine
GROUND_TRUTH = {
    "math applied to ai": [
        "From Statistical Relational to Neurosymbolic Artif_2108.11451v4.txt",
        "MerLin A Discovery Engine for Photonic and Hybrid_2602.11092v2.txt",
        "Memristors -- from In-memory computing Deep Learn_2004.14942v1.txt"
    ],
    "how machines process information": [
        "What Does Information Science Offer for Data Scien_2506.03165v1.txt",
        "Exploring the Landscape of Natural Language Proces_2307.10652v5.txt",
        "PyThaiNLP Thai Natural Language Processing in Pyt_2312.04649v1.txt"
    ],
    "teaching computers to play games": [
        "Accelerating Training in Pommerman with Imitation_1911.04947v2.txt",
        "Memory-Efficient Episodic Control Reinforcement Le_1911.09560v1.txt"
    ],
    "making algorithms fair and unbiased": [
        "Reinforcement Learning with Stepwise Fairness Cons_2211.03994v1.txt"
    ],
    "extreme q-learning maclaurin expansion": [
        "Stabilizing Extreme Q-learning by Maclaurin Expans_2406.04896v2.txt"
    ],
    "episodic control sample and memory efficiency": [
        "Memory-Efficient Episodic Control Reinforcement Le_1911.09560v1.txt",
        "Sample-Efficient Reinforcement Learning with Maxim_1911.09615v1.txt"
    ],
    "imitation learning pommerman": [
        "Accelerating Training in Pommerman with Imitation_1911.04947v2.txt"
    ],
    "stepwise fairness constraints": [
        "Reinforcement Learning with Stepwise Fairness Cons_2211.03994v1.txt"
    ]
}
def calculate_metrics(retrieved_docs, relevant_docs):
    if not relevant_docs:
        return 0.0, 0.0, 0.0
    retrieved_set = set(retrieved_docs)
    relevant_set = set(relevant_docs)
    true_positives = len(retrieved_set.intersection(relevant_set))
    precision = true_positives / len(retrieved_set) if retrieved_set else 0.0
    recall = true_positives / len(relevant_set)
    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0.0
    return precision, recall, f1_score
def evaluate_models():
    engine = SearchEngine()
    algorithms = ["Cosine", "BM25"]
    avg_metrics = {algo: {"P": [], "R": [], "F1": []} for algo in algorithms}
    top_k = 5
    for query, relevant_docs in GROUND_TRUTH.items():
        results = {
            "Cosine": [res["document"] for res in engine.search_cosine(query, top_k)],
            "BM25": [res["document"] for res in engine.search_bm25(query, top_k)]
        }
        for algo in algorithms:
            retrieved_docs = results[algo]
            p, r, f1 = calculate_metrics(retrieved_docs, relevant_docs)
            avg_metrics[algo]["P"].append(p)
            avg_metrics[algo]["R"].append(r)
            avg_metrics[algo]["F1"].append(f1)
    print("Resultats finaux")
    for algo in algorithms:
        mean_p = np.mean(avg_metrics[algo]["P"])
        mean_r = np.mean(avg_metrics[algo]["R"])
        mean_f1 = np.mean(avg_metrics[algo]["F1"])
        print(f"> {algo}")
        print(f"   Précision : {mean_p:.4f}")
        print(f"   Rappel    : {mean_r:.4f}")
        print(f"   F1-Score  : {mean_f1:.4f}\n")
if __name__ == "__main__":
    evaluate_models()
