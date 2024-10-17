import logging
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple
import pytrec_eval

logger = logging.getLogger(__name__)

def evaluate_rag(
            qrels: Dict[str, Dict[str, int]],
            results: Dict[str, Dict[str, float]],
            k_values: List[int],
            ignore_identical_ids: bool = True
    ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]:

        if ignore_identical_ids:
            logger.info(
                'For evaluation, we ignore identical query and document ids (default), '
                'please explicitly set ``ignore_identical_ids=False`` to ignore this.')
            popped = []
            for qid, rels in results.items():
                for pid in list(rels):
                    if qid == pid:
                        results[qid].pop(pid)  # remove identical query-document pairs
                        popped.append(pid)

        # Filter results to only keep queries that are present in qrels
        filtered_results = {qid: rels for qid, rels in results.items() if qid in qrels}

        # Initialize dictionaries for evaluation metrics
        ndcg = {}
        _map = {}
        recall = {}
        precision = {}

        # Initialize metric values for each k in k_values
        for k in k_values:
            ndcg[f"NDCG@{k}"] = 0.0
            _map[f"MAP@{k}"] = 0.0
            recall[f"Recall@{k}"] = 0.0
            precision[f"P@{k}"] = 0.0

        # Define strings for pytrec_eval evaluation
        map_string = "map_cut." + ",".join([str(k) for k in k_values])
        ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
        recall_string = "recall." + ",".join([str(k) for k in k_values])
        precision_string = "P." + ",".join([str(k) for k in k_values])

        # Perform evaluation using pytrec_eval with filtered results
        evaluator = pytrec_eval.RelevanceEvaluator(qrels,
                                                   {map_string, ndcg_string, recall_string, precision_string})
        scores = evaluator.evaluate(filtered_results)

        # Aggregate the scores for each query and each k
        for query_id in scores.keys():
            for k in k_values:
                ndcg[f"NDCG@{k}"] += scores[query_id]["ndcg_cut_" + str(k)]
                _map[f"MAP@{k}"] += scores[query_id]["map_cut_" + str(k)]
                recall[f"Recall@{k}"] += scores[query_id]["recall_" + str(k)]
                precision[f"P@{k}"] += scores[query_id]["P_" + str(k)]

        # Compute the average scores for each k
        for k in k_values:
            ndcg[f"NDCG@{k}"] = round(ndcg[f"NDCG@{k}"] / len(scores), 5)
            _map[f"MAP@{k}"] = round(_map[f"MAP@{k}"] / len(scores), 5)
            recall[f"Recall@{k}"] = round(recall[f"Recall@{k}"] / len(scores), 5)
            precision[f"P@{k}"] = round(precision[f"P@{k}"] / len(scores), 5)

        # Log the results for each metric
        for _eval in [ndcg, _map, recall, precision]:
            logger.info("\n")
            for k in _eval.keys():
                logger.info("{}: {:.4f}".format(k, _eval[k]))

        return ndcg, _map, recall, precision