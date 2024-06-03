class Evaluator:
    def __init__(self, search_engine):
        self.search_engine = search_engine

    @staticmethod
    def calculate_precision_recall(retrieved_docs, relevant_docs):
        num_retrieved = len(retrieved_docs)
        num_relevant = len(relevant_docs)
        intersection = set(retrieved_docs) & set(relevant_docs)
        num_intersection = len(intersection)
        precision = num_intersection / num_retrieved if num_retrieved > 0 else 0
        recall = num_intersection / num_relevant if num_relevant > 0 else 0
        return precision, recall

    def calculate_avg_precision_recall(self, dataset, data_set_index, num_of_queries):
        avg_precision, avg_recall = 0, 0
        for _, qrel in zip(range(num_of_queries), dataset.qrels_iter()):
            query = None
            for itr_query in dataset.queries_iter():
                if itr_query.query_id == qrel.query_id:
                    query = itr_query.text
                    break
            retrieved_docs = self.search_engine.search(query, data_set_index)
            relevant_docs = list(dataset.qrels_dict()[qrel.query_id].keys())
            precision, recall = Evaluator.calculate_precision_recall(retrieved_docs, relevant_docs)
            avg_precision += precision
            avg_recall += recall
        avg_precision /= num_of_queries
        avg_recall /= num_of_queries
        return avg_precision, avg_recall
