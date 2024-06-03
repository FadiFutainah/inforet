from sklearn.metrics.pairwise import cosine_similarity

from data_processor import init_nlp_dependencies, preprocess_data
from data_set import DataSet
from tf_idf_index import TfIdfIndex
from utils import SingletonMeta


class SearchEngine(metaclass=SingletonMeta):
    def __init__(self):
        print('initializing search engine...')
        self.results_size = 10
        init_nlp_dependencies()
        self.data_processor = preprocess_data
        self.data_sets = [DataSet('antique/train'), DataSet('wikir/en1k/training')]
        self.tfidf_indexes = [TfIdfIndex(data_set, self.data_processor) for data_set in self.data_sets]
        self.transformer_indexes = []
        print('search engine initialized successfully')

    def search(self, query, data_set_index):
        data_set = self.data_sets[data_set_index]
        vectorizer = self.tfidf_indexes[data_set_index].vectorizer
        tfidf_matrix = self.tfidf_indexes[data_set_index].tfidf_matrix
        normalized_query = self.data_processor(query)
        query_vector = vectorizer.transform([normalized_query])
        cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
        most_similar_docs_indices = cosine_similarities.argsort()[-self.results_size:][::-1]
        results = [0] * self.results_size
        docs_list = list(most_similar_docs_indices)
        for i, doc in enumerate(data_set.data.docs_iter()):
            if i in docs_list:
                results[docs_list.index(i)] = doc.doc_id
        return results
