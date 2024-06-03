import os.path

from sklearn.feature_extraction.text import TfidfVectorizer

from file_manager import FileManager


class TfIdfIndex:
    def __init__(self, data_set, preprocess_function):
        print(f'initializing tf-idf index on dataset {data_set.name}...')
        self.data_set = data_set
        self.preprocess_function = preprocess_function
        self.vectorizer_path = f'data/vectorizer-{self.data_set.file_name}.pkl'
        self.tfidf_matrix_path = f'data/tfidf_matrix-{self.data_set.file_name}.pkl'
        self.vectorizer, self.tfidf_matrix = self._initialize_index()
        print(f'tf-idf index created successfully on {self.data_set.name}')

    def _initialize_index(self):
        vectorizer, tfidf_matrix = None, None
        if os.path.exists(self.vectorizer_path):
            print('loading vectorizer...')
            vectorizer = FileManager.load_object(self.vectorizer_path)
            print('vectorizer loaded successfully')
        if os.path.exists(self.tfidf_matrix_path):
            print('loading tfidf_matrix...')
            tfidf_matrix = FileManager.load_object(self.tfidf_matrix_path)
            print('tfidf_matrix loaded successfully')
        if vectorizer is None or tfidf_matrix is None:
            print('creating tf-idf index...')
            vectorizer, tfidf_matrix = self.create_index()
            print('tf-idf index created successfully')
            self.save(vectorizer, tfidf_matrix)
            print('tf-idf index saved successfully')
        return vectorizer, tfidf_matrix

    def create_index(self):
        vectorizer = TfidfVectorizer(preprocessor=self.preprocess_function)
        data_frame = self.data_set.data_frame.dropna(subset=['doc'])
        documents = data_frame['doc']
        tfidf_matrix = vectorizer.fit_transform(documents)
        return vectorizer, tfidf_matrix

    def save(self, tfidf_matrix, vectorizer):
        FileManager.save_object(tfidf_matrix, self.tfidf_matrix_path)
        FileManager.save_object(vectorizer, self.vectorizer_path)
