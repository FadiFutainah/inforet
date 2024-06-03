import pickle


class FileManager:
    @staticmethod
    def get_path(name):
        if not name.endswith('.pkl'):
            name = f'{name}.pkl'
        return name

    @staticmethod
    def save_object(obj, name):
        path = FileManager.get_path(name)
        with open(path, 'wb') as file:
            pickle.dump(obj, file)

    @staticmethod
    def load_object(name):
        path = FileManager.get_path(name)
        with open(path, 'rb') as file:
            obj = pickle.load(file)
        return obj
