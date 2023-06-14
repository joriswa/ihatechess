import pickle


def save_game_data(game_data, file_path):
    with open(file_path, "wb") as file:
        pickle.dump(game_data, file)


def load_game_data(file_path):
    with open(file_path, "rb") as file:
        obj = pickle.load(file)
    return obj
