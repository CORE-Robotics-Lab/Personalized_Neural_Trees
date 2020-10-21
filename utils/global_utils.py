
import pickle
import os



def save_pickle(file_location, file, special_string, want_to_print=False):
    """
    stores information into a pickle file
    :param file_location: location to store
    :param file: data
    :param special_string: string to save the data with
    :param want_to_print: want to print a little debug statement
    :return:
    """
    pickle.dump(file, open(os.path.join(file_location, special_string), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    if want_to_print:
        print("Dumped", file, "into ", file_location, "safely!")


def load_in_embedding(neural_net, embedding_list, player_id):
    """
    load in an embedding from a list of embeddings into a bddt
    :param neural_net: Neural Network model
    :param embedding_list: list of embeddings
    :param player_id:  id within list
    :return:
    """
    curr_embedding = embedding_list[player_id]
    curr_dict = neural_net.state_dict()
    curr_dict['bayesian_embedding'] = curr_embedding
    neural_net.load_state_dict(curr_dict)


def store_embedding_back(neural_net, embedding_list, player_id):
    """
    store embedding back to the list
    :param neural_net: Neural Network model
    :param embedding_list:  list of embeddings
    :param player_id: id within list
    :return:
    """
    curr_dict = neural_net.state_dict()
    new_embedding = curr_dict['bayesian_embedding'].clone()
    # curr_embedding = embedding_list[player_id]
    embedding_list[player_id] = new_embedding
    return embedding_list

def load_in_embedding_bnn(NeuralNet, embedding_list, player_id):
    """

    load in an embedding from a list of embeddings into a bnn
    :param NeuralNet: Neural Network model
    :param embedding_list: list of embeddings
    :param player_id:  id within list
    :return:
    """
    curr_embedding = embedding_list[player_id]
    curr_dict = NeuralNet.state_dict()
    curr_dict['EmbeddingList.0.embedding'] = curr_embedding
    NeuralNet.load_state_dict(curr_dict)


def store_embedding_back_bnn(NeuralNet, embedding_list, player_id, DEBUG=False):
    curr_dict = NeuralNet.state_dict()
    new_embedding = curr_dict['EmbeddingList.0.embedding'].clone()
    curr_embedding = embedding_list[player_id]
    if DEBUG:
        print(curr_embedding)
        print(new_embedding)
    embedding_list[player_id] = new_embedding
    return embedding_list
