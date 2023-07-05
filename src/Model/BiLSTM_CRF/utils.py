import json
from os.path import exists, join
import torch
from Model.BiLSTM_CRF.model import BiRnnCrf

START_TAG = "<START>"
STOP_TAG = "<STOP>"

PAD = "<PAD>"
OOV = "<OOV>"

FILE_ARGUMENTS = "arguments.json"
FILE_MODEL = "model.pth"


def save_json_file(obj, file_path):
    with open(file_path, "w", encoding="utf8") as f:
        f.write(json.dumps(obj, ensure_ascii=False))


def load_json_file(file_path):
    with open(file_path, encoding="utf8") as f:
        return json.load(f)


def arguments_filepath(model_dir):
    return join(model_dir, FILE_ARGUMENTS)


def model_filepath(model_dir):
    return join(model_dir, FILE_MODEL)


def build_model(processor, load=True, verbose=False):
    model = BiRnnCrf(len(processor.vocab), len(processor.tags),
                     embedding_dim=100, hidden_dim=128, num_rnn_layers=1)

    # weights
    model_path = model_filepath("/content/Model")
    if exists(model_path) and load:
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
        if verbose:
            print("load model weights from {}".format(model_path))
    return model


def running_device(device):
    return device if device else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
