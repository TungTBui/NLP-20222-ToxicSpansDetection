from os import mkdir
import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from Model.BiLSTM_CRF.utils import *
from Model.BiLSTM_CRF.pre_processing import *


def __eval_model(model, device, dataloader, desc):
    model.eval()
    with torch.no_grad():
        # eval
        losses, nums = zip(*[
            (model.loss(xb.to(device), yb.to(device)), len(xb))
            for xb, yb in tqdm(dataloader, desc=desc)])
        return np.sum(np.multiply(losses, nums)) / np.sum(nums)


def __save_loss(losses, file_path):
    pd.DataFrame(data=losses, columns=["epoch", "batch", "train_loss", "val_loss"]).to_csv(file_path, index=False)


def __save_model(model_dir, model):
    model_path = model_filepath(model_dir)
    torch.save(model.state_dict(), model_path)
    print("save model => {}".format(model_path))


def train(model_dir="/content/Model", batch_size=8, lr=1e-3, weight_decay=0., epochs=10, hidden_dim=128):
    model_dir = model_dir
    corpus_dir = "sample_corpus"
    val_por = 0.1
    test_por = 0.1
    max_seq_len = 100
    batch_size = batch_size
    lr = lr
    weight_decay = weight_decay
    device = "cpu"
    epochs = epochs
    recovery = False
    hidden_dim = hidden_dim

    var = {
    "corpus_dir": corpus_dir,
    "--model_dir": model_dir,
    "--num_epoch": epochs,
    "--lr": lr,
    "--weight_decay": weight_decay,
    "--batch_size": batch_size,
    "--device": device,
    "--max_seq_len": max_seq_len,
    "--val_split": val_por,
    "--test_split": test_por,
    "--recovery": recovery,
    "--save_best_val_model": True,
    "--embedding_dim": 100,
    "--hidden_dim": hidden_dim,
    "--num_rnn_layers": 1,
    "--rnn_type": "lstm"
    }

    if not exists(model_dir):
        mkdir(model_dir)
    save_json_file(var, arguments_filepath(model_dir))

    preprocessor = Preprocessor(config_dir=corpus_dir, save_config_dir=model_dir, verbose=True)
    model = build_model(preprocessor, load=recovery, verbose=True)

    # loss
    loss_path = join(model_dir, "loss.csv")
    losses = pd.read_csv(loss_path).values.tolist() if recovery and exists(loss_path) else []

    # datasets
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = preprocessor.load_dataset(
        corpus_dir, val_por, test_por, max_seq_len)
    train_dl = DataLoader(TensorDataset(x_train, y_train), batch_size, shuffle=True)
    valid_dl = DataLoader(TensorDataset(x_val, y_val), batch_size * 2)
    test_dl = DataLoader(TensorDataset(x_test, y_test), batch_size * 2)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.)

    device = running_device(device)
    model.to(device)

    val_loss = 0
    best_val_loss = 1e4
    for epoch in range(epochs):
        # train
        model.train()
        bar = tqdm(train_dl)
        for bi, (xb, yb) in enumerate(bar):
            model.zero_grad()

            loss = model.loss(xb.to(device), yb.to(device))
            loss.backward()
            optimizer.step()
            bar.set_description("{:2d}/{} loss: {:5.2f}, val_loss: {:5.2f}".format(
                epoch+1, epochs, loss, val_loss))
            losses.append([epoch, bi, loss.item(), np.nan])

        # evaluation
        val_loss = __eval_model(model, device, dataloader=valid_dl, desc="eval").item()
        # save losses
        losses[-1][-1] = val_loss
        __save_loss(losses, loss_path)

        # save model
        if not True or val_loss < best_val_loss:
            best_val_loss = val_loss
            __save_model(model_dir, model)
            print("save model(epoch: {}) => {}".format(epoch, loss_path))

    # test
    test_loss = __eval_model(model, device, dataloader=test_dl, desc="test").item()
    last_loss = losses[-1][:]
    last_loss[-1] = test_loss
    losses.append(last_loss)
    __save_loss(losses, loss_path)
    print("training completed. test loss: {:.2f}".format(test_loss))

