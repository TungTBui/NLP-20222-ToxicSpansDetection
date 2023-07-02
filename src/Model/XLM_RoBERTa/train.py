from simpletransformers.ner import NERModel, NERArgs
from sklearn.metrics import accuracy_score

def train_xlm_roberta(model_type, train_data, eval_data, tags, num_epochs, learning_rate, train_batch, eval_batch):

    num_examples = train_data.shape[0]  # replace with the size of your dataset
    batch_size = 8  # replace with your batch size
    N = 8  # replace with the number of epochs between each save

    steps_per_epoch = num_examples / batch_size
    save_steps = N * steps_per_epoch

    args = NERArgs()
    args.num_train_epochs = num_epochs 
    args.learning_rate = learning_rate

    args.save_steps = save_steps*2
    args.save_model_every_epoch	 = False
    args.save_eval_checkpoints = False
    args.overwrite_output_dir = True

    args.train_batch_size = train_batch
    args.eval_batch_size = eval_batch
    args.use_cached_eval_features = False
    args.use_multiprocessing = False
    args.reprocess_input_data = True

    args.use_early_stopping = True
    args.early_stopping_delta = 0.01
    args.early_stopping_metric = "mcc"
    args.early_stopping_metric_minimize = False
    args.early_stopping_patience = 5

    if model_type == "base":
        model = NERModel("xlmroberta", "xlm-roberta-base", labels=tags, args=args)
    if model_type == "large":
        model = NERModel("xlmroberta", "xlm-roberta-base", labels=tags, args=args)
    model.train_model(train_data, validation_data=eval_data, acc=accuracy_score)
