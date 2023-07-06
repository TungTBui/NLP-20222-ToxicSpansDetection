import pandas as pd
import numpy as np
from ast import literal_eval
from sklearn.metrics import precision_recall_fscore_support


def evaluate(model, test, test_word_tag):

    def IBO_pred(tokens):
        ibo_list = []
        value_list = []
        pred = model.predict(tokens)
        for i in range(len(pred[0])):
            ibo_pred = pred[0][i][0]
            value_pred = list(ibo_pred.keys())
            ibo_pred = list(ibo_pred.values())
            ibo_list.append(ibo_pred[0])
            value_list.append(value_pred[0])
        return value_list, ibo_list

    def IBO_pred_test(test):
        list_ibo = []
        list_value = []
        df_pred = pd.DataFrame()
        for i in range(len(test)):
            text = test['content'][i]
            tokens = text.split()
            ibo_, t = IBO_pred(tokens)
            list_ibo.append(ibo_)
            list_value.append(t)
        df_pred['Value_pred'] = list_value
        df_pred['IBO_pred'] = list_ibo
        return df_pred

    def tokenize(text, pos):
        tokens = text.split()
        alignment = []
        start = 0
        for t in tokens:
            res = text.find(t, start)
            alignment.append(pos[res:res + len(t)])
            start = res + len(t)
        assert len(tokens) == len(alignment)
        return tokens, alignment

    def y_pred(data, df_predict):
        index_pred = []
        for i in range(len(df_predict)):
            value_predict_i = df_predict['Value_pred'][i]

            text = data['content'][i]
            pos = [i for i in range(len(text))]

            tokens, alignment = tokenize(text, pos)
            df_point = pd.DataFrame()

            df_point['spans'] = [0 for i in range(len(text))]

            for i, token in enumerate(value_predict_i):

                if token == 'B-T' or token == 'I-T':
                    for ali in alignment[i]:
                        df_point['spans'][ali] = 1
            index_pred.append(list(df_point['spans']))
        return index_pred

    def y_true(data):
        index_true = []

        for i in range(len(data)):
            text = data['content'][i]
            pos = [i for i in range(len(text))]
            df_point = pd.DataFrame()
            df_point['spans'] = pos
            df_point['spans'] = 0

            if not data['index_spans'][i]:
                index_true.append(list(df_point['spans']))
            else:
                for j in data['index_spans'][i]:
                    df_point['spans'][j] = 1
                index_true.append(list(df_point['spans']))
        return index_true

    # Main evaluate function logic starts here

    result, model_outputs, wrong_predictions = model.eval_model(test_word_tag)

    test['index_spans'] = test['index_spans'].apply(literal_eval)

    df_ibo = IBO_pred_test(test)

    true = y_true(test)
    pred = y_pred(test, df_ibo)

    # Dataframe for saving evaluation metrics
    scores_f1_macro = []
    scores_f1_micro = []
    scores_precision_macro = []
    scores_precision_micro = []
    scores_recall_macro = []
    scores_recall_micro = []

    for i in range(len(true)):
        score_macro = precision_recall_fscore_support(true[i], pred[i], average='macro', zero_division=0)
        score_micro = precision_recall_fscore_support(true[i], pred[i], average='micro', zero_division=0)

        scores_f1_macro.append(score_macro[2])
        scores_f1_micro.append(score_micro[2])
        scores_precision_macro.append(score_macro[0])
        scores_precision_micro.append(score_micro[0])
        scores_recall_macro.append(score_macro[1])
        scores_recall_micro.append(score_micro[1])

    scores = pd.DataFrame()
    scores['eval_loss'] = [list(result.values())[0]]
    scores['F1_ner'] = [list(result.values())[1]]
    scores['F1-micro'] = [np.mean(scores_f1_micro)]
    scores['F1-macro'] = [np.mean(scores_f1_macro)]
    scores['Precision-macro'] = [np.mean(scores_precision_macro)]
    scores['Precision-micro'] = [np.mean(scores_precision_micro)]
    scores['Recall-macro'] = [np.mean(scores_recall_macro)]
    scores['Recall-micro'] = [np.mean(scores_recall_micro)]

    return scores

# Example usage:
# scores = evaluate(model, test, test_word_tag)
