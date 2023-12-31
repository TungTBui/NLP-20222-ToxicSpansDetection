import argparse
import ast

from Model.BiLSTM_CRF.utils import *
from Model.BiLSTM_CRF.pre_processing import *


class WordsTagger:
    def __init__(self, model_dir, device=None):
        args_ = load_json_file(arguments_filepath(model_dir))
        args = argparse.Namespace(**args_)
        args.model_dir = model_dir
        self.args = args

        self.preprocessor = Preprocessor(config_dir=model_dir, verbose=False)
        self.model = build_model(self.preprocessor, verbose=False, model_path=model_filepath(model_dir))
        self.device = running_device(device)
        self.model.to(self.device)

        self.model.eval()

    def __call__(self, sentences, begin_tags="BS"):
        """predict texts

        :param sentences: a text or a list of text
        :param begin_tags: begin tags for the beginning of a span
        :return:
        """
        if not isinstance(sentences, (list, tuple)):
            raise ValueError("sentences must be a list of sentence")

        try:
            sent_tensor = np.asarray([self.preprocessor.sent_to_vector(s) for s in sentences])
            sent_tensor = torch.from_numpy(sent_tensor).to(self.device)
            with torch.no_grad():
                _, tags = self.model(sent_tensor)
            tags = self.preprocessor.decode_tags(tags)
        except RuntimeError as e:
            print("*** runtime error: {}".format(e))
            raise e
        return tags, self.tokens_from_tags(sentences, tags, begin_tags=begin_tags)

    @staticmethod
    def tokens_from_tags(sentences, tags_list, begin_tags):
        """extract entities from tags

        :param sentences: a list of sentence
        :param tags_list: a list of tags
        :param begin_tags:
        :return:
        """
        if not tags_list:
            return []

        def _tokens(sentence, ts):
            # begins: [(idx, label), ...]
            all_begin_tags = begin_tags + "O"
            begins = [(idx, t[2:]) for idx, t in enumerate(ts) if t[0] in all_begin_tags]
            begins = [
                         (idx, label)
                         for idx, label in begins
                         if ts[idx] != "O" or (idx > 0 and ts[idx - 1] != "O")
                     ] + [(len(ts), "")]

            tokens_ = [(sentence[s:e], label) for (s, label), (e, _) in zip(begins[:-1], begins[1:]) if label]
            return [((t, tag) if tag else t) for t, tag in tokens_]

        tokens_list = [_tokens(sentence, ts) for sentence, ts in zip(sentences, tags_list)]
        return tokens_list


def predict(model_dir, sentence):
    device = 0

    results = WordsTagger(model_dir, device)([sentence])
    result_1 = json.dumps(results[0][0], ensure_ascii=False)
    result_2 = json.dumps(results[1][0], ensure_ascii=False)
    bio = ast.literal_eval(result_1)
    tokens = sentence.split(" ")
    bio = ast.literal_eval(result_1)
    toxic_spans = []
    toxic = []
    i = 0
    while i < len(bio):
        if bio[i] == "B-T":
          toxic = [tokens[i]]
          for j in range(i+1, len(bio)):
            if bio[j] != "I-T":
              i = j - 1
              break
            toxic.append(tokens[j])
          toxic_spans.append(" ".join(toxic))
        i += 1
    return toxic_spans


