import csv
from transformers import AutoTokenizer
import os


# parent class for dataset processor
class Data_Processor:
    def __init__(self, args):
        self.tokenizer = AutoTokenizer.from_pretrained(args.DATA.plm)
        self.data_dir = args.DATA.data_dir
        self.max_len = args.DATA.max_len  # the previously set max length of the input
        self.plm = args.DATA.plm

    def __str__(self):
        pattern = '''Data Configs: 
        data_dir: {} 
        max_len: {}'''
        return pattern.format(self.data_dir, self.max_len)

    def _get_pair_examples(self, file_dir):
        examples = []
        with open(file_dir, 'r', encoding='utf-8') as f:
            lines = csv.reader(f, delimiter='\t')
            next(lines)  # skip the headline
            for i, line in enumerate(lines):
                if 'semcor' in file_dir:
                    sent_id, target_id, target_loc, lemma, pos, sentence, label = line
                    domain = 0
                else:
                    target_loc, lemma, pos, sentence, label = line
                    domain = 1

                label = int(label)
                inputs = self.tokenizer(lemma, sentence)
                input_ids = inputs['input_ids']
                att_mask = inputs['attention_mask']
                if 'roberta' in self.plm:
                    len_input1 = len(self.tokenizer.tokenize(lemma, add_special_tokens=True))
                    len_input2 = len(input_ids) - len_input1
                    type_ids = len_input1 * [0] + len_input2 * [1]
                else:
                    type_ids = inputs['token_type_ids']

                cur_len = len(input_ids)
                if cur_len < self.max_len:
                    input_ids = input_ids + [self.tokenizer.pad_token_id] * (self.max_len - cur_len)
                    type_ids = type_ids + [0] * (self.max_len - cur_len)
                    att_mask = att_mask + [0] * (self.max_len - cur_len)
                else:
                    input_ids = input_ids[:self.max_len]
                    type_ids = type_ids[:self.max_len]
                    att_mask = att_mask[:self.max_len]

                examples.append([input_ids, type_ids, att_mask, label, domain])
                if (i + 1) % 10000 == 0:
                    print(f'{i + 1} sentences have been processed.')

            print(f'{file_dir} finished.')
        return examples


class WSD_Processor(Data_Processor):
    def __init__(self, args):
        super(WSD_Processor, self).__init__(args)
        self.threshold = args.DATA.wsd_threshold

    # use toy data for testing
    def get_wsd_train(self):
        data_path = os.path.join(self.data_dir, f'semcor/threshold/train_{self.threshold}.tsv')
        data = self._get_pair_examples(data_path)
        return data

    def get_wsd_test(self):
        data_path = os.path.join(self.data_dir, f'semcor/threshold/test_{self.threshold}.tsv')
        data = self._get_pair_examples(data_path)
        return data

    def get_wsd_toy(self):
        data_path = os.path.join(self.data_dir, f'semcor/toy.tsv')
        data = self._get_pair_examples(data_path)
        return data


class MD_Processor(Data_Processor):
    def __init__(self, args):
        super(MD_Processor, self).__init__(args)

    def get_all_train(self):
        data_path = os.path.join(self.data_dir, f'VUA_All/train.tsv')
        data = self._get_pair_examples(data_path)
        return data

    def get_all_val(self):
        data_path = os.path.join(self.data_dir, f'VUA_All/val.tsv')
        data = self._get_pair_examples(data_path)
        return data

    def get_all_test(self):
        data_path = os.path.join(self.data_dir, f'VUA_All/test.tsv')
        data = self._get_pair_examples(data_path)
        return data

    def get_verb_train(self):
        data_path = os.path.join(self.data_dir, f'VUA_Verb/train.tsv')
        data = self._get_pair_examples(data_path)
        return data

    def get_verb_val(self):
        data_path = os.path.join(self.data_dir, f'VUA_Verb/val.tsv')
        data = self._get_pair_examples(data_path)
        return data

    def get_verb_test(self):
        data_path = os.path.join(self.data_dir, f'VUA_Verb/test.tsv')
        data = self._get_pair_examples(data_path)
        return data

    def get_trofi(self):
        data_path = os.path.join(self.data_dir, f'TroFi/TroFi.tsv')
        data = self._get_pair_examples(data_path)
        return data

    def get_mohx(self):
        data_path = os.path.join(self.data_dir, f'MOH-X/MOH-X.tsv')
        data = self._get_pair_examples(data_path)
        return data

    def get_acad(self):
        data_path = os.path.join(self.data_dir, f'VUA_All/genre/acad.tsv')
        data = self._get_pair_examples(data_path)
        return data

    def get_conv(self):
        data_path = os.path.join(self.data_dir, f'VUA_All/genre/conv.tsv')
        data = self._get_pair_examples(data_path)
        return data

    def get_fict(self):
        data_path = os.path.join(self.data_dir, f'VUA_All/genre/fict.tsv')
        data = self._get_pair_examples(data_path)
        return data

    def get_news(self):
        data_path = os.path.join(self.data_dir, f'VUA_All/genre/news.tsv')
        data = self._get_pair_examples(data_path)
        return data

    def get_adj(self):
        data_path = os.path.join(self.data_dir, f'VUA_All/pos/adj.tsv')
        data = self._get_pair_examples(data_path)
        return data

    def get_adv(self):
        data_path = os.path.join(self.data_dir, f'VUA_All/pos/adv.tsv')
        data = self._get_pair_examples(data_path)
        return data

    def get_noun(self):
        data_path = os.path.join(self.data_dir, f'VUA_All/pos/noun.tsv')
        data = self._get_pair_examples(data_path)
        return data

    def get_verb(self):
        data_path = os.path.join(self.data_dir, f'VUA_All/pos/verb.tsv')
        data = self._get_pair_examples(data_path)
        return data


if __name__ == '__main__':
    from configs.default import get_config
    import argparse

    def parse_option():
        parser = argparse.ArgumentParser(description='Train on VUA ALL')
        parser.add_argument('--cfg', type=str, default='./configs/vua_all.yaml', metavar="FILE",
                            help='path to config file')
        parser.add_argument('--gpu', default='0', type=str, help='gpu device ids')
        parser.add_argument('--seed', default='4', type=int, help='random seed')
        parser.add_argument('--eval', action='store_true', help="evaluation only")
        parser.add_argument('--wsd_threshold', default=3, type=int)
        parser.add_argument('--log', default='log_trofi', type=str)
        args, unparsed = parser.parse_known_args()
        config = get_config(args)

        return config

    args = parse_option()
    processor = WSD_Processor(args)
    test_data = processor.get_wsd_toy()

    mp = MD_Processor(args)
    mohx_data = mp.get_mohx()
    for item in mohx_data:
        print(item)

