import torch
import types
from typing import List, Tuple, Optional

from transformers import BertTokenizerFast, BertTokenizer
from colbert.modeling.tokenization.utils import _split_into_batches

def truncate_sequences_from_beginning(
    self,
    ids: List[int],
    pair_ids: Optional[List[int]] = None,
    num_tokens_to_remove: int = 0,
    truncation_strategy = "longest_first",
    stride: int = 0,
) -> Tuple[List[int], List[int], List[int]]:
    """ Only truncates single sequences (doesn't work for pairs) and truncates from the beginning.
    Ignores pair_ids, truncation_strategy"""

    if num_tokens_to_remove <= 0:
        return ids, pair_ids, []

    overflowing_tokens = []
    for _ in range(num_tokens_to_remove):
        if not overflowing_tokens:
            window_len = min(len(ids), stride + 1)
        else:
            window_len = 1
        overflowing_tokens.extend(ids[:window_len])
        ids = ids[1:]

    return (ids, pair_ids, overflowing_tokens)


class QueryTokenizer():
    def __init__(self, query_maxlen, truncate_from_start=False):
        self.tok = BertTokenizer.from_pretrained('bert-base-uncased')

        # Monkey patch tokenizers's truncate sequences function to truncate sequences from the beginning
        if truncate_from_start:
            self.tok.truncate_sequences = types.MethodType(truncate_sequences_from_beginning, self.tok)
            print("Monkey patched")
        self.query_maxlen = query_maxlen

        self.Q_marker_token, self.Q_marker_token_id = '[Q]', self.tok.get_vocab()['[unused0]']
        self.cls_token, self.cls_token_id = self.tok.cls_token, self.tok.cls_token_id
        self.sep_token, self.sep_token_id = self.tok.sep_token, self.tok.sep_token_id
        self.mask_token, self.mask_token_id = self.tok.mask_token, self.tok.mask_token_id

        assert self.Q_marker_token_id == 1 and self.mask_token_id == 103

    def tokenize(self, batch_text, add_special_tokens=False):
        assert type(batch_text) in [list, tuple], (type(batch_text))

        tokens = [self.tok.tokenize(x, add_special_tokens=False) for x in batch_text]

        if not add_special_tokens:
            return tokens

        prefix, suffix = [self.cls_token, self.Q_marker_token], [self.sep_token]
        tokens = [prefix + lst + suffix + [self.mask_token] * (self.query_maxlen - (len(lst)+3)) for lst in tokens]

        return tokens

    def encode(self, batch_text, add_special_tokens=False):
        assert type(batch_text) in [list, tuple], (type(batch_text))

        ids = self.tok(batch_text, add_special_tokens=False)['input_ids']

        if not add_special_tokens:
            return ids

        prefix, suffix = [self.cls_token_id, self.Q_marker_token_id], [self.sep_token_id]
        ids = [prefix + lst + suffix + [self.mask_token_id] * (self.query_maxlen - (len(lst)+3)) for lst in ids]

        return ids

    def tensorize(self, batch_text, bsize=None):
        assert type(batch_text) in [list, tuple], (type(batch_text))

        # add placehold for the [Q] marker
        batch_text = ['. ' + x for x in batch_text]

        obj = self.tok(batch_text, padding='max_length', truncation=True,
                       return_tensors='pt', max_length=self.query_maxlen)

        ids, mask = obj['input_ids'], obj['attention_mask']

        # postprocess for the [Q] marker and the [MASK] augmentation
        ids[:, 1] = self.Q_marker_token_id
        ids[ids == 0] = self.mask_token_id

        if bsize:
            batches = _split_into_batches(ids, mask, bsize)
            return batches

        return ids, mask
