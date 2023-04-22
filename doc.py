import json
import os
import random
from typing import Optional, List

import numpy as np
import torch
import torch.utils.data.dataset

from config import args
from dict_hub import get_entity_dict, get_link_graph, get_tokenizer
from logger_config import logger
from triplet import reverse_triplet
from triplet_mask import construct_mask, construct_self_negative_mask

entity_dict = get_entity_dict()

if args.use_link_graph:
    # make the lazy data loading happen
    get_link_graph()


def _custom_tokenize(text: str,
                     text_pair: Optional[str] = None,
                     add_special_tokens: bool = True,
                     return_token_type_ids: bool = True) -> dict:
    tokenizer = get_tokenizer()
    encoded_inputs = tokenizer(text=text,
                               text_pair=text_pair if text_pair else None,
                               add_special_tokens=add_special_tokens,
                               max_length=args.max_num_tokens,
                               return_token_type_ids=return_token_type_ids,
                               truncation=True)
    return encoded_inputs


def _parse_entity_name(entity: str) -> str:
    if args.task.lower() == 'wn18rr':
        # family_alcidae_NN_1
        entity = ' '.join(entity.split('_')[:-2])
        return entity
    # a very small fraction of entities in wiki5m do not have name
    return entity or ''


def _concat_name_desc(entity: str, entity_desc: str) -> str:
    if entity_desc.startswith(entity):
        entity_desc = entity_desc[len(entity):].strip()
    if entity_desc:
        return '{}: {}'.format(entity, entity_desc)
    return entity


def get_neighbor_desc(head_id: str, tail_id: str = None) -> str:
    neighbor_ids = get_link_graph().get_neighbor_ids(head_id)
    # neighbor_ids = link_graph.get_neighbor_ids(head_id)
    # avoid label leakage during training
    if not args.is_test:
        neighbor_ids = [n_id for n_id in neighbor_ids if n_id != tail_id]
    entities = [entity_dict.get_entity_by_id(n_id).entity for n_id in neighbor_ids]
    entities = [_parse_entity_name(entity) for entity in entities]
    return ' '.join(entities)


class Example:

    def __init__(self, head_id, relation, tail_id=None, pred_gold_score=None, negatives=None, is_train=False, **kwargs):
        self.head_id = head_id
        self.tail_id = tail_id
        self.relation = relation
        self.pred_gold_score = pred_gold_score
        self.negatives = negatives

        if self.negatives is not None and args.task.lower() in ['wiki5m_trans'] and is_train:
            random.shuffle(negatives)
            self.negatives = negatives[:1]

    @property
    def head_desc(self):
        if not self.head_id:
            return ''
        return entity_dict.get_entity_by_id(self.head_id).entity_desc

    @property
    def tail_desc(self):
        if not self.tail_id:
            return ''
        return entity_dict.get_entity_by_id(self.tail_id).entity_desc

    @property
    def head(self):
        if not self.head_id:
            return ''
        return entity_dict.get_entity_by_id(self.head_id).entity

    @property
    def tail(self):
        if not self.tail_id:
            return ''
        return entity_dict.get_entity_by_id(self.tail_id).entity

    def vectorize(self, is_train=False, num_negatives=1) -> dict:
        head_desc, tail_desc = self.head_desc, self.tail_desc
        positive = self.tail_id
        if not args.is_test and args.use_hard_negative:
            negatives = self.negatives
            negative_ids = np.random.permutation(range(len(negatives))).tolist()[:num_negatives]
            negative_ids = negative_ids[:num_negatives]
            negatives = [negatives[negative_id] for negative_id in negative_ids]
            negative_descs = [entity_dict.get_entity_by_id(negative).entity_desc for negative in negatives]

        if args.use_link_graph:
            if len(head_desc.split()) < 20:
                head_desc += ' ' + get_neighbor_desc(head_id=self.head_id, tail_id=positive)
            if len(tail_desc.split()) < 20:
                tail_desc += ' ' + get_neighbor_desc(head_id=positive, tail_id=self.head_id)
            if not args.is_test and args.use_hard_negative:
                descs = []
                for negative, negative_desc in zip(negatives, negative_descs):
                    if len(negative_desc.split()) < 20:
                        negative_desc += ' ' + get_neighbor_desc(head_id=negative, tail_id=self.head_id)
                    descs.append(negative_desc)
                negative_descs = descs

        head_word = _parse_entity_name(self.head)
        head_text = _concat_name_desc(head_word, head_desc)
        hr_encoded_inputs = _custom_tokenize(text=head_text,
                                             text_pair=self.relation)

        head_encoded_inputs = _custom_tokenize(text=head_text)

        tail_word = _parse_entity_name(entity_dict.get_entity_by_id(positive).entity)
        tail_encoded_inputs = _custom_tokenize(text=_concat_name_desc(tail_word, tail_desc))

        negative_encoded_inputs = []
        if not args.is_test and args.use_hard_negative:
            for negative, negative_desc in zip(negatives, negative_descs):
                negative_word = _parse_entity_name(entity_dict.get_entity_by_id(negative).entity)
                negative_text = _concat_name_desc(negative_word, negative_desc)
                negative_encoded_inputs.append(_custom_tokenize(text=negative_text))

        return {'hr_token_ids': hr_encoded_inputs['input_ids'],
                'hr_token_type_ids': hr_encoded_inputs['token_type_ids'],
                'tail_token_ids': tail_encoded_inputs['input_ids'],
                'tail_token_type_ids': tail_encoded_inputs['token_type_ids'],
                'head_token_ids': head_encoded_inputs['input_ids'],
                'head_token_type_ids': head_encoded_inputs['token_type_ids'],
                'negative_token_ids': [inputs['input_ids'] for inputs in negative_encoded_inputs]
                if not args.is_test and args.use_hard_negative and len(negatives) > 0 else [],
                'negative_token_type_ids': [inputs['token_type_ids'] for inputs in negative_encoded_inputs]
                if not args.is_test and args.use_hard_negative and len(negatives) > 0 else [],
                'obj': self,
                'negative': [Example(None, None, negative, None) for negative in negatives]
                if not args.is_test and args.use_hard_negative and len(negatives) > 0 else [],
                'is_train': is_train}


class Dataset(torch.utils.data.dataset.Dataset):

    def __init__(self, path, task, examples=None, is_train=False, num_negatives=1, epoch=0):
        self.path_list = path.split(',')
        self.task = task
        assert all(os.path.exists(path) for path in self.path_list) or examples
        self.is_train = is_train
        self.num_negatives = num_negatives
        self.epoch = epoch
        if examples:
            self.examples = examples
        else:
            self.examples = []
            for path in self.path_list:
                if not self.examples:
                    if args.use_hard_negative:
                        self.examples = load_data(path, add_forward_triplet=True, add_backward_triplet=False,
                                                  is_train=is_train)
                    else:
                        self.examples = load_data(path)
                else:
                    if args.use_hard_negative:
                        self.examples.extend(load_data(path, add_forward_triplet=True, add_backward_triplet=False,
                                                       is_train=is_train))
                    else:
                        self.examples.extend(load_data(path))

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index].vectorize(self.is_train, self.num_negatives)


def load_data(path: str,
              add_forward_triplet: bool = True,
              add_backward_triplet: bool = True,
              is_train: bool = False,
              start: int = -1,
              end: int = -1) -> List[Example]:
    assert path.endswith('.json'), 'Unsupported format: {}'.format(path)
    assert add_forward_triplet or add_backward_triplet
    logger.info('In test mode: {}'.format(args.is_test))

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        if 0 <= start < end and end >= 0:
            data = data[start:end]
        cnt = len(data)
        examples = []
        from tqdm import tqdm
        for i in tqdm(range(cnt)):
            obj = data[i]
            if 'negatives' in obj and len(obj['negatives']) == 0:
                continue
            if add_forward_triplet:
                obj['is_train'] = is_train
                examples.append(Example(**obj))
            if add_backward_triplet:
                reversed_triplet = reverse_triplet(obj)
                reversed_triplet['is_train'] = is_train
                examples.append(Example(**reversed_triplet))
            data[i] = None

        logger.info('Load {} examples from {}'.format(len(examples), path))
        return examples


def collate(batch_data: List[dict]) -> dict:
    hr_token_ids, hr_mask = to_indices_and_mask(
        [ex['hr_token_ids'] for ex in batch_data], pad_token_id=get_tokenizer().pad_token_id)
    hr_token_type_ids = to_indices_and_mask(
        [ex['hr_token_type_ids'] for ex in batch_data], need_mask=False)

    tail_token_ids, tail_mask = to_indices_and_mask(
        [ex['tail_token_ids'] for ex in batch_data], pad_token_id=get_tokenizer().pad_token_id)
    tail_token_type_ids = to_indices_and_mask(
        [ex['tail_token_type_ids'] for ex in batch_data], need_mask=False)

    head_token_ids, head_mask = to_indices_and_mask(
        [ex['head_token_ids'] for ex in batch_data], pad_token_id=get_tokenizer().pad_token_id)
    head_token_type_ids = to_indices_and_mask(
        [ex['head_token_type_ids'] for ex in batch_data], need_mask=False)

    if not args.is_test and args.use_hard_negative:
        negative_token_ids, negative_mask = to_indices_and_mask(
            [negative_token_ids for ex in batch_data
             for negative_token_ids in ex['negative_token_ids']],
            pad_token_id=get_tokenizer().pad_token_id)
        negative_token_type_ids = to_indices_and_mask(
            [negative_token_type_ids for ex in batch_data
             for negative_token_type_ids in ex['negative_token_type_ids']],
            need_mask=False)

    batch_exs = [ex['obj'] for ex in batch_data]
    batch_negatives = [negative for ex in batch_data for negative in ex['negative']] \
        if not args.is_test and args.use_hard_negative else None
    batch_dict = {
        'hr_token_ids': hr_token_ids,
        'hr_mask': hr_mask,
        'hr_token_type_ids': hr_token_type_ids,
        'tail_token_ids': tail_token_ids,
        'tail_mask': tail_mask,
        'tail_token_type_ids': tail_token_type_ids,
        'head_token_ids': head_token_ids,
        'head_mask': head_mask,
        'head_token_type_ids': head_token_type_ids,
        'negative_token_ids': negative_token_ids if not args.is_test
                                                    and args.use_hard_negative else None,
        'negative_mask': negative_mask if not args.is_test and args.use_hard_negative else None,
        'negative_token_type_ids': negative_token_type_ids if not args.is_test and args.use_hard_negative else None,
        'batch_data': batch_exs,
        'triplet_mask': construct_mask(row_exs=batch_exs) if not args.is_test else None,
        'triplet_negative_mask': construct_mask(batch_exs, col_exs=batch_negatives) if not args.is_test
                                                                                       and args.use_hard_negative
        else None,
        'self_negative_mask': construct_self_negative_mask(batch_exs) if not args.is_test else None,
    }

    return batch_dict


def to_indices_and_mask(batch_tensor, pad_token_id=0, need_mask=True):
    max_len = max([len(t) for t in batch_tensor])
    indices = []
    if need_mask:
        mask = []
    for t in batch_tensor:
        length = len(t)
        padding_len = max_len - length
        indices.append(t + padding_len * [pad_token_id])
        if need_mask:
            mask.append([1] * length + [0] * padding_len)
    indices = torch.tensor(indices, dtype=torch.int64)

    if need_mask:
        mask = torch.tensor(mask, dtype=torch.int64)
        return indices, mask
    else:
        return indices
