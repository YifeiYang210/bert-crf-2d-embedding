
import logging
import os
import sys
import torch
import pickle

from torch.utils.data import TensorDataset
from tqdm import tqdm

logger = logging.getLogger(__name__)

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, label=None, boxes=None, box_pos_ids=None, box_nums=None):
        self.guid = guid
        self.text = text 
        self.label = label
        self.boxes = boxes
        self.box_pos_ids = box_pos_ids
        self.box_nums = box_nums

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, ori_tokens, boxes=None, box_pos_ids=None, box_nums=None):

        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.ori_tokens = ori_tokens
        self.boxes = boxes
        self.box_pos_ids = box_pos_ids
        self.box_nums = box_nums


class NerProcessor(object):
    def read_examples_from_file(self, data_dir, mode):
        # file_path = os.path.join(data_dir, "{}.txt".format(mode))
        # TODO
        file_path = os.path.join(data_dir)
        guid_index = 1
        examples = []
        with open(file_path, encoding="utf-8") as f:
            words = []
            labels = []
            boxes = []
            box_pos_ids = []
            box_nums = []
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if words:
                        examples.append(InputExample(guid="{}-{}".format(mode, guid_index),
                                                     text=words,
                                                     label=labels,
                                                     boxes=boxes,
                                                     box_pos_ids=box_pos_ids,
                                                     box_nums=box_nums,
                                                     ))
                        guid_index += 1
                        words = []
                        labels = []
                        boxes = []
                        box_pos_ids = []
                        box_nums = []
                else:
                    splits = line.split("\t")
                    words.append(splits[0])
                    if len(splits) > 1:
                        # labels.append(splits[-1].replace("\n", ""))
                        labels.append(splits[1])
                        box = splits[3]
                        box = [abs(int(b)) for b in box.split()]
                        boxes.append(box)
                        box_pos_ids.append(int(splits[4]))
                        box_nums.append(int(splits[5].replace("\n", "")))
                    else:
                        # Examples could have no label for mode = "test"
                        labels.append("O")
            if words:
                examples.append(InputExample(guid="%s-%d".format(mode, guid_index),
                                             words=words,
                                             labels=labels,
                                             boxes=boxes,
                                             box_pos_ids=box_pos_ids,
                                             box_nums=box_nums))
        return examples


def convert_examples_to_features(examples1,
                                 label_list,
                                 max_seq_length,
                                 tokenizer,
                                 cls_token_at_end=False,
                                 cls_token="[CLS]",
                                 cls_token_segment_id=1,
                                 sep_token="[SEP]",
                                 sep_token_extra=False,
                                 pad_on_left=False,
                                 pad_token=0,
                                 cls_token_box=[0, 0, 0, 0],
                                 sep_token_box=[1000, 1000, 1000, 1000],
                                 pad_token_box=[0, 0, 0, 0],
                                 cls_token_box_pos_ids=0,
                                 sep_token_box_pos_ids=0,
                                 pad_token_box_pos_ids=0,
                                 cls_token_box_nums=0,
                                 sep_token_box_nums=0,
                                 pad_token_box_nums=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 mask_padding_with_zero=True):

    label_map = {label: i for i, label in enumerate(label_list)}
    print(type(examples1))
    features = []

    for (ex_index, example) in enumerate(examples1):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples1)))

        tokens = []
        labels = []
        ori_tokens = []
        token_boxes = []
        token_box_pos_ids = []
        token_box_nums = []

        for word, label, box, box_pos_id, box_num in zip(example.text, example.label, example.boxes,
                                                         example.box_pos_ids,
                                                         example.box_nums):
            # 防止wordPiece情况出现，不过貌似不会
            token = tokenizer.tokenize(word)
            tokens.extend(token)
            ori_tokens.append(word)
            token_boxes.extend([box] * len(token))
            token_box_pos_ids.extend([box_pos_id] * len(token))
            token_box_nums.extend([box_num] * len(token))

            # 单个字符不会出现wordPiece
            for m in range(len(token)):
                if m == 0:
                    labels.append(label_map[label])
                else:
                    if label == "O":
                        labels.append(label_map["O"])
                    else:
                        labels.append(label_map[label])
            
        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 2)]  # -2 的原因是因为序列需要加一个句首和句尾标志
            labels = labels[0:(max_seq_length - 2)]
            token_boxes = token_boxes[: (max_seq_length - 2)]
            token_box_pos_ids = token_box_pos_ids[: (max_seq_length - 2)]
            token_box_nums = token_box_nums[: (max_seq_length - 2)]
            ori_tokens = ori_tokens[0:(max_seq_length - 2)]

        ori_tokens = [cls_token] + ori_tokens + [sep_token]

        segment_ids = [sequence_a_segment_id] * len(tokens)
        tokens = [cls_token] + tokens
        labels = [label_map["O"]] + labels
        token_boxes = [cls_token_box] + token_boxes
        token_box_pos_ids = [cls_token_box_pos_ids] + token_box_pos_ids
        token_box_nums = [cls_token_box_nums] + token_box_nums

        tokens += [sep_token]
        labels += [label_map["O"]]
        segment_ids = [0] + segment_ids + [0]
        token_boxes += [sep_token_box]
        token_box_pos_ids += [sep_token_box_pos_ids]
        token_box_nums += [sep_token_box_nums]

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        assert len(ori_tokens) == len(tokens), f"{len(ori_tokens)}, {len(tokens)}, {ori_tokens}"
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            # we don't concerned about it!
            labels.append(0)
            tokens.append("**NULL**")
            token_boxes.append([0, 0, 0, 0])
            token_box_pos_ids.append(0)
            token_box_nums.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(labels) == max_seq_length
        assert len(token_boxes) == max_seq_length
        assert len(token_box_pos_ids) == max_seq_length, f"{token_box_pos_ids}, {len(token_box_pos_ids)}"
        assert len(token_box_nums) == max_seq_length

        if ex_index < 1:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("ori_tokens: %s" % " ".join([str(x) for x in ori_tokens]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label_ids: %s" % " ".join([str(x) for x in labels]))
            logger.info("boxes: %s", " ".join([str(x) for x in token_boxes]))
            logger.info("box_pos_ids: %s", " ".join([str(x) for x in token_box_pos_ids]))
            logger.info("box_nums: %s", " ".join([str(x) for x in token_box_nums]))

        # if not os.path.exists(os.path.join(output_dir, 'label2id.pkl')):
        #     with open(os.path.join(output_dir, 'label2id.pkl'), 'wb') as w:
        #         pickle.dump(label_map, w)

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=labels,
                              ori_tokens=ori_tokens,
                              boxes=token_boxes,
                              box_pos_ids=token_box_pos_ids,
                              box_nums=token_box_nums,
                              ))

    return features


def get_labels(path):
    if path:
        with open(path, "r") as f:
            labels = f.read().splitlines()
        if "O" not in labels:
            labels = ["O"] + labels
        return labels
    else:
        return ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]


def get_Dataset(args, processor, tokenizer, label_list, mode="train"):
    if mode == "train":
        filepath = args.train_file
    elif mode == "eval":
        filepath = args.eval_file
    elif mode == "test":
        filepath = args.test_file
    else:
        raise ValueError("mode must be one of train, eval, or test")

    examples1 = processor.read_examples_from_file(filepath, mode)
    print(type(examples1))
    features = convert_examples_to_features(
        examples1=examples1, label_list=label_list, max_seq_length=args.max_seq_length, tokenizer=tokenizer
    )

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    all_bboxes = torch.tensor([f.boxes for f in features], dtype=torch.long)
    all_bbox_pos_ids = torch.tensor([f.box_pos_ids for f in features], dtype=torch.long)
    all_bbox_nums = torch.tensor([f.box_nums for f in features], dtype=torch.long)

    data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_bboxes, all_bbox_pos_ids, all_bbox_nums)

    return examples1, features, data

