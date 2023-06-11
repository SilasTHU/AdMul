from torch.utils.data import Dataset, Sampler, RandomSampler
import torch
import math


class dataset(Dataset):
    """wrap in PyTorch Dataset"""

    def __init__(self, examples):
        super(dataset, self).__init__()
        self.examples = examples

    def __getitem__(self, idx):
        return self.examples[idx]

    def __len__(self):
        return len(self.examples)


class BalanceSampler(Sampler):
    """
    a self-defined dataset sampler. It can make sure that two dataset are sampled by their size ratio.

    dataset: (pytorch concat dataset) a concat dataset of 2 separate datasets
    batch_size: (int) the batch size for the concat dataset

    """

    def __init__(self, dataset, batch_size):
        super(BalanceSampler, self).__init__(dataset)
        self.dataset = dataset
        self.batch_size = batch_size
        self.dataset0_size = len(dataset.datasets[0])
        self.dataset1_size = len(dataset.datasets[1])
        if self.dataset0_size != self.dataset1_size:
            raise ValueError("two datasets must have equal samples.")
        self.number_of_datasets = len(dataset.datasets)

    def __len__(self):
        return self.dataset0_size + self.dataset1_size

    def __iter__(self):
        samplers_list = []
        sampler_iterators = []
        for dataset_idx in range(self.number_of_datasets):
            cur_dataset = self.dataset.datasets[dataset_idx]
            sampler = RandomSampler(cur_dataset)
            samplers_list.append(sampler)
            cur_sampler_iterator = sampler.__iter__()
            sampler_iterators.append(cur_sampler_iterator)
            # for item in sampler:
            #     print(item)

        sample_cnt_0 = self.batch_size // 2
        sample_cnt_1 = self.batch_size // 2
        sample_cnts = [sample_cnt_0, sample_cnt_1]
        batch_cnt = math.ceil((self.dataset0_size + self.dataset1_size) / self.batch_size)  # 一共需要迭代多少个batch

        push_index_val = [0] + self.dataset.cumulative_sizes[:-1]
        sample_batch_indices = []  # 最后存储的索引

        for batch_idx in range(batch_cnt):
            cur_samples = []
            if batch_idx != batch_cnt - 1:  # 每一个batch按比例采样batch_size个样本
                for dataset_idx, sample_cnt in zip(range(self.number_of_datasets), sample_cnts):
                    for i in range(sample_cnt):
                        try:
                            cur_sample = sampler_iterators[dataset_idx].__next__()
                            cur_sample = cur_sample + push_index_val[dataset_idx]  # 原来的下标和合并后数据集的下标不一致
                            cur_samples.append(cur_sample)
                        except StopIteration:
                            sampler_iterator = samplers_list[dataset_idx].__iter__()
                            cur_sample_org = sampler_iterator.__next__()
                            cur_sample = cur_sample_org + push_index_val[dataset_idx]
                            cur_samples.append(cur_sample)
            else:  # 最后一个batch直接迭代到迭代器的末尾
                for dataset_idx in range(self.number_of_datasets):
                    while True:
                        try:
                            cur_sample = sampler_iterators[dataset_idx].__next__()
                            cur_sample = cur_sample + push_index_val[dataset_idx]  # 原来的下标和合并后数据集的下标不一致
                            cur_samples.append(cur_sample)
                        except StopIteration:
                            break
            sample_batch_indices.extend(cur_samples)
        return iter(sample_batch_indices)


def collate_fn(examples):
    input_ids, type_ids, att_mask, labels, domains = map(list, zip(*examples))

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    type_ids = torch.tensor(type_ids, dtype=torch.long)
    att_mask = torch.tensor(att_mask, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)
    domains = torch.tensor(domains, dtype=torch.long)

    return input_ids, type_ids, att_mask, labels, domains
