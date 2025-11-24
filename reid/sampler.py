# reid/sampler.py
import random
from torch.utils.data import Sampler
from collections import defaultdict

class PKSampler(Sampler):
    """
    P-K sampler: each batch contains P identities, K images per identity.
    samples: list of tuples (path, pid)
    batch_size = P * K
    """
    def __init__(self, data_source, batch_size, P, K):
        self.data_source = data_source  # dataset.samples or list of (path, pid)
        self.batch_size = batch_size
        self.P = P
        self.K = K
        # build pid->indices
        self.index_dic = {}
        for idx, (_, pid) in enumerate(data_source):
            self.index_dic.setdefault(pid, []).append(idx)
        self.pids = list(self.index_dic.keys())
        # ensure each pid has enough samples (if not, we will sample with replacement)
        self.length = 0
        for pid in self.pids:
            n = len(self.index_dic[pid])
            self.length += max(n, K)

    def __iter__(self):
        # generate batches
        pids = self.pids.copy()
        random.shuffle(pids)
        batch = []
        for pid in pids:
            idxs = self.index_dic[pid]
            if len(idxs) >= self.K:
                selected = random.sample(idxs, self.K)
            else:
                selected = random.choices(idxs, k=self.K)
            batch.extend(selected)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        # if leftover, yield (pad with random)
        if len(batch) > 0:
            while len(batch) < self.batch_size:
                pid = random.choice(self.pids)
                idxs = self.index_dic[pid]
                batch.append(random.choice(idxs))
            yield batch

    def __len__(self):
        return self.length // self.batch_size
