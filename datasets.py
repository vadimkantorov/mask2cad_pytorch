import os
import json
import math
import collections

import bisect
import copy
import itertools

import torch
import torch.utils.data
import torch.nn.functional as F
import torchvision

def create_aspect_ratio_groups(aspect_ratios, k=0):
    bins = (2 ** torch.linspace(-1, 1, 2 * k + 1)).tolist() if k > 0 else [1.0]
	bins = sorted(copy.deepcopy(bins))
	groups = [bisect.bisect_right(bins, y) for y in aspect_ratios]
    # count number of elements per group
    counts = torch.unique(groups, return_counts=True)[1]
    fbins = [0] + bins + [math.inf]
    print("Using", fbins, "as bins for aspect ratio quantization")
    print("Count of instances per bin:", counts)
    return groups
    

class RenderedViews(torchvision.datasets.VisionDataset):
    def __init__(self, root, object_rotation_quat_path, dataset, transforms = None, ext = '.jpg'):
        super().__init__(root = root, transforms = transforms)
        self.dataset = dataset
        self.ext = ext
        self.object_rotation_quat = torch.tensor(list(map(json.load(open(object_rotation_quat_path)).get, dataset.categories)), dtype = torch.float32)

    def __getitem__(self, idx):
        images, targets = self.dataset.__getitem__(idx[0], read_image = False, read_mask = False) 
        view_dir = os.path.join(self.root, targets['shape_path'])

        or_jpg = lambda path, ext = '.png': torchvision.io.read_image(path if os.path.exists(path) else path.replace(ext, '.jpg'))
        no_img = lambda idx: [k for k in idx if k > 0]
        # TODO: rerender to eliminate fixup
        fixup = lambda path: path if os.path.exists(path) else os.path.join(os.path.dirname(os.path.dirname(path)), 'model.obj', os.path.basename(path))
        
        views = torch.stack([or_jpg(os.path.join(self.root, targets['image_id']) if k == 0 else fixup(os.path.join(view_dir, f'{k:04}' + self.ext))) for k in no_img(idx[1:])])

        targets['shape_views'] = views.expand(-1, 3, -1, -1) / 255.0

        return images, targets 

class RenderedViewsSequentialSampler(torch.utils.data.Sampler):
    def __init__(self, num_examples, num_rendered_views):
        self.num_examples = num_examples
        self.num_rendered_views = num_rendered_views
        self.idx = torch.arange(1, 1 + num_rendered_views).unsqueeze(0).expand(num_examples, -1)

    def __iter__(self):
        return iter(self.idx.tolist())

    def __len__(self):
        return len(self.idx)

class RenderedViewsRandomSampler(torch.utils.data.Sampler):
    def __init__(self, num_examples, num_rendered_views, num_sampled_views, num_sampled_boxes):
        self.num_examples = num_examples
        self.num_rendered_views = num_rendered_views
        self.num_sampled_views = num_sampled_views
        self.num_sampled_boxes = num_sampled_boxes
        self.idx = None

    def set_epoch(self, epoch):
        rng = torch.Generator()
        rng.manual_seed(epoch)

        example_idx   = torch.arange(self.num_examples, dtype = torch.int64)[:, None]
        main_view_idx = torch. zeros(self.num_examples, dtype = torch.int64)[:, None]
        novel_view_idx = 1 + torch.rand(self.num_examples * self.num_sampled_boxes, self.num_rendered_views, generator = rng).argsort(-1)[..., :self.num_sampled_views].reshape(self.num_examples, -1)

        #self.idx = torch.cat([example_idx, main_view_idx, novel_view_idx], dim = -1)
        self.idx = torch.cat([example_idx, novel_view_idx], dim = -1)
        
    def __iter__(self):
        return iter(self.idx.tolist())

    def __len__(self):
        return len(self.idx)

class UniqueShapeRenderedViewsSequentialSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, num_rendered_views):
        shape2idx = {m['model'] : i for i, m in enumerate(dataset.dataset.metadata)}
        self.idx = torch.cat([torch.tensor(list(shape2idx.values()), dtype = torch.int64).unsqueeze(-1), torch.arange(1, 1 + num_rendered_views, dtype = torch.int64).repeat(len(shape2idx), 1)], dim = -1)
        
    def __iter__(self):
        return iter(self.idx.tolist())

    def __len__(self):
        return len(self.idx)

    def __len__(self):
        return len(self.dataset)
    
def stack_jagged(tensors, fill_value = 0):
    shape = [len(tensors)] + [max(t.shape[dim] for t in tensors) for dim in range(len(tensors[0].shape))]
    res = torch.full(shape, fill_value, dtype = tensors[0].dtype, device = tensors[0].device)
    for u, t in zip(res, tensors):
        u[tuple(map(slice, t.shape))] = t
    return u

def collate_fn(batch):
    assert batch

    images = stack_jagged([b[0] for b in batch])
    
    targets = dict(
        image_id        = [b[1]['image_id']                     for b in batch], 
        shape_path      = [b[1]['shape_path']                   for b in batch], 
        mask_path       = [b[1]['mask_path']                    for b in batch], 
        category        = [b[1]['category']                     for b in batch],

        num_boxes       = torch.tensor([len(b[1]['boxes']       for b in batch]),
        boxes           = stack_jagged([b[1]['boxes']           for b in batch]),
        masks           = stack_jagged([b[1]['masks']           for b in batch]), 
        shape_idx       = stack_jagged([b[1]['shape_idx']       for b in batch]), 
        labels          = stack_jagged([b[1]['labels']          for b in batch]), 
        object_location = stack_jagged([b[1]['object_location'] for b in batch]),
        object_rotation = stack_jagged([b[1]['object_rotation'] for b in batch]),
        views           = stack_jagged([b[2] for b in batch]) if len(batch[0]) > 2 else None,
    )
   
    return images, targets

# https://github.com/pytorch/pytorch/issues/23430
# https://discuss.pytorch.org/t/how-to-use-my-own-sampler-when-i-already-use-distributedsampler/62143/22
# https://github.com/catalyst-team/catalyst/blob/master/catalyst/data/sampler.py
class DatasetFromSampler(torch.utils.data.Dataset):
	def __init__(self, sampler: torch.utils.data.Sampler):
		self.sampler = sampler
		self.sampler_list = None

	def __getitem__(self, index: int):
		if self.sampler_list is None:
			self.sampler_list = list(self.sampler)
		return self.sampler_list[index]

	def __len__(self) -> int:
		return len(self.sampler)


class DistributedSamplerWrapper(torch.utils.data.DistributedSampler):
	"""
	Wrapper over `Sampler` for distributed training.
	Allows you to use any sampler in distributed mode.
	It is especially useful in conjunction with
	`torch.nn.parallel.DistributedDataParallel`. In such case, each
	process can pass a DistributedSamplerWrapper instance as a DataLoader
	sampler, and load a subset of subsampled data of the original dataset
	that is exclusive to it.
	.. note::
		Sampler is assumed to be of constant size.
	"""

	def __init__(
			self,
			sampler,
			num_replicas: typing.Optional[int] = None,
			rank: typing.Optional[int] = None,
			shuffle: bool = False,
	):
		"""
		Args:
			sampler: Sampler used for subsampling
			num_replicas (int, optional): Number of processes participating in
			  distributed training
			rank (int, optional): Rank of the current process
			  within ``num_replicas``
			shuffle (bool, optional): If true sampler will shuffle the indices
		"""
		super().__init__(
			DatasetFromSampler(sampler),
			num_replicas = num_replicas,
			rank = rank,
			shuffle = shuffle,
		)
		self.sampler = sampler

	def __iter__(self):
		# comments are specific for BucketingBatchSampler as self.sampler, variable names are kept from Catalyst
		self.dataset = DatasetFromSampler(self.sampler)  # hack for DistributedSampler compatibility
		indexes_of_indexes = super().__iter__()  # type: List[int] # batch indices of BucketingBatchSampler
		subsampler_indexes = self.dataset  # type: List[List[int]] # original example indices
		ddp_sampling_operator = operator.itemgetter(
			*indexes_of_indexes)  # operator to extract rank specific batches from original sampled
		return iter(ddp_sampling_operator(subsampler_indexes))  # type: Iterable[List[int]]

	def state_dict(self):
		return self.sampler.state_dict()

	def load_state_dict(self, state_dict):
		self.sampler.load_state_dict(state_dict)

	def set_epoch(self, epoch):
		super().set_epoch(epoch)
		self.sampler.set_epoch(epoch)

	@property
	def batch_idx(self):
		return self.sampler.batch_idx

	@batch_idx.setter
	def batch_idx(self, value):
		self.sampler.batch_idx = value



class GroupedBatchSampler(torch.utils.data.BatchSampler):
    """
    Wraps another sampler to yield a mini-batch of indices.
    It enforces that the batch only contain elements from the same group.
    It also tries to provide mini-batches which follows an ordering which is
    as close as possible to the ordering from the original sampler.
    Args:
        sampler (Sampler): Base sampler.
        group_ids (list[int]): If the sampler produces indices in range [0, N),
            `group_ids` must be a list of `N` ints which contains the group id of each sample.
            The group ids must be a continuous set of integers starting from
            0, i.e. they must be in the range [0, num_groups).
        batch_size (int): Size of mini-batch.
    """
    def __init__(self, sampler, group_ids, batch_size):
        if not isinstance(sampler, torch.utils.data.Sampler):
            raise ValueError(
                "sampler should be an instance of "
                "torch.utils.data.Sampler, but got sampler={}".format(sampler)
            )
        self.sampler = sampler
        self.group_ids = group_ids
        self.batch_size = batch_size

    def __iter__(self):
		_repeat_to_at_least = lambda iterable, n: list(itertools.chain.from_iterable(itertools.repeat(iterable, math.ceil(n / len(iterable)))))

        buffer_per_group  = collections.defaultdict(list)
        samples_per_group = dollections.defaultdict(list)

        num_batches = 0
        for idx in self.sampler:
            group_id = self.group_ids[idx]
            buffer_per_group[group_id].append(idx)
            samples_per_group[group_id].append(idx)
            if len(buffer_per_group[group_id]) == self.batch_size:
                yield buffer_per_group[group_id]
                num_batches += 1
                del buffer_per_group[group_id]
            assert len(buffer_per_group[group_id]) < self.batch_size

        # now we have run out of elements that satisfy
        # the group criteria, let's return the remaining
        # elements so that the size of the sampler is
        # deterministic
        expected_num_batches = len(self)
        num_remaining = expected_num_batches - num_batches
        if num_remaining > 0:
            # for the remaining batches, take first the buffers with largest number
            # of elements
            for group_id, _ in sorted(buffer_per_group.items(),
                                      key=lambda x: len(x[1]), reverse=True):
                remaining = self.batch_size - len(buffer_per_group[group_id])
                samples_from_group_id = _repeat_to_at_least(samples_per_group[group_id], remaining)
                buffer_per_group[group_id].extend(samples_from_group_id[:remaining])
                assert len(buffer_per_group[group_id]) == self.batch_size
                yield buffer_per_group[group_id]
                num_remaining -= 1
                if num_remaining == 0:
                    break
        assert num_remaining == 0

    def __len__(self):
        return len(self.sampler) // self.batch_size
