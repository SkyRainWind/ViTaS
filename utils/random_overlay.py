from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


import json
import os
import cv2 as cv
import torch
import torchvision.transforms as TF
import torchvision.datasets as datasets

places_dataloader = None
places_iter = None

# TODO: change the path
def load_config(key=None):
    path = os.path.join('/home/csq/work_tyf/orienc/M3L/utils', 'aug_config.cfg')
    with open(path) as f:
        data = json.load(f)
    if key is not None:
        return data[key]
    return data


def _load_places(batch_size=256, image_size=84, num_workers=8, use_val=False):
	global places_dataloader, places_iter
	partition = 'val' if use_val else 'train'
	print(f'Loading {partition} partition of places365_standard...')
	for data_dir in load_config('datasets'):
		if os.path.exists(data_dir):
			fp = os.path.join(data_dir, 'places365_standard', partition)
			if not os.path.exists(fp):
				print(f'Warning: path {fp} does not exist, falling back to {data_dir}')
				fp = data_dir
			places_dataloader = torch.utils.data.DataLoader(
				datasets.ImageFolder(fp, TF.Compose([
					TF.RandomResizedCrop(image_size),
					TF.ToTensor()
				])), # TODO: change to following
				# datasets.ImageFolder(fp, TF.Compose([
				# 	TF.RandomResizedCrop(image_size),
				# 	TF.RandomHorizontalFlip(),
				# 	TF.ToTensor()
				# ])),
				batch_size=batch_size, shuffle=True,
				num_workers=num_workers, pin_memory=True)
			places_iter = iter(places_dataloader)
			break
	if places_iter is None:
		raise FileNotFoundError('failed to find places365 data at any of the specified paths')
	print('Loaded dataset from', data_dir)


def _get_places_batch(batch_size):
	global places_iter
	try:
		imgs, _ = next(places_iter)
		if imgs.size(0) < batch_size:
			places_iter = iter(places_dataloader)
			imgs, _ = next(places_iter)
	except StopIteration:
		places_iter = iter(places_dataloader)
		imgs, _ = next(places_iter)
	return imgs.cuda()


def random_overlay(x, dataset='places365_standard'):
	"""Randomly overlay an image from Places"""
	global places_iter
	alpha = 0.5

	if dataset == 'places365_standard':
		if places_dataloader is None:
			_load_places(batch_size=x.size(0), image_size=x.size(-1))
		imgs = _get_places_batch(batch_size=x.size(0)).repeat(1, x.size(1)//3, 1, 1)
		print('imgs shape: ', imgs.shape)
		print('imgs: ',imgs)
		test = imgs[0].cpu().numpy()
		test = test[:3]
		# test = test.transpose(1, 2, 0)
		print('test shape: ', test.shape)
		cv.imwrite("1.jpg", test)
		exit(0)
		# print('hh shape', x.shape, imgs.shape)
	else:
		raise NotImplementedError(f'overlay has not been implemented for dataset "{dataset}"')

	return ((1-alpha)*(x/255.) + (alpha)*imgs)*255.