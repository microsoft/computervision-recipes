import argparse
import os

class opts(object):
	def __init__(self,
		task: str = "mot",
		dataset: str = "jde",
		exp_id: str = "default",
		test: bool = True,
		load_model: str = "",
		resume: bool = True,
		gpus: str = "0, 1",
		num_workers: int = 8,
		not_cuda_benchmark: bool = True
		seed: int = 317,
		print_iter: int = 0,
		hide_data_time: bool = True,
		save_all: bool = True,
		metric: str = "loss",
		vis_thresh: float = 0.5,
		arch: str = "dla_34",
		head_conv: int = -1,
		down_ratio: int = 4,
		input_res: int = -1,
		input_h: int = -1,
		input_w: int = -1,
		lr: float = 1e-4,
		lr_step: str = "20,27",
		num_epochs: int = 30,
		batch_size: int = 12,
		master_batch_size: int = -1,
		num_iters: int = -1,
		val_intervals: int = 5,
		trainval: bool = True,
		K: int = 128,
		not_prefetch_test: bool = True,
		fix_res: bool = True,
		keep_res: bool = True,
		test_mot16: bool = False,
		val_mot15: bool = False,
		test_mot15: bool = False,
		val_mot16: bool = False,
		test_mot17: bool = False,
		val_mot17: bool = False,
		val_mot20: bool = False,
		test_mot20: bool = False,
		conf_thres: float = 0.6,
		det_thres: float = 0.3,
		nms_thres: float = 0.4,
		track_buffer: int = 30,
		min_box_area: float = 200,
		input_video: str = "../videos/MOT16-03.mp4",
		output_format: str = "video",
		output_root: str = "../results",
		data_cfg: str = "../src/lib/cfg/data.json",
		data_dir: str = "/data/yfzhang/MOT/JDE",
		mse_loss: bool = True,
		reg_loss: str = "l1",
		hm_weight: float = 1,
		off_weight: float = 1,
		wh_weight: float = 0.1,
		id_loss: str = "ce",
		id_weight: float = 1,
		reid_dim: int = 512,
		norm_wh: bool = True,
		dense_wh: bool = True,
		cat_spec_wh: bool = True,
		not_reg_offset: bool = True
		):
		opt = argparse.Namespace()

		# basic experiment setting
		opt.task = task
		opt.dataset = dataset
		opt.exp_id = exp_id
		opt.test = test
		opt.load_model = load_model
		opt.resume = resume

		opt.gpus_str = gpus
		opt.gpus = [int(gpu) for gpu in gpus.split(',')]
		opt.gpus = [i for i in range(len(opt.gpus))] if opt.gpus[0] >=0 else [-1]
		
		opt.num_workers = num_workers
		opt.not_cuda_benchmark = not_cuda_benchmark
		opt.seed = seed
		opt.print_iter = print_iter
		opt.hide_data_time = hide_data_time
		opt.save_all = save_all
		opt.metric = metric
		opt.vis_thresh = vis_thresh
		opt.arch = arch

		opt.head_conv = head_conv
		if opt.head_conv == -1: # init default
			opt.head_conv = 256
		opt.pad = 31
		opt.num_stacks = 1

		opt.down_ratio = down_ratio
		opt.input_res = input_res
		opt.input_h = input_h
		opt.input_w = input_w
		opt.lr = lr
		opt.lr_step = [int(i) for i in lr_step.split(',')]
		opt.num_epochs = num_epochs
		opt.batch_size = batch_size

		opt.master_batch_size = master_batch_size
		if opt.master_batch_size == -1:
			opt.master_batch_size = opt.batch_size // len(opt.gpus)
		rest_batch_size = (opt.batch_size - opt.master_batch_size)
		opt.chunk_sizes = [opt.master_batch_size]
		for i in range(len(opt.gpus) - 1):
			slave_chunk_size = rest_batch_size // (len(opt.gpus) - 1)
			if i < rest_batch_size % (len(opt.gpus) - 1):
				slave_chunk_size += 1
			opt.chunk_sizes.append(slave_chunk_size)

		opt.num_iters = num_iters
		opt.val_intervals = val_intervals

		opt.trainval = trainval
		if opt.trainval:
			opt.val_intervals = 100000000

		opt.K = K
		opt.not_prefetch_test = not_prefetch_test
		opt.fix_res = fix_res
		opt.keep_res = keep_res
		opt.fix_res = not keep_res
		opt.test_mot16 = test_mot16
		opt.val_mot15 = val_mot15
		opt.test_mot15 = test_mot15
		opt.val_mot16 = val_mot16
		opt.test_mot16 = test_mot16
		opt.val_mot17 = val_mot17
		opt.val_mot20 = val_mot20
		opt.test_mot20 = test_mot20
		opt.conf_thres = conf_thres
		opt.det_thres = det_thres
		opt.nms_thres = nms_thres
		opt.track_buffer = track_buffer
		opt.min_box_area = min_box_area
		opt.input_video = input_video
		opt.output_format = output_format
		opt.output_root = output_root
		opt.data_cfg = data_cfg
		opt.data_dir = data_dir
		opt.mse_loss = mse_loss
		opt.reg_loss = reg_loss
		opt.hm_weight = hm_weight
		opt.off_weight = off_weight
		opt.wh_weight = wh_weight
		opt.id_loss = id_loss
		opt.id_weight = id_weight
		opt.reid_dim = reid_dim
		opt.norm_wh = norm_wh
		opt.dense_wh = dense_wh
		opt.cat_spec_wh = cat_spec_wh
		opt.not_reg_offset = not_reg_offset
		opt.reg_offset = not opt.not_reg_offset

		opt.root_dir = os.path.join(os.path.dirname(__file__), '..', '..')
	    opt.exp_dir = os.path.join(opt.root_dir, 'exp', opt.task)
	    opt.save_dir = os.path.join(opt.exp_dir, opt.exp_id)
	    opt.debug_dir = os.path.join(opt.save_dir, 'debug')

	    if opt.resume and opt.load_model == '':
	    	model_path = opt.save_dir[:-4] if opt.save_dir.endswith('TEST') else opt.save_dir
	    	opt.load_model = os.path.join(model_path, 'model_last.pth')

	    self.opt = opt
	    self._init_dataset_info()
	    
	def _init_dataset_info(self):
		default_dataset_info = {
	    'mot': {
	    'default_resolution': [608, 1088],
	    'num_classes': 1,
	    'mean': [0.408, 0.447, 0.470],
	    'std': [0.289, 0.274, 0.278],
	    'dataset': 'jde',
	    'nID': 14455
	    }
	    }
	    class Struct:
	    	def __init__(self, entries):
	    		for k,v in entries.items():
	    			self.__setattr__(k,v)
	    dataset = Struct(default_dataset_info[self.opt.task])
	    self.opt.dataset = dataset.dataset
	    self.update_dataset_info_and_set_heads(dataset)

	def update_dataset_info_and_set_heads(self, dataset):
		input_h, input_w = dataset.default_resolution
		self.opt.mean, self.opt.std = dataset.mean, dataset.std
		self.opt.num_classes = dataset.num_classes

		# input_h(w): opt.input_h overrides opt.input_res overrides dataset default
	    input_h = self.opt.input_res if self.opt.input_res > 0 else input_h
	    input_w = self.opt.input_res if self.opt.input_res > 0 else input_w
	    self.opt.input_h = self.opt.input_h if self.opt.input_h > 0 else input_h
	    self.opt.input_w = self.opt.input_w if opt.input_w > 0 else input_w
	    self.opt.output_h = self.opt.input_h // self.opt.down_ratio
	    self.opt.output_w = self.opt.input_w // self.opt.down_ratio
	    self.opt.input_res = max(opt.input_h, opt.input_w)
	    self.opt.output_res = max(opt.output_h, opt.output_w)

	    if self.opt.task == 'mot':
	    	self.opt.heads - {'hm': self.opt.num_classes,
	    						'wh': 2 if not self.opt.cat_spec_wh else 2 * self.opt.num_classes,
	    						'id': self.opt.reid_dim}
	    	if self.opt.reg_offset:
	    		self.heads.update({'reg': 2})
	    	self.opt.nID = dataset.nID
	    	self.opt.img_size = (1088, 608)
	    else:
	    	assert 0, 'task not defined'

	def opt(self):
		return self.opt
