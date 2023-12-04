import logging
from pathlib import Path

import torch
import yaml
from kornia.color import ycbcr_to_rgb
from torch.utils.data import DataLoader
from tqdm import tqdm

import loader
from config import ConfigDict, from_dict
from pipeline.detect import Detect
from pipeline.fuse import Fuse
from tools.dict_to_device import dict_to_device

import numpy
from functools import reduce
from module.detect.utils.metrics import ap_per_class


class EvalFD:
    def __init__(self, config: str | Path | ConfigDict, save_dir: str | Path):
        # init logger
        log_f = '%(asctime)s | %(filename)s[line:%(lineno)d] | %(levelname)s | %(message)s'
        logging.basicConfig(level='INFO', format=log_f)
        logging.info(f'TarDAL-v1 Inference Script')

        # init config
        if isinstance(config, str) or isinstance(config, Path):
            config = yaml.safe_load(Path(config).open('r'))
            config = from_dict(config)  # convert dict to object
        else:
            config = config
        self.config = config

        # debug mode
        if config.debug.fast_run:
            logging.warning('fast run mode is on, only for debug!')

        # save label as txt warning
        if config.inference.save_txt:
            logging.warning('labels will be saved as txt, this will slow down the inference speed!')

        # create save(output) folder
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / 'images').mkdir(exist_ok=True)
        (save_dir / 'labels').mkdir(exist_ok=True)
        logging.info(f'create save folder {str(save_dir)}')
        self.save_dir = save_dir

        # init dataset & dataloader
        data_t = getattr(loader, config.dataset.name)  # dataset type
        self.data_t = data_t
        p_dataset = data_t(root=config.dataset.root, mode='val', config=config)
        self.p_loader = DataLoader(
            p_dataset, batch_size=config.inference.batch_size, shuffle=False,
            collate_fn=data_t.collate_fn, pin_memory=True, num_workers=config.inference.num_workers,
        )

        # init pipeline
        fuse = Fuse(config, mode='inference')
        self.fuse = fuse
        detect = Detect(config, mode='inference', nc=len(p_dataset.classes), classes=p_dataset.classes, labels=p_dataset.labels)
        self.detect = detect

    @torch.inference_mode()
    def run(self):
        # matrix
        seen = 0
        dt, p, r, f1, mp, mr, map50, map_all = [0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        j_dict, stats, ap50, ap, ap_class = [], [], [], [], []

        p_l = tqdm(self.p_loader, total=len(self.p_loader), ncols=80)
        for sample in p_l:
            sample = dict_to_device(sample, self.fuse.device)
            # set description
            p_l.set_description(f'infer {sample["name"][0]} ({len(sample["name"])} images)')
            # f_net forward
            fus = self.fuse.inference(ir=sample['ir'], vi=sample['vi']) # fus = self.fuse.eval(ir=sample['ir'], vi=sample['vi'])
            # recolor
            if self.data_t.color:
                fus = torch.cat([fus, sample['cbcr']], dim=1)
                fus = ycbcr_to_rgb(fus)
            # d_net
            pred = self.detect.inference(fus)
            seen_x, preview = self.detect.eval(imgs=fus, targets=sample['labels'], stats=stats, preview=False)
            seen += seen_x
            self.data_t.pred_save(
                fus, [self.save_dir / name for name in sample['name']],
                shape=sample['shape'], pred=pred,
                save_txt=self.config.inference.save_txt,
            )
            
        # compute statistics
        stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]
        names = reduce(lambda x, y: x | y, [{idx: name} for idx, name in enumerate(self.data_t.classes)])
        if len(stats) and stats[0].any():
            tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, names=names)
            ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
            mp, mr, map50, map_all = p.mean(), r.mean(), ap50.mean(), ap.mean()
        num_t = numpy.bincount(stats[3].astype(int), minlength=len(self.data_t.classes))  # number of targets per class
        if num_t.sum() == 0:
            logging.warning(f'no labels found, can not compute metrics without labels.')
        
        # log to console (per class)
        logging.info(f'Precision: {mp:.4f} | Recall: {mr:.4f} | mAP50: {map50:.4f} | mAP: {map_all:.4f}')
        for i, c in enumerate(ap_class):
            logging.info(
                f'{names[c]} | tot: {num_t[c]} | p: {p[i]:.4f} | r: {r[i]:.4f} | ap50: {ap50[i]:.4f} | ap: {ap[i]:.4f}'
            )



