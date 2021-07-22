# -*- coding: utf-8 -*-
# @Author: Wenwen Yu
# @Created Time: 7/12/2020 9:50 PM

import os
import shutil

import numpy as np
from numpy import inf
import distance
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.metrics import AverageMetricTracker
from logger import TensorboardWriter
from utils.label_util import LabelTransformer
from utils import decode_util


class Trainer:
    """
    Trainer class
    """

    def __init__(self, model, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, max_len_step=None):
        '''

        :param model:
        :param optimizer:
        :param config:
        :param data_loader:
        :param valid_data_loader:
        :param lr_scheduler:
        :param max_len_step: controls number of batches(steps) in each epoch.
        '''
        self.config = config
        self.distributed = config['distributed']
        if self.distributed:
            self.local_master = (config['local_rank'] == 0)
            self.global_master = (dist.get_rank() == 0)
        else:
            self.local_master = True
            self.global_master = True
        self.logger = config.get_logger('trainer', config['trainer']['log_verbosity']) if self.local_master else None

        # setup GPU device if available, move model into configured device
        self.device, self.device_ids = self._prepare_device(config['local_rank'], config['local_world_size'])
        self.model = model.to(self.device)

        self.optimizer = optimizer

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']
        monitor_open = cfg_trainer['monitor_open']
        if monitor_open:
            self.monitor = cfg_trainer.get('monitor', 'off')
        else:
            self.monitor = 'off'

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.monitor_mode = 'off'
            self.monitor_best = 0
        else:
            self.monitor_mode, self.monitor_metric = self.monitor.split()
            assert self.monitor_mode in ['min', 'max']

            self.monitor_best = inf if self.monitor_mode == 'min' else -inf
            self.early_stop = cfg_trainer.get('early_stop', inf)
            self.early_stop = inf if self.early_stop == -1 else self.early_stop

        self.start_epoch = 1

        if self.local_master:
            self.checkpoint_dir = config.save_dir
            # setup visualization writer instance
            self.writer = TensorboardWriter(config.log_dir, self.logger, cfg_trainer['tensorboard'])

        # load checkpoint for resume training or finetune
        self.finetune = config['finetune']
        if config.resume is not None:
            self._resume_checkpoint(config.resume)
        else:
            if self.finetune:
                self.logger_warning("Finetune mode must set resume args to specific checkpoint path")
                raise RuntimeError("Finetune mode must set resume args to specific checkpoint path")
        # load checkpoint then load to multi-gpu, avoid 'module.' prefix
        if self.config['trainer']['sync_batch_norm'] and self.distributed:
            # sync_batch_norm only support one gpu per process mode
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)

        if self.distributed:  # move model to distributed gpu
            self.model = DDP(self.model, device_ids=self.device_ids, output_device=self.device_ids[0],
                             find_unused_parameters=True)

        # iteration-based training
        self.len_step = len(data_loader)
        self.data_loader = data_loader
        if max_len_step is not None:  # max length of iteration step of every epoch
            self.len_step = min(max_len_step, self.len_step)
        self.valid_data_loader = valid_data_loader

        do_validation = self.config['trainer']['do_validation']
        self.validation_start_epoch = self.config['trainer']['validation_start_epoch']
        self.do_validation = (self.valid_data_loader is not None and do_validation)
        self.lr_scheduler = lr_scheduler

        log_step = self.config['trainer']['log_step_interval']
        self.log_step = log_step if log_step != -1 and 0 < log_step < self.len_step else int(
            np.sqrt(data_loader.batch_size))

        # do validation interval
        val_step_interval = self.config['trainer']['val_step_interval']
        # self.val_step_interval = val_step_interval if val_step_interval!= -1 and 0 < val_step_interval < self.len_step\
        #                                             else int(np.sqrt(data_loader.batch_size))
        self.val_step_interval = val_step_interval

        # build metrics tracker and wrapper tensorboard writer.
        self.train_metrics = AverageMetricTracker('loss',
                                                  writer=self.writer if self.local_master else None)
        self.val_metrics = AverageMetricTracker('loss', 'word_acc', 'word_acc_case_insensitive', 'edit_distance_acc',
                                                writer=self.writer if self.local_master else None)

    def train(self):
        """
        Full training logic, including train and validation.
        """

        if self.distributed:
            dist.barrier()  # Syncing machines before training

        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):

            # ensure distribute worker sample different data,
            # set different random seed by passing epoch to sampler
            if self.distributed:
                self.data_loader.sampler.set_epoch(epoch)

            self.valid_data_loader.batch_sampler.set_epoch(
                epoch) if self.valid_data_loader.batch_sampler is not None else None

            torch.cuda.empty_cache()
            result_dict = self._train_epoch(epoch)
            # import pdb;pdb.set_trace()

            # validate after training an epoch
            if self.do_validation and epoch >= self.validation_start_epoch:
                val_metric_res_dict = self._valid_epoch(epoch)
                # import pdb;pdb.set_trace()
                val_res = f"\nValidation result after {epoch} epoch: " \
                          f"Word_acc: {val_metric_res_dict['word_acc']:.6f} " \
                          f"Word_acc_case_ins: {val_metric_res_dict['word_acc_case_insensitive']:.6f} " \
                          f"Edit_distance_acc: {val_metric_res_dict['edit_distance_acc']:.6f}"
            else:
                val_res = ''

            # update lr after training an epoch, epoch-wise
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            # every epoch log information
            self.logger_info(
                '[Epoch End] Epoch:[{}/{}] Loss: {:.6f} LR: {:.8f}'.
                format(epoch, self.epochs,
                       result_dict['loss'], self._get_lr()) + val_res
            )

            # evaluate model performance according to configured metric, check early stop, and
            # save best checkpoint as model_best
            best = False
            if self.monitor_mode != 'off' and self.do_validation and epoch >= self.validation_start_epoch:
                best, not_improved_count = self._is_best_monitor_metric(best, not_improved_count, val_metric_res_dict)
                if not_improved_count > self.early_stop:  # epoch level count
                    self.logger_info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.early_stop))
                    break
            # epoch-level save period
            if best or (epoch % self.save_period == 0 and epoch >= self.validation_start_epoch):
                self._save_checkpoint(epoch, save_best=best)

    def _is_best_monitor_metric(self, best, not_improved_count, val_result_dict, update_not_improved_count=True):
        '''
        monitor metric
        :param best: bool
        :param not_improved_count: int
        :param val_result_dict: dict
        :param update_monitor_best: bool,  true: update monitor_best when epoch-level validation
        :return:
        '''
        val_monitor_metric_res = val_result_dict[self.monitor_metric]
        try:
            # check whether model performance improved or not, according to specified metric(monitor_metric)
            improved = (self.monitor_mode == 'min' and val_monitor_metric_res <= self.monitor_best) or \
                       (self.monitor_mode == 'max' and val_monitor_metric_res >= self.monitor_best)
        except KeyError:
            self.logger_warning("Warning: Metric '{}' is not found. "
                                "Model performance monitoring is disabled.".format(self.monitor_metric))
            self.monitor_mode = 'off'
            improved = False
        if improved:
            self.monitor_best = val_monitor_metric_res
            not_improved_count = 0
            best = True
        else:
            if update_not_improved_count:  # update when do epoch-level validation, step-level not changed count
                not_improved_count += 1
        return best, not_improved_count

    def _train_epoch(self, epoch):
        '''
        Training logic for an epoch
        :param epoch: Integer, current training epoch.
        :return: A log dict that contains average loss and metric in this epoch.
        '''
        self.model.train()

        self.train_metrics.reset()

        ## step iteration start ##
        for step_idx, input_data_item in enumerate(self.data_loader):

            batch_size = input_data_item['batch_size']
            if batch_size == 0: continue

            images = input_data_item['images']
            text_label = input_data_item['labels']

            # # step-wise lr scheduler, comment this, using epoch-wise lr_scheduler
            # if self.lr_scheduler is not None:
            #     self.lr_scheduler.step()

            # for step_idx in range(self.len_step):
            step_idx += 1
            # import pdb;pdb.set_trace()
            # prepare input data
            images = images.to(self.device)
            target = LabelTransformer.encode(text_label)
            target = target.to(self.device)
            target = target.permute(1, 0)
            with torch.autograd.set_detect_anomaly(self.config['trainer']['anomaly_detection']):
                outputs = self.model(images, target[:, :-1])  # need to remove <EOS> in target
                loss = F.cross_entropy(outputs.contiguous().view(-1, outputs.shape[-1]),
                                       target[:, 1:].contiguous().view(-1),  # need to remove <SOS> in target
                                       ignore_index=LabelTransformer.PAD)

                # backward and update parameters
                self.optimizer.zero_grad()
                loss.backward()
                # self.average_gradients(self.model)
                self.optimizer.step()

            ## Train batch done. Logging results

            # due to training mode (bn, dropout), we don't calculate acc

            batch_total = images.shape[0]
            reduced_loss = loss.item()  # mean results of ce

            if self.distributed:
                # obtain the sum of all train metrics at all processes by all_reduce operation
                # Must keep track of global batch size,
                # since not all machines are guaranteed equal batches at the end of an epoch
                reduced_metrics_tensor = torch.tensor([batch_total, reduced_loss]).float().to(self.device)
                # Use a barrier() to make sure that all process have finished above code
                dist.barrier()
                # averages metric tensor across the whole world
                # import pdb;pdb.set_trace()
                # reduced_metrics_tensor = self.mean_reduce_tensor(reduced_metrics_tensor)
                reduced_metrics_tensor = self.sum_tesnor(reduced_metrics_tensor)
                batch_total, reduced_loss = reduced_metrics_tensor.cpu().numpy()
                reduced_loss = reduced_loss / dist.get_world_size()
            # update metrics and write to tensorboard
            global_step = (epoch - 1) * self.len_step + step_idx - 1
            self.writer.set_step(global_step, mode='train') if self.local_master else None
            # write tag is loss/train (mode =train)
            self.train_metrics.update('loss', reduced_loss,
                                      batch_total)  # here, loss is mean results over batch, accumulate values

            # log messages
            if step_idx % self.log_step == 0 or step_idx == 1:
                self.logger_info(
                    'Train Epoch:[{}/{}] Step:[{}/{}] Loss: {:.6f} Loss_avg: {:.6f} LR: {:.8f}'.
                        format(epoch, self.epochs, step_idx, self.len_step,
                               self.train_metrics.val('loss'),
                               self.train_metrics.avg('loss'), self._get_lr()))
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            # do validation after val_step_interval iteration
            if self.do_validation and step_idx % self.val_step_interval == 0 and epoch >= self.validation_start_epoch:
                val_metric_res_dict = self._valid_epoch(epoch)  # average metric
                self.logger_info(
                    '[Step Validation] Epoch:[{}/{}] Step:[{}/{}] Word_acc: {:.6f} Word_acc_case_ins {:.6f}'
                    'Edit_distance_acc: {:.6f}'.
                        format(epoch, self.epochs, step_idx, self.len_step,
                               val_metric_res_dict['word_acc'], val_metric_res_dict['word_acc_case_insensitive'],
                               val_metric_res_dict['edit_distance_acc']))
                # check if best metric, if true, then save as model_best checkpoint.
                best, not_improved_count = self._is_best_monitor_metric(False, 0, val_metric_res_dict,
                                                                        update_not_improved_count=False)
                if best:  # step-level valida then save model
                    self._save_checkpoint(epoch, best, step_idx)

            # decide whether continue iter
            if step_idx == self.len_step:
                break
        ## step iteration end ##

        log_dict = self.train_metrics.result()
        return log_dict

    def _valid_epoch(self, epoch):
        '''
         Validate after training an epoch or regular step, this is a time-consuming procedure if validation data is big.
        :param epoch: Integer, current training epoch.
        :return: A dict that contains information about validation
        '''

        self.model.eval()
        self.val_metrics.reset()

        for step_idx, input_data_item in enumerate(self.valid_data_loader):

            batch_size = input_data_item['batch_size']
            images = input_data_item['images']
            text_label = input_data_item['labels']

            if self.distributed:
                word_acc, word_acc_case_ins, edit_distance_acc, total_distance_ref, batch_total = \
                    self._distributed_predict(batch_size, images, text_label)
            else:  # one cpu or gpu non-distributed mode
                with torch.no_grad():
                    images = images.to(self.device)
                    # target = LabelTransformer.encode(text_label)
                    # target = target.to(self.device)
                    # target = target.permute(1, 0)

                    if hasattr(self.model, 'module'):
                        model = self.model.module
                    else:
                        model = self.model

                    # (bs, max_len)
                    outputs, _ = decode_util.greedy_decode_with_probability(model, images, LabelTransformer.max_length,
                                                                            LabelTransformer.SOS,
                                                                            LabelTransformer.EOS,
                                                                            _padding_symbol_index=LabelTransformer.PAD,
                                                                            _result_device=images.device,
                                                                            _is_padding=True)
                    correct = 0
                    correct_case_ins = 0
                    total_distance_ref = 0
                    total_edit_distance = 0
                    for index, (pred, text_gold) in enumerate(zip(outputs[:, 1:], text_label)):
                        predict_text = ""
                        for i in range(len(pred)):  # decode one sample
                            if pred[i] == LabelTransformer.EOS: break
                            if pred[i] == LabelTransformer.UNK: continue

                            decoded_char = LabelTransformer.decode(pred[i])
                            predict_text += decoded_char

                        # calculate edit distance
                        ref = len(text_gold)
                        edit_distance = distance.levenshtein(text_gold, predict_text)
                        total_distance_ref += ref
                        total_edit_distance += edit_distance

                        # calculate word accuracy related
                        # predict_text = predict_text.strip()
                        # text_gold = text_gold.strip()
                        if predict_text == text_gold:
                            correct += 1
                        if predict_text.lower() == text_gold.lower():
                            correct_case_ins += 1
                    batch_total = images.shape[0]  # valid batch size of current steps
                    # calculate accuracy directly, due to non-distributed
                    word_acc = correct / batch_total
                    word_acc_case_ins = correct_case_ins / batch_total
                    edit_distance_acc = 1 - total_edit_distance / total_distance_ref

            # update valid metric and write to tensorboard,
            self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + step_idx, 'valid') \
                if self.local_master else None
            # self.val_metrics.update('loss', loss, batch_total)  # tag is loss/valid (mode =valid)
            self.val_metrics.update('word_acc', word_acc, batch_total)
            self.val_metrics.update('word_acc_case_insensitive', word_acc_case_ins, batch_total)
            self.val_metrics.update('edit_distance_acc', edit_distance_acc, total_distance_ref)

        val_metric_res_dict = self.val_metrics.result()

        # rollback to train mode
        self.model.train()

        return val_metric_res_dict

    def _distributed_predict(self, batch_size, images, text_label):
        # Allows distributed prediction on uneven batches.
        # Test set isn't always large enough for every GPU to get a batch

        # obtain the sum of all val metrics at all processes by all_reduce operation
        # dist.barrier()
        # batch_size = images.size(0)
        correct = correct_case_ins = valid_batches = total_edit_distance = total_distance_ref = 0
        if batch_size:  # not empty samples at current gpu validation process
            with torch.no_grad():
                images = images.to(self.device)
                # target = LabelTransformer.encode(text_label)
                # target = target.to(self.device)
                # target = target.permute(1, 0)

                if hasattr(self.model, 'module'):
                    model = self.model.module
                else:
                    model = self.model
                outputs, _ = decode_util.greedy_decode_with_probability(model, images, LabelTransformer.max_length,
                                                                        LabelTransformer.SOS,
                                                                        LabelTransformer.EOS,
                                                                        _padding_symbol_index=LabelTransformer.PAD,
                                                                        _result_device=images.device,
                                                                        _is_padding=True)

                for index, (pred, text_gold) in enumerate(zip(outputs[:, 1:], text_label)):
                    predict_text = ""
                    for i in range(len(pred)):  # decode one sample
                        if pred[i] == LabelTransformer.EOS: break
                        if pred[i] == LabelTransformer.UNK: continue

                        decoded_char = LabelTransformer.decode(pred[i])
                        predict_text += decoded_char

                    # calculate edit distance
                    ref = len(text_gold)
                    edit_distance = distance.levenshtein(text_gold, predict_text)
                    total_distance_ref += ref
                    total_edit_distance += edit_distance

                    # calculate word accuracy related
                    # predict_text = predict_text.strip()
                    # text_gold = text_gold.strip()
                    if predict_text == text_gold:
                        correct += 1
                    if predict_text.lower() == text_gold.lower():
                        correct_case_ins += 1

            valid_batches = 1  # can be regard as dist.world_size
        # sum metrics across all valid process
        sum_metrics_tensor = torch.tensor([batch_size, valid_batches,
                                           correct, correct_case_ins, total_edit_distance,
                                           total_distance_ref]).float().to(self.device)
        # # Use a barrier() to make sure that all process have finished above code
        # dist.barrier()
        sum_metrics_tensor = self.sum_tesnor(sum_metrics_tensor)
        sum_metrics_tensor = sum_metrics_tensor.cpu().numpy()
        batch_total, valid_batches = sum_metrics_tensor[0:2]
        # averages metric across the valid process
        # loss= sum_metrics_tensor[2] / valid_batches
        correct, correct_case_ins, total_edit_distance, total_distance_ref = sum_metrics_tensor[2:]
        word_acc = correct / batch_total
        word_acc_case_ins = correct_case_ins / batch_total
        edit_distance_acc = 1 - total_edit_distance / total_distance_ref
        return word_acc, word_acc_case_ins, edit_distance_acc, total_distance_ref, batch_total

    def _get_lr(self):
        for group in self.optimizer.param_groups:
            return group['lr']

    def average_gradients(self, model):
        '''
        Gradient averaging
        :param model:
        :return:
        '''
        size = float(dist.get_world_size())
        for param in model.parameters():
            dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
            param.grad.data /= size

    def mean_reduce_tensor(self, tensor: torch.Tensor):
        ''' averages tensor across the whole world'''
        sum_tensor = self.sum_tesnor(tensor)
        return sum_tensor / dist.get_world_size()

    def sum_tesnor(self, tensor: torch.Tensor):
        '''obtain the sum of tensor at all processes'''
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        return rt

    def logger_info(self, msg):
        self.logger.info(msg) if self.local_master else None

    def logger_warning(self, msg):
        self.logger.warning(msg) if self.local_master else None

    def _prepare_device(self, local_rank, local_world_size):
        '''
         setup GPU device if available, move model into configured device
        :param local_rank:
        :param local_world_size:
        :return:
        '''
        if self.distributed:
            ngpu_per_process = torch.cuda.device_count() // local_world_size
            device_ids = list(range(local_rank * ngpu_per_process, (local_rank + 1) * ngpu_per_process))

            if torch.cuda.is_available() and local_rank != -1:
                torch.cuda.set_device(device_ids[0])  # device_ids[0] =local_rank if local_world_size = n_gpu per node
                device = 'cuda'
                self.logger_info(
                    f"[Process {os.getpid()}] world_size = {dist.get_world_size()}, "
                    + f"rank = {dist.get_rank()}, n_gpu/process = {ngpu_per_process}, device_ids = {device_ids}"
                )
            else:
                self.logger_warning('Training will be using CPU!')
                device = 'cpu'
            device = torch.device(device)
            return device, device_ids
        else:
            n_gpu = torch.cuda.device_count()
            n_gpu_use = local_world_size
            if n_gpu_use > 0 and n_gpu == 0:
                self.logger_warning("Warning: There\'s no GPU available on this machine,"
                                    "training will be performed on CPU.")
                n_gpu_use = 0
            if n_gpu_use > n_gpu:
                self.logger_warning("Warning: The number of GPU\'s configured to use is {}, but only {} are available "
                                    "on this machine.".format(n_gpu_use, n_gpu))
                n_gpu_use = n_gpu

            list_ids = list(range(n_gpu_use))
            if n_gpu_use > 0:
                torch.cuda.set_device(list_ids[0])  # only use first available gpu as devices
                self.logger_warning(f'Training is using GPU {list_ids[0]}!')
                device = 'cuda'
            else:
                self.logger_warning('Training is using CPU!')
                device = 'cpu'
            device = torch.device(device)
            return device, list_ids

    def _save_checkpoint(self, epoch, save_best=False, step_idx=None):
        '''
        Saving checkpoints
        :param epoch:  current epoch number
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        :return:
        '''
        # only both local and global master process do save model
        if not (self.local_master and self.global_master):
            return

        if hasattr(self.model, 'module'):
            arch_name = type(self.model.module).__name__
            model_state_dict = self.model.module.state_dict()
        else:
            arch_name = type(self.model).__name__
            model_state_dict = self.model.state_dict()
        state = {
            'arch': arch_name,
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.monitor_best,
            'config': self.config
        }
        if step_idx is None:
            filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
        else:
            filename = str(self.checkpoint_dir / 'checkpoint-epoch{}-step{}.pth'.format(epoch, step_idx))
        torch.save(state, filename)
        self.logger_info("Saving checkpoint: {} ...".format(filename))

        if save_best:
            best_path = str(self.checkpoint_dir / 'model_best.pth')
            shutil.copyfile(filename, best_path)
            self.logger_info(
                f"Saving current best (at {epoch} epoch): model_best.pth Best {self.monitor_metric}: {self.monitor_best:.6f}")

        # if save_best:
        #     best_path = str(self.checkpoint_dir / 'model_best.pth')
        #     torch.save(state, best_path)
        #     self.logger_info(
        #         f"Saving current best: model_best.pth Best {self.monitor_metric}: {self.monitor_best:.6f}.")
        # else:
        #     filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
        #     torch.save(state, filename)
        #     self.logger_info("Saving checkpoint: {} ...".format(filename))

    def _resume_checkpoint(self, resume_path):
        '''
        Resume from saved checkpoints
        :param resume_path: Checkpoint path to be resumed
        :return:
        '''
        resume_path = str(resume_path)
        self.logger_info("Loading checkpoint: {} ...".format(resume_path))
        # map_location = {'cuda:%d' % 0: 'cuda:%d' % self.config['local_rank']}
        checkpoint = torch.load(resume_path, map_location=self.device)
        self.start_epoch = checkpoint['epoch'] + 1 if not self.finetune else 1
        self.monitor_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        if checkpoint['config']['model_arch'] != self.config['model_arch']:  # TODO verify adapt and adv arch
            self.logger_warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        # self.model.load_state_dict(checkpoint['state_dict'])
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if not self.finetune:  # resume mode will load optimizer state and continue train
            if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
                self.logger_warning(
                    "Warning: Optimizer type given in config file is different from that of checkpoint. "
                    "Optimizer parameters not being resumed.")
            else:
                self.optimizer.load_state_dict(checkpoint['optimizer'])

        if self.finetune:
            self.logger_info("Checkpoint loaded. Finetune training from epoch {}".format(self.start_epoch))
        else:
            self.logger_info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))

    # def test_train(self):
