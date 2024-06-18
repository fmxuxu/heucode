import os
import torch

from torch.serialization import default_restore_location


def _load_checkpoint(file_name):
    # loading to cpu
    state = torch.load(file_name, map_location=lambda s, l: default_restore_location(s, 'cpu'))

    return state

class Train_Checker:
    # 在初始化时，它会从config中读取一些参数，如保存/加载路径、保留最近几个epochs/updates等参数，并调用load_state_dict()方法来从指定位置加载之前训练过的模型权重。
    def __init__(self, cfg):#初始化方法
        # 在训练过程中，模型参数会被周期性地保存到文件，在需要恢复训练时，该类可以自动加载最近的checkpoint文件。
        self.cfg = cfg
        self.save_dir = cfg.save_dir
        self.load_checkpoint = cfg.load_checkpoint
        self.checkpoint_path = cfg.checkpoint_path

        self.keep_last_epochs = cfg.keep_last_epochs
        self.keep_last_updates = cfg.keep_last_updates

        self.state_dict = self.load_state_dict()
    # 返回当前模型所处的epoch
    def start_epoch(self):
        epoch = 1
        if self.load_checkpoint:
            epoch = 1 + self.state_dict["num_epoches"]
        return epoch
    #返回当前的迭代次数
    def num_updates(self):
        update = 0
        if self.load_checkpoint:
            update = self.state_dict["num_updates"]
        return update
    #用于加载指定路径下的 checkpoint 文件，并返回其 state_dict。
    def load_state_dict(self):
        state_dict = None#如果成功从文件中读取到 model_state 等参数，则会将这些参数加入到一个 Python 字典中并返回。否则返回 None 值。
        if self.load_checkpoint:
            state_dict = self._load_checkpoint(self.checkpoint_path)
            print(f"Load trained model from: {self.checkpoint_path}")
        return state_dict
    #将保存的状态的模型权重删除。
    def delete_model_state(self):
        if self.state_dict is not None:
            self.state_dict["model_state"] = None
    # 用来将加载回来的 state_dict 赋值给当前的模型，来实现模型恢复。
    def load_model(self, model):
        if self.load_checkpoint:
            model.load_state_dict(self.state_dict["model_state"], strict=False)
    #加载优化器的state_dict
    def load_optimizer_state(self, optimizer):
        if self.load_checkpoint:
            optimizer.load_state_dict(self.state_dict["optimizer_state"])
    # 加载学习率调度器的state_dict。
    def load_lr_scheduler_state(self, lr_scheduler):
        if self.load_checkpoint:
            lr_scheduler.load_state_dict(self.state_dict["lr_scheduler_state"])
    #将当前模型及其状态保存为 checkpoint
    def save_checkpoint(self, epoch, num_updates, model, lr_scheduler, optimizer, save_best=False):
        state_dict = {
            "cfg": self.cfg,
            "num_epoches": epoch,
            "num_updates": num_updates,
            "model_state": model.state_dict(),
            "lr_scheduler_state": lr_scheduler.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "optimizer_history": [],
            "extra_state": {}
        }

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        if save_best:
            self._save_checkpoint(f"{self.save_dir}/checkpoint_best.pt", **state_dict)

        if not self.keep_last_epochs == -1:
            self._save_checkpoint(f"{self.save_dir}/checkpoint_{epoch}.pt", **state_dict)
            to_remove_file = f"{self.save_dir}/checkpoint_{epoch - self.keep_last_epochs}.pt"
            if os.path.lexists(to_remove_file):
                os.remove(to_remove_file)
        if not self.keep_last_updates == -1:
            self._save_checkpoint(f"{self.save_dir}/checkpoint_{num_updates}.pt", **state_dict)
            to_remove_file = f"{self.save_dir}/checkpoint_{num_updates - self.keep_last_updates}.pt"
            if os.path.lexists(to_remove_file):
                os.remove(to_remove_file)

        self._save_checkpoint(f"{self.save_dir}/checkpoint_last.pt", **state_dict)

    def _load_checkpoint(self, file_name):
        # loading to cpu 用来加载 checkpoint 文件，并返回 state_dict。它会调用 PyTorch 的 torch.load() 函数来加载模型文件，然后将其返回。同时，通过 map_location 参数可以指定在哪个设备上恢复模型参数。
        state = torch.load(file_name, map_location=lambda s, l: default_restore_location(s, 'cpu'))

        return state

    def _save_checkpoint(self, file_name, cfg, num_epoches, num_updates, model_state={}, \
                        lr_scheduler_state={}, optimizer_state={}, optimizer_history=[], extra_state={}):
        state = {
            "cfg": cfg,
            "num_epoches": num_epoches,
            "num_updates": num_updates,
            "model_state": model_state,
            "lr_scheduler_state": lr_scheduler_state,
            "optimizer_state": optimizer_state,
            "optimizer_history": optimizer_history,
            "extra_state": extra_state
        }

        torch.save(state, file_name)

class Evaluate_Checker:

    def __init__(self, checkpoint_path):

        self.state_dict = self.load_state_dict(checkpoint_path)

    def load_state_dict(self, checkpoint_path):
        state_dict = _load_checkpoint(checkpoint_path)
        print(f"Load state dict from: {checkpoint_path}")
        return state_dict

    def load_model(self, model):
        model.load_state_dict(self.state_dict["model_state"], strict=False)

    def delete_model_state(self):
        self.state_dict["model_state"] = None

    def start_epoch(self):
        epoch = 1 + self.state_dict["num_epoches"]
        return epoch

    def num_updates(self):
        update = self.state_dict["num_updates"]
        return update