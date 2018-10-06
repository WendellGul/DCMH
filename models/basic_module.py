import torch as t
import time


class BasicModule(t.nn.Module):
    """
    封装nn.Module，主要提供save和load两个方法
    """
    def __init__(self):
        super(BasicModule, self).__init__()
        self.module_name = str(type(self))

    def load(self, path, use_gpu=False):
        """
        可加载指定路径的模型
        """
        if not use_gpu:
            self.load_state_dict(t.load(path, map_location=lambda storage, loc: storage))
        else:
            self.load_state_dict(t.load(path))

    def save(self, name=None):
        """
        保存模型，默认使用"模型名字+时间"作为文件名
        """
        if name is None:
            prefix = self.module_name + '_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        t.save(self.state_dict(), 'checkpoints/' + name)
        return name

    def forward(self, *input):
        pass

