from abc import abstractmethod
import numpy as np

class scheduler():
    """
    定义一个学习率调度器基类（scheduler）
    """
    def __init__(self, optimizer) -> None:
        self.optimizer = optimizer
        self.step_count = 0 # # 记录步数（step的调用次数）
    
    @abstractmethod
    def step():
        pass

class StepLR(scheduler):
    """
    定义一个固定周期衰减学习率的调度器（StepLR）
    每调用一次就步数加一，到达指定step_size则衰减学习率
    """
    def __init__(self, optimizer, step_size=30, gamma=0.1) -> None:
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma

    def step(self) -> None:
        self.step_count += 1
        if self.step_count >= self.step_size:
            self.optimizer.init_lr *= self.gamma
            self.step_count = 0

class MultiStepLR(scheduler):
    """
    定义一个多周期衰减学习率的调度器（MultiStepLR）
    每调用一次就步数加一，到达指定milestones则衰减学习率
    """
    def __init__(self, optimizer, milestones, gamma=0.1) -> None:
        super().__init__(optimizer)
        self.milestones = sorted(milestones)
        self.gamma = gamma
        self.step_count = 0
        self.current_milestone_index = 0

    def step(self) -> None:
        self.step_count += 1
        # 检查当前步数是否到达下一个里程碑
        if (self.current_milestone_index < len(self.milestones) and
            self.step_count == self.milestones[self.current_milestone_index]):
            self.optimizer.init_lr *= self.gamma
            self.current_milestone_index += 1

class ExponentialLR(scheduler):
    """
    定义一个指数衰减学习率的调度器（ExponentialLR）
    每调用一次就步数加一，衰减学习率
    """
    def __init__(self, optimizer, gamma=0.9) -> None:
        super().__init__(optimizer)
        self.gamma = gamma

    def step(self) -> None:
        self.optimizer.init_lr *= self.gamma