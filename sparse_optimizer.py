from transformers import AdamW 
from torch.optim import Optimizer
import torch
import math
import numpy as np

class SparseAdamW(AdamW):
    def __init__(self,
                sparse_lambda = 0.1,
                lambda_schedule = None,
                max_lambda = None,
                lambda_num = None,
                **kwargs
                ):
        super().__init__(**kwargs)
        self.sparse_lambda = sparse_lambda
        print(f"lambda in optimizer={self.sparse_lambda}")
        self.lambda_idx = 0
        self.lambda_schedule = lambda_schedule
        self._build_lambda_list(max_lambda, lambda_num)
    # 定义了一些与稀疏正则化有关的超参数，例如sparse_lambda（初始稀疏正则化强度）、lambda_schedule（稀疏正则化强度变化方式，可以选择"linear"、"log_linear"或"exp_linear"，或者直接提供一个list作为预定的强度序列）
    # max_lambda（最大稀疏正则化强度）和lambda_num（当使用线性调度时，表示总共有多少个不同的稀疏强度值）。
    # 调用父类AdamW的初始化方法，并打印当前使用的稀疏正则化强度sparse_lambda。
    # 初始化变量lambda_idx用于追踪当前正在使用的稀疏正则化强度索引，并根据给定的max_lambda和lambda_num构建稀疏正则化强度列表_lambdas。
    def _build_lambda_list(self, max_lambda, lambda_num):
        if self.lambda_schedule is None:
            self._lambdas = None
            return
        if isinstance(self.lambda_schedule, list):
            self._lambdas = self.lambda_schedule
        if self.lambda_schedule == "linear":
            assert max_lambda is not None and lambda_num is not None, print(f"when using linear schedule, max_lambda and lambda_num must be provided, but got ({max_lambda} and {lambda_num})")
            self._lambdas = np.linspace(self.sparse_lambda, max_lambda, lambda_num)
        elif self.lambda_schedule == "log_linear":
            assert max_lambda is not None and lambda_num is not None, print(f"when using log_linear schedule, max_lambda and lambda_num must be provided, but got ({max_lambda} and {lambda_num})")
            self._lambdas = np.log(np.linspace(np.exp(self.sparse_lambda), np.exp(max_lambda), lambda_num))
        elif self.lambda_schedule == "exp_linear":
            assert max_lambda is not None and lambda_num is not None, print(f"when using exp_linear schedule, max_lambda and lambda_num must be provided, but got ({max_lambda} and {lambda_num})")
            self._lambdas = np.exp(np.linspace(np.log(self.sparse_lambda), np.log(max_lambda), lambda_num))
        else:
            raise NotImplementedError
    # 根据lambda_schedule构建一个稀疏正则化强度的变化序列。
    # 如果未指定调度方式，则不构建序列；如果给定了一个列表，则直接使用该列表；否则根据指定的线性、对数线性或指数线性规律生成序列。

    def step_lambda(self):
        if self._lambdas is None:
            print("no lambda schedule is specified, do nothing")
            return
        else:
            if self.lambda_idx < len(self._lambdas) - 1:
                self.lambda_idx += 1
                self.sparse_lambda = self._lambdas[self.lambda_idx]
                print(f"use lambda={self.sparse_lambda}")
            else:
                print(f"reach end of self._lambdas, keep using lambda={self.sparse_lambda}")
    # 在每次调用此方法时，如果存在稀疏正则化强度变化序列，则根据lambda_idx更新sparse_lambda的值。当达到序列末尾时，保持使用最后一个稀疏正则化强度。

    
    def step(self, closure = None):
        """
        Performs a single optimization step.
        Arguments:
            closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                # Add weight decay at the end (fixed version)
                
                # params with sparsity regularization do not need weight decay
                # still hard to decide: which quantity stands for $\eta$ in Adam? group['lr] or stepsize?
                to_add = torch.div(exp_avg, denom) * (-step_size)
                if group["weight_decay"] > 0.0:
                    # p.data.add_(p.data, alpha=(-group["lr"] * group["weight_decay"]))
                    to_add = to_add + (-group["lr"] * group["weight_decay"]) * p.data
                p.data.add_(to_add) 


                if self.sparse_lambda > 0:
                    p.data[p.data > self.sparse_lambda] -= self.sparse_lambda
                    p.data[p.data < -self.sparse_lambda] += self.sparse_lambda
                    p.data[abs(p.data) < self.sparse_lambda] = 0.0
                
        return loss
    # 实现了单步优化过程，这是优化器的核心方法。
    # 方法内部首先获取模型参数及其梯度，并检查是否存在稀疏梯度。如果梯度是稀疏的，则抛出错误提示应当使用SparseAdam而非AdamW。
    # 按照AdamW算法更新一阶动量exp_avg和二阶动量exp_avg_sq，计算分母denom，然后计算本次更新的步长step_size。
    # 在权重更新阶段，考虑了权重衰减（weight decay）。如果设置了权重衰减参数，则在梯度方向的基础上叠加了权重衰减项。
    # 最后，对满足一定稀疏条件的权重进行稀疏化处理，即当权重的绝对值小于sparse_lambda时将其置零，大于sparse_lambda时做相应的增减，以此推动模型参数趋于稀疏。