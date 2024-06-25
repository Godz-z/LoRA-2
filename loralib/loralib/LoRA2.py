#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import LoRALayer
from typing import Optional, List
import numpy as np

class SVDLinear(nn.Linear, LoRALayer):
    # SVD-based adaptation implemented in a dense layer
    def __init__(
            self,
            in_features: int,
            out_features: int,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.,
            fan_in_fan_out: bool = False,
            merge_weights: bool = True,
            **kwargs,

    ):
        # 初始化函数（__init__）接收以下参数：
        #
        # in_features：输入特征维度。
        # out_features：输出特征维度。
        # r：指定低秩分解的秩，默认为0，即不启用低秩近似。
        # lora_alpha：LoRA层的超参数，可能用于控制低秩矩阵更新的缩放因子。
        # lora_dropout：可能用于LoRA层的dropout率。
        # fan_in_fan_out：一个布尔值，表示是否按照输入/输出特征的范数对权重矩阵进行重置。
        # merge_weights：一个布尔值，指示是否合并权重。
        # **kwargs：其他传递给父类nn.Linear的额外参数。
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            k = 8
            self.lora_A = nn.Parameter(
                self.weight.new_zeros((k, in_features))
            )
            self.lora_E = nn.Parameter(
                self.weight.new_zeros(r, 1)
            )
            self.lora_B = nn.Parameter(
                self.weight.new_zeros((r, k))
            )
            self.lora_aa = nn.Parameter(
                self.weight.new_zeros((k, r))
            )
            self.lora_bb = nn.Parameter(
                self.weight.new_zeros((out_features, k))
            )
            # ------新修改，门控函数
            # self.gate.a = nn.Parameter(torch.randn(1, in_features))
            # self.gate.b = nn.Parameter(torch.randn(out_features, 1))
            # ------结束
            self.ranknum = nn.Parameter(
                self.weight.new_zeros(1), requires_grad=False
            )
            self.ranknum.data.fill_(float(self.r))
            self.scaling = self.lora_alpha if self.lora_alpha > 0 else float(self.r)
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            self.ranknum.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T
        # 调用父类nn.Linear和LoRALayer的初始化函数以设置基本的全连接层结构。
        # 初始化类属性fan_in_fan_out。
        # 如果r > 0，表示启用低秩近似：
        # 定义三个新的可训练参数矩阵lora_A, lora_E, lora_B，分别对应原始权重矩阵通过SVD分解后的部分。
        # 定义一个存储秩数值的参数ranknum，但其梯度不需要更新。
        # 将原始权重矩阵的梯度更新设为False，即冻结原始权重矩阵。
        # 调用reset_parameters()方法重置网络参数。
        # 若fan_in_fan_out为True，则将权重矩阵按行和列交换（转置）。

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A,B the same way as the default for nn.Linear
            # and E (singular values) for zero
            nn.init.zeros_(self.lora_E)
            nn.init.normal_(self.lora_A, mean=0.0, std=0.02)
            nn.init.normal_(self.lora_B, mean=0.0, std=0.02)
            nn.init.normal_(self.lora_aa, mean=0.0, std=0.02)
            nn.init.normal_(self.lora_bb, mean=0.0, std=0.02)

    # 首先，nn.Linear.reset_parameters(self)调用父类nn.Linear的reset_parameters方法，目的是重置该线性层的标准权重和偏置参数。
    # 接下来，代码检查当前实例是否具有属性'lora_A'。如果存在，那么对扩展的属性进行初始化：
    # 使用nn.init.zeros_函数将'lora_E'中的所有元素初始化为零。这意味着初始化得到的'lora_E'矩阵将是一个全零矩阵，其元素代表一些可能的奇异值。
    # 使用nn.init.normal_函数以均值0.0和标准差0.02来初始化'lora_A'。这使得'lora_A'的元素服从正态分布，有助于在训练开始时引入随机性，促进模型的学习过程。
    # 同样地，'lora_B'也被初始化为服从均值0.0和标准差0.02的正态分布。

    def train(self, mode: bool = True):
        def T(w):
            return w.T if self.fan_in_fan_out else w

        nn.Linear.train(self, mode)
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.r > 0:
                # self.weight.data -= T(
                # ------修改前向函数
                #     self.lora_B @ (self.lora_A * self.lora_E)
                # ) * self.lora_B1 @ (self.lora_A1 * self.lora_E1) * self.scaling / (self.ranknum + 1e-5)
                # ------修改结束
                result1 = T(self.lora_B @ self.lora_A)
                result2 = T(self.lora_bb @ self.lora_aa)
                print(len(result1), len(result1[0]))
                print(len(result2), len(result2[0]))
                # print(f"result1的维度是: {self.result1.shape}")
                # print(f"result2的维度是: {self.result2.shape}")
                self.weight.data -= (result1 * self.lora_E * result2) * self.scaling / (self.ranknum + 1e-5)
            self.merged = False

    # 定义了一个内部辅助函数T，它根据self.fan_in_fan_out的布尔值来决定是否对输入矩阵w进行转置操作。
    # 调用父类nn.Linear的train方法，并将当前的mode传入，这样可以确保基础线性层也相应地切换到训练或预测模式。
    # 当满足以下条件时执行一段逻辑：
    # self.merge_weights为True，表明需要合并权重。
    # self.merged为True，表示权重已经合并过。
    # 在这种情况下，进行如下操作：
    # 如果self.r > 0，说明某些条件满足，对权重进行更新。
    # 这里通过矩阵运算(self.lora_B @ (self.lora_A*self.lora_E))计算出一个新的矩阵，然后乘以一个标量self.scaling / (self.ranknum+1e-5)，并从原始权重中减去这个结果。
    # 将self.merged设置为False，表示权重不再处于合并状态。

    def eval(self):
        def T(w):
            return w.T if self.fan_in_fan_out else w

        nn.Linear.eval(self)
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0:
                # self.weight.data += T(
                #     # ------修改前向函数
                #     self.lora_B @ (self.lora_A * self.lora_E)
                # ) * self.lora_B1 @ (self.lora_A1 * self.lora_E1) * self.scaling / (self.ranknum + 1e-5)
                # ------修改结束
                result1 = T(self.lora_B @ self.lora_A)
                result2 = T(self.lora_bb @ self.lora_aa)
                # print(f"result1的维度是: {self.result1.shape}")
                # print(f"result2的维度是: {self.result2.shape}")
                print(len(result1), len(result1[0]))
                print(len(result2), len(result2[0]))
                self.weight.data += (result1 * self.lora_E * result2) * self.scaling / (self.ranknum + 1e-5)
            self.merged = True

    # 内部定义了一个辅助函数T，与之前分析的一致，根据self.fan_in_fan_out的布尔值来决定是否对输入矩阵w进行转置操作。
    # 调用父类nn.Linear的eval方法，将模型切换至评估模式，此时模型会关闭dropout等仅在训练阶段启用的特性，并且不进行梯度计算。
    # 检查以下条件：
    # self.merge_weights为True，意味着在评估模式下需要合并权重。
    # self.merged为False，表示权重尚未合并。
    # 若满足上述条件，则执行以下操作：
    # 如果self.r > 0，将执行一个矩阵运算(self.lora_B @ (self.lora_A * self.lora_E))，并将其结果乘以标量self.scaling / (self.ranknum+1e-5)后加到原始权重上。
    # 设置self.merged为True，表示权重已合并完成。

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.T if self.fan_in_fan_out else w

        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            # ------ 修改前向函数
            if self.r > 0:
                result1 = self.lora_A.T @ self.lora_B.T
                result2 = self.lora_aa.T @ self.lora_bb.T
                lora_E_flat = self.lora_E.view(-1)
                # 然后，使用 torch.diag 创建一个对角矩阵
                lora_E_diag = torch.diag(lora_E_flat)
                result += self.lora_dropout(x) @ (result1 @ lora_E_diag @ result2) * self.scaling / (
                        self.ranknum + 1e-5)
            # ------ 修改结束
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)
    # 内部定义了一个辅助函数T，与之前分析的一致，根据self.fan_in_fan_out的布尔值来决定是否对输入矩阵w进行转置操作。
    # 调用父类nn.Linear的eval方法，将模型切换至评估模式，此时模型会关闭dropout等仅在训练阶段启用的特性，并且不进行梯度计算。
    # 检查以下条件：
    # self.merge_weights为True，意味着在评估模式下需要合并权重。
    # self.merged为False，表示权重尚未合并。
    # 若满足上述条件，则执行以下操作：
    # 如果self.r > 0，将执行一个矩阵运算(self.lora_B @ (self.lora_A * self.lora_E))，并将其结果乘以标量self.scaling / (self.ranknum+1e-5)后加到原始权重上。
    # 设置self.merged为True，表示权重已合并完成。


class RankAllocator(object):
    """
    The RankAllocator for AdaLoRA Model that will be called every training step.
    Paper: https://openreview.net/pdf?id=lq62uWRJjiY

    Args:
        model: the model that we apply AdaLoRA to.
        lora_r (`int`): The initial rank for each incremental matrix.
        target_rank (`int`): The target average rank of incremental matrix.
        init_warmup (`int`): The steps of initial fine-tuning warmup.
        final_warmup (`int`): The step of final fine-tuning.
        mask_interval (`int`): The time internval between two budget allocations.
        beta1 (`float`): The hyperparameter of EMA for sensitivity smoothing.
        beta2 (`float`): The hyperparameter of EMA for undertainty quantification.
        total_step (`int`): The total training steps, correctly configured before training.
        target_total_rank (`Optinal[int]`): The speficified final total rank.
        tb_writter (`SummaryWriter`): Tensorboard SummaryWriter.
        tb_writter_loginterval (`int`): The logging interval of SummaryWriter.
    """

    def __init__(
            self, model,
            lora_r: int,
            target_rank: int,
            init_warmup: int,
            final_warmup: int,
            mask_interval: int,
            beta1: float,
            beta2: float,
            total_step: Optional[int] = None,
            target_total_rank: Optional[int] = None,
            tb_writter=None,
            tb_writter_loginterval: int = 500,
    ):
        self.ave_target_rank = target_rank
        self.target_rank = target_total_rank
        self.lora_init_rank = lora_r
        self.initial_warmup = init_warmup
        self.final_warmup = final_warmup
        self.mask_interval = mask_interval
        self.beta1 = beta1
        self.beta2 = beta2
        self.total_step = total_step

        self.model = model
        self.ipt = {}
        self.exp_avg_ipt = {}
        self.exp_avg_unc = {}
        self.cat_ipt = {}
        self.rank_pattern = {}
        self.get_lora_param_name()

        self.tb_writter = tb_writter
        self.log_interval = tb_writter_loginterval
        self.freeze_counter = 0

        # assert (self.beta1 < 1 and self.beta1 > 0)
        # assert (self.beta2 < 1 and self.beta2 > 0)

    # model: 一个模型对象，将被存储在实例变量 self.model 中。
    # lora_r: 一个整数，表示 LoRA 初始化时的秩大小，保存为 self.lora_init_rank。
    # target_rank: 目标秩大小，保存为 self.ave_target_rank。
    # init_warmup: 初始预热步数，保存为 self.initial_warmup。
    # final_warmup: 最终预热步数，保存为 self.final_warmup。
    # mask_interval: 掩码更新间隔，保存为 self.mask_interval。
    # beta1, beta2: 两个浮点数，分别是优化器中使用的 Adam 参数，分别对应一阶动量和二阶动量，保存为 self.beta1 和 self.beta2。
    # total_step: 可选的总训练步数，保存为 self.total_step。
    # target_total_rank: 另一个可选的目标总秩大小，保存为 self.target_rank。
    # tb_writter: 可能是一个 tensorboard 的写入器对象，用于记录训练过程中的信息，保存为 self.tb_writter。
    # tb_writter_loginterval: tensorboard 记录日志的间隔步数，保存为 self.log_interval。

    def set_total_step(self, total_step: int):
        # Set total step number
        self.total_step = total_step
        assert self.total_step > self.initial_warmup + self.final_warmup

    # 这个方法用来设置总的训练步数。当调用此方法时，会将传入的整数参数 total_step 赋值给类实例变量 self.total_step。
    # 同时，它还包含了一个断言，确保设置的 total_step 大于初始预热步数 (self.initial_warmup) 加上最终预热步数 (self.final_warmup)。这是因为预热步数应该在总的训练步数之内，以保证预热阶段的正确执行。

    def get_rank_pattern(self):
        # Return rank pattern
        return self.rank_pattern

    # 这个方法用来获取秩模式（rank pattern）。在初始化时，可能已经根据某种规则生成并存储了一个秩模式（self.rank_pattern），该方法将其返回给调用者。
    # 返回的是之前存储在实例变量 self.rank_pattern 中的数据，具体是什么形式取决于实际应用和初始化时如何生成的。

    def get_lora_param_name(self):
        # Prepare the budget scheduler
        self.name_set = set()
        self.total_rank = 0
        self.shape_dict = {}
        for n, p in self.model.named_parameters():
            # if "lora_A" in n:
            #     name_mat = n.replace("lora_A", "%s")
            #     self.name_set.add(name_mat)
            #     self.total_rank += p.size(0)
            #     self.shape_dict[n] = p.shape
            if "lora_aa" in n:
                name_mat = n.replace("lora_aa", "%s")
                self.name_set.add(name_mat)
                self.total_rank += p.size(1)
                self.shape_dict[n] = p.shape
            if "lora_B" in n:
                self.shape_dict[n] = p.shape
            # if "lora_bb" in n:
            #     self.shape_dict[n] = p.shape
        self.name_set = list(sorted(self.name_set))
        if self.target_rank is None:
            self.target_rank = self.ave_target_rank * len(self.name_set)

    # 初始化几个变量：
    # self.name_set: 用于存储找到的包含“lora_A”的参数名称格式化后的集合。
    # self.total_rank: 用于累计所有匹配到的“lora_A”参数的秩（这里通过参数的size(0)获取秩大小）。
    # self.shape_dict: 用于存储特定参数名称及其对应的形状。
    # 遍历模型的所有参数，通过model.named_parameters()获取每个参数的名字(n)和参数对象(p)：
    # 如果参数名(n)中包含子字符串"lora_A"，则将该参数名中的"lora_A"替换为"%s"并添加到name_set中，并累加秩（total_rank），同时将参数名和其形状存入shape_dict。
    # 如果参数名(n)中包含子字符串"lora_B"，只将其形状信息存入shape_dict。
    # 将name_set从集合转换为排序后的列表。
    # 检查self.target_rank是否为None，如果是，则根据ave_target_rank（平均目标秩）和name_set（即LORA参数A的数量）的长度来计算并设置self.target_rank，这可能是为了在各个LORA参数上分配平均的目标秩

    def schedule_threshold(self, step: int):
        # Global budget schedule
        mask_ind = False
        target_rank = self.target_rank
        initial_warmup = self.initial_warmup
        final_warmup = self.final_warmup
        total_step = self.total_step
        self.global_step = step
        progress = step / total_step
        # # beta1从0.1开始，逐渐增加至0.9
        # self.beta1 = min(0.1 + (2 * progress), 0.9)
        # # beta2从0.1开始，逐渐增加至0.9
        # self.beta2 = min(0.1 + (2 * progress), 0.9)
        #sigmoid函数
        # self.beta1 = 0.9 + (0.7 - 0.9) / (1 + np.exp(-5 * (progress - 0.5)))
        # # beta1从0.9开始，逐渐减少至0.1
        # self.beta1 = max(0.9 - (0.3 * progress),0.6)
        # # beta2从0.9开始，逐渐减少至0.1
        # self.beta2 = max(0.9 - (0.3 * progress),0.6)
        print(f"Progress: {progress:.2f}, Beta1: {self.beta1:.2f}, Beta2: {self.beta2:.2f}")
        if step <= initial_warmup:
            # Initial warmup
            curr_rank = self.total_rank
            mask_ind = False
        elif step > total_step - final_warmup:
            # Final fine-tuning
            curr_rank = self.target_rank
            # Fix the rank pattern by
            # always masking the same unimportant singluar values
            mask_ind = True
        else:
            # Budget decreasing
            mul_coeff = 1 - (step - initial_warmup) / (total_step - final_warmup - initial_warmup)
            curr_rank = target_rank + (self.total_rank - target_rank) * (mul_coeff ** 3)
            curr_rank = int(curr_rank)
            mask_ind = True if step % self.mask_interval == 0 else False
        return curr_rank, mask_ind
        # 初始化一些变量：

    # mask_ind：布尔值，表示是否应用mask。
    # target_rank：目标秩。
    # initial_warmup、final_warmup：分别表示初始预热阶段和最终微调阶段的步数。
    # total_step：总训练步数。
    # 更新全局步数self.global_step为传入的step。
    # 根据当前步数进行如下判断和操作：
    # 初始预热阶段（step <= initial_warmup）：维持当前的总秩（curr_rank = self.total_rank），不应用mask（mask_ind = False）。
    # 最终微调阶段（step > total_step - final_warmup）：设置当前秩为目标秩（curr_rank = self.target_rank），并固定rank模式，始终对同一组不重要的奇异值应用mask（mask_ind = True）。
    # 预算递减阶段（在预热和微调阶段之间）：计算当前步数在预算递减过程中的系数（mul_coeff），然后据此计算当前秩（线性插值至目标秩），并将结果向下取整。此外，如果当前步数满足mask间隔条件（如每过一定步数才进行一次mask操作），则设置mask_ind = True，否则为False。

    def update_ipt(self, model):
        for n, p in model.named_parameters():
            if "lora_aa" in n or "lora_B" in n or "lora_E" in n:
                if n not in self.ipt:
                    self.ipt[n] = torch.zeros_like(p)
                    self.exp_avg_ipt[n] = torch.zeros_like(p)
                    self.exp_avg_unc[n] = torch.zeros_like(p)
                with torch.no_grad():
                    # Calculate sensitivity
                    self.ipt[n] = (p * p.grad).abs().detach()
                    # Update sensitivity
                    # 确保beta值不超过最大设定值
                    self.exp_avg_ipt[n] = self.beta1 * self.exp_avg_ipt[n] + \
                                          (1 - self.beta1) * self.ipt[n]
                    # Update uncertainty
                    self.exp_avg_unc[n] = self.beta2 * self.exp_avg_unc[n] + \
                                          (1 - self.beta2) * (self.ipt[n] - self.exp_avg_ipt[n]).abs()

    # 遍历模型中的所有参数，通过model.named_parameters()获取参数名(n)和参数对象(p)：
    # 判断参数名(n)中是否包含子字符串"lora_"，只有当参数与LORA模块相关时，才会对其进行敏感度更新。
    # 如果某个"Lora_"开头的参数n不在字典self.ipt中，则初始化该参数的敏感度指标ipt[n]、指数移动平均敏感度exp_avg_ipt[n]和指数移动平均不确定性exp_avg_unc[n]，均为与参数p形状相同的全零张量。
    # 使用torch.no_grad()上下文管理器，在此环境下进行梯度计算不会累积新的梯度，确保不对模型参数造成影响。
    # 计算参数的敏感度（绝对值的乘积）：
    # self.ipt[n] = (p * p.grad).abs().detach()，这里利用当前参数值p与对应的梯度p.grad相乘后取绝对值，得到参数当前步的敏感度，并使用.detach()使其不再依赖于计算图，防止后续反向传播的影响。
    # 更新指数移动平均敏感度：
    # self.exp_avg_ipt[n] = self.beta1 * self.exp_avg_ipt[n] + (1-self.beta1)*self.ipt[n]，这里采用类似Adam优化器的动量项，结合过去的敏感度历史和当前敏感度计算指数移动平均值。
    # 更新指数移动平均不确定性：
    # self.exp_avg_unc[n] = self.beta2 * self.exp_avg_unc[n] + (1-self.beta2)*(self.ipt[n]-self.exp_avg_ipt[n]).abs()，这里的不确定性估计通过计算当前敏感度与指数移动平均敏感度之差的绝对值的指数移动平均得到。

    def calculate_score(self, n, p=None, metric="ipt"):
        if metric == "ipt":
            # Combine the senstivity and uncertainty
            ipt_score = self.exp_avg_ipt[n] * self.exp_avg_unc[n]
        elif metric == "mag":
            ipt_score = p.abs().detach().clone()
        else:
            raise ValueError("Unexcptected Metric: %s" % metric)
        return ipt_score
        # 根据metric参数的值选择不同的计算策略：

    # 如果metric等于字符串"ipt"，则计算的是基于重要性感知的敏感度得分。通过将之前计算并存储的指数移动平均敏感度self.exp_avg_ipt[n]与指数移动平均不确定性self.exp_avg_unc[n]相乘得到。这种方法反映了参数的重要性和变化不确定性。
    # 如果metric等于字符串"mag"，则计算的是参数的绝对值得分。直接取得参数p（如果没有提供，则可能是从模型中通过参数名n获取）的绝对值，并使用.detach().clone()确保计算结果不会影响原始梯度流且创建一个新的拷贝。
    # 若提供的metric不是上述两种情况之一，方法会抛出一个ValueError异常，提示用户提供了未预期的度量标准。
    # 最后，无论采用哪种计算方式，都返回计算得到的得分ipt_score。
    def _combine_ipt(self, ipt_E, ipt_AB):
        ipt_AB = ipt_AB.sum(dim=1, keepdim=False)
        sum_ipt = ipt_E.view(-1) + ipt_AB.view(-1)
        return sum_ipt

    # 首先对ipt_AB进行操作：
    # ipt_AB.sum(dim=1, keepdim=False)：沿着指定维度（这里是第1维）对ipt_AB求和，并通过keepdim=False将求和后的结果转换为一维张量，这样就消除了原张量在此维度上的大小。
    # 然后对ipt_E进行重塑：
    # ipt_E.view(-1)：将ipt_E变为一维张量，-1表示自动推断出合适的维度大小以保持元素总数不变。
    # 将两者相加：
    # sum_ipt = ipt_E.view(-1) + ipt_AB.view(-1)：将重塑后的ipt_E与求和后的ipt_AB按元素相加，生成一个表示总体敏感度指标的一维张量sum_ipt。

    def mask_to_target_rank(self, model, curr_rank):
        is_dict = {}
        combine_dict = {}
        singular_dict = {}
        # temp_matrices = {}
        # is_dict：在这段代码中没有使用，可能用于其他部分。
        # combine_dict：用于存储按参数名称组合后的重要性得分。
        # singular_dict：用于存储单独参数的重要性得分。
        # lora_A_matrix = model.lora_A
        # lora_B_matrix = model.lora_B
        # lora_A1_matrix = model.lora_A1
        # lora_B1_matrix = model.lora_B1
        for n, p in model.named_parameters():
            # if "lora_A" in n:
            #     rdim, hdim_a = p.shape
            #     ipt_score = self.calculate_score(n, metric="ipt")
            #     comb_ipt = torch.mean(ipt_score, dim=1, keepdim=False).view(-1, 1)
            #     name_mat = n.replace("lora_A", "%s")
            #     if name_mat not in combine_dict:
            #         combine_dict[name_mat] = [comb_ipt]
            #     else:
            #         combine_dict[name_mat].append(comb_ipt)
            # 对于每种类型的LoRA参数（"lora_A", "lora_A1", "lora_B", "lora_B1", "lora_E"），使用calculate_score函数计算重要性得分。这个函数可能基于不同的评价标准（如这里的"ipt"）来计算得分。
            if "lora_aa" in n:
                rdim, hdim_a = p.shape
                ipt_score = self.calculate_score(n, metric="ipt")
                comb_ipt = torch.mean(ipt_score, dim=0, keepdim=False).view(-1, 1)
                name_mat = n.replace("lora_aa", "%s")
                if name_mat not in combine_dict:
                    combine_dict[name_mat] = [comb_ipt]
                else:
                    combine_dict[name_mat].append(comb_ipt)
            if "lora_B" in n:
                hdim_b, rdim = p.shape
                ipt_score = self.calculate_score(n, metric="ipt")
                comb_ipt = torch.mean(ipt_score, dim=1, keepdim=False).view(-1, 1)
                name_mat = n.replace("lora_B", "%s")
                if name_mat not in combine_dict:
                    combine_dict[name_mat] = [comb_ipt]
                else:
                    combine_dict[name_mat].append(comb_ipt)
            if "lora_E" in n:
                ipt_score = self.calculate_score(n, p=p, metric="ipt")
                name_mat = n.replace("lora_E", "%s")
                singular_dict[name_mat] = ipt_score

        # Combine the importance scores
        all_is = []
        for name_mat in combine_dict:
            ipt_E = singular_dict[name_mat]
            ipt_AB = torch.cat(combine_dict[name_mat], dim=1)
            sum_ipt = self._combine_ipt(ipt_E, ipt_AB)
            name_E = name_mat % "lora_E"
            is_dict[name_E] = sum_ipt.view(-1, 1)
            all_is.append(sum_ipt.view(-1))

        mask_threshold = torch.kthvalue(torch.cat(all_is), (self.total_rank - curr_rank))[0].item()

        # Mask out unimportant singular values
        with torch.no_grad():
            curr_sum_rank = 0
            sum_param = 0
            for n, p in model.named_parameters():
                if "lora_E" in n:
                    p.data.masked_fill_(is_dict[n] <= mask_threshold, 0.0)
                    ranknum = (is_dict[n] > mask_threshold).sum().item()


        return mask_threshold

    # 初始化三个字典：
    # is_dict: 用于存储最终各个"Lora_E"子矩阵的重要性得分及对应的掩码阈值之后的结果。
    # combine_dict: 用于临时存储"Lora_A"和"Lora_B"子矩阵平均重要性得分的集合。
    # singular_dict: 存储"Lora_E"子矩阵的原始重要性得分。
    # 遍历模型的所有参数，对于包含特定前缀“lora_A”、“lora_B”和“lora_E”的参数，执行以下操作：
    # 计算"IPT"（Importance Score）得分，这里使用了自定义方法calculate_score。
    # 对于"Lora_A"和"Lora_B"子矩阵，计算每行或每列的平均IPT得分并将其添加到combine_dict中。
    # 对于"Lora_E"子矩阵，直接将其IPT得分存入singular_dict。
    # 合并重要性得分：
    # 对于combine_dict中的每个键（即"Lora_A"和"Lora_B"替换后的通用名），取出对应"E"部分的得分，将其与"Lora_A"和"Lora_B"合并得分连接在一起，然后调用 _combine_ipt 方法得到总和得分。
    # 将综合得分存储至is_dict中，并同时将所有综合得分扁平化后收集到列表all_is中。
    # 计算掩码阈值：
    # 使用torch.kthvalue函数找出所有综合得分中的第(total_rank - curr_rank)小的得分作为掩码阈值，这将决定哪些得分会被认为不重要而被掩蔽。
    # 掩蔽不重要的奇异值：
    # 在不需要梯度更新的情况下，遍历"Lora_E"子矩阵参数，并根据阈值掩蔽掉重要性得分较低的分量。
    # 记录当前秩数、参数数量以及参数所占的预算（可能是以某种计算单位衡量的参数规模）。
    # 如果存在TensorBoard记录器 (tb_writter) 并且满足日志记录间隔条件，则将相关统计量写入TensorBoard中。

    def update_and_mask(self, model, global_step):
        if global_step < self.total_step - self.final_warmup:
            # Update importance scores element-wise
            self.update_ipt(model)
            # do not update ipt during final fine-tuning
        # Budget schedule
        curr_rank, mask_ind = self.schedule_threshold(global_step)
        if mask_ind:
            # Mask to target budget
            mask_threshold = self.mask_to_target_rank(model, curr_rank)
        else:
            mask_threshold = None
        self._maybe_tb_writter_log(model)
        return curr_rank, mask_threshold

    # 首先检查当前全局步数是否小于总训练步数减去最后的微调阶段步数。若满足此条件：
    # 调用self.update_ipt(model)方法更新模型参数的重要性得分（importance-based parameter tuning，ipt）。
    # 调用预算调度方法self.schedule_threshold(global_step)，根据当前训练步数确定目标秩（curr_rank）以及是否需要应用掩码（mask_ind）。
    # 如果mask_ind为真（即需要应用掩码）：
    # 调用self.mask_to_target_rank(model, curr_rank)方法计算掩码阈值（mask_threshold），并据此对模型参数进行掩码操作，使模型参数满足目标秩的要求。
    # 如果不需要应用掩码（即mask_ind为假）：
    # 设置mask_threshold为None。
    # 调用内部方法self._maybe_tb_writter_log(model)，可能将相关信息写入TensorBoard日志。
    # 方法最后返回当前的目标秩（curr_rank）和掩码阈值（mask_threshold）。

    def _maybe_tb_writter_log(self, model):
        if self.tb_writter is not None and self.global_step % self.log_interval == 0:
            with torch.no_grad():
                regu_loss = []
                A = B = aa = bb = None
                for n, p in model.named_parameters():
                    if "lora_A" in n:
                        A = p.data.detach().clone().T  # 保存转置的A
                    elif "lora_B" in n:
                        B = p.data.detach().clone().T  # 保存转置的B
                    elif "lora_aa" in n:
                        aa = p.data.detach().clone().T  # 保存转置的aa
                    elif "lora_bb" in n:
                        bb = p.data.detach().clone().T  # 保存转置的bb
                    if "lora_A" in n or "lora_B" in n:
                        mat = p.data.detach().clone()
                        mat_cov = mat @ mat.T if "lora_A" in n else mat.T @ mat
                        I = torch.eye(*mat_cov.size(), out=torch.empty_like(mat_cov))
                        I.requires_grad = False
                        orth_regu = torch.norm(mat_cov - I, p="fro")
                        regu_loss.append(orth_regu.item())
                        self.tb_writter.add_scalar(
                            "Orth_regu_loss/%s" % n, orth_regu.item(), self.global_step
                        )
                    if "lora_aa" in n or "lora_bb" in n:
                        mat = p.data.detach().clone()
                        mat_cov = mat @ mat.T if "lora_A1" in n else mat.T @ mat
                        I = torch.eye(*mat_cov.size(), out=torch.empty_like(mat_cov))
                        I.requires_grad = False
                        orth_regu = torch.norm(mat_cov - I, p="fro")
                        regu_loss.append(orth_regu.item())
                        self.tb_writter.add_scalar(
                            "Orth_regu_loss/%s" % n, orth_regu.item(), self.global_step
                        )
                    if A is not None and B is not None and aa is not None and bb is not None:
                        AB = A @ B
                        aabb = aa @ bb
                        # 计算转置乘积的Frobenius范数来确定正交性
                        orth_regu = torch.norm(AB @ aabb, p="fro")
                        regu_loss.append(orth_regu.item())
                        # 将计算结果记录到TensorBoard
                        self.tb_writter.add_scalar(
                            "Orth_regu_loss", orth_regu.item(), self.global_step
                        )
                self.tb_writter.add_scalar(
                    "train/orth_regu_loss", sum(regu_loss) / len(regu_loss), self.global_step
                )
    # 首先检查是否有TensorBoard writer对象（self.tb_writter）并且当前全局步数（self.global_step）能被日志记录间隔（self.log_interval）整除。如果满足这两个条件，将继续执行日志记录操作。
    # 使用with torch.no_grad()语句块来确保在计算正交约束损失时不积累梯度。
    # 遍历模型的所有参数，查找那些名字中包含"lora_A"或"lora_B"的参数，并针对每个这样的参数执行以下操作：
    # 克隆并分离参数数据（mat = p.data.detach().clone()）以避免影响模型训练状态。
    # 计算参数矩阵与其转置的点积（mat_cov = mat @ mat.T 或 mat_cov = mat.T @ mat），形成协方差矩阵。
    # 创建单位矩阵 I，其大小与协方差矩阵相同，并设置 .requires_grad = False 来表明它不是一个需要计算梯度的张量。
    # 计算正交约束损失，即矩阵 mat_cov 与单位矩阵 I 之间的Frobenius范数之差（orth_regu = torch.norm(mat_cov-I, p="fro")）。
    # 将单个参数的正交约束损失值写入TensorBoard，路径为 "Orth_regu_loss/%s" 加上参数名。
    # 计算所有正交约束损失的平均值，并将平均损失值写入TensorBoard，路径为 "train/orth_regu_loss"。


def compute_orth_regu(model, regu_weight=0.1):
    # The function to compute orthongonal regularization for SVDLinear in `model`.
    regu_loss, num_param = 0., 0
    for n, p in model.named_parameters():
        if "lora_A" in n:
            A = p.data.detach().clone().T  # 保存转置的A
        elif "lora_B" in n:
            B = p.data.detach().clone().T  # 保存转置的B
        elif "lora_aa" in n:
            aa = p.data.detach().clone().T  # 保存转置的aa
        elif "lora_bb" in n:
            bb = p.data.detach().clone().T  # 保存转置的bb
        if "lora_A" in n or "lora_B" in n:
            para_cov = p @ p.T if "lora_A" in n else p.T @ p
            I = torch.eye(*para_cov.size(), out=torch.empty_like(para_cov))
            I.requires_grad = False
            regu_loss += torch.norm(para_cov - I, p="fro")
            num_param += 1
        if "lora_aa" in n or "lora_bb" in n:
            para_cov = p @ p.T if "lora_aa" in n else p.T @ p
            I = torch.eye(*para_cov.size(), out=torch.empty_like(para_cov))
            I.requires_grad = False
            regu_loss += torch.norm(para_cov - I, p="fro")
            num_param += 1
        if A is not None and B is not None and aa is not None and bb is not None:
            AB = A @ B
            aabb = aa @ bb
            product = AB @ aabb
            I = torch.eye(*para_cov.size(), device=para_cov.device)
            regu_loss += torch.norm(para_cov - I, p="fro")
            num_param += 1

    return regu_weight * regu_loss / num_param