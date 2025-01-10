import numpy as np
import math

import torch
import torch.nn.functional as F


class GRR:
    def __init__(self, y_eps=np.inf):
        self.y_eps = y_eps

    def __call__(self, data, train_and_val=True):
        if not np.isinf(self.y_eps):
            perturb_mask = data.train_mask | data.val_mask if train_and_val else data.train_mask

            q = 1.0 / (math.exp(self.y_eps) + data.num_classes - 1)
            p = q * math.exp(self.y_eps)

            y = data.y[perturb_mask]
            y = F.one_hot(y.argmax(dim=1), num_classes=data.num_classes)
            pr_y = y * p + (1 - y) * q
            out = torch.multinomial(pr_y, num_samples=1)
            y_perturbed = F.one_hot(out.squeeze(), num_classes=data.num_classes)

            data.y[perturb_mask] = y_perturbed.float()

            data.P_Y = torch.ones(data.num_classes, data.num_classes, device=data.y.device) * q
            data.P_Y.fill_diagonal_(p)

        else:
            data.P_Y = torch.zeros(data.num_classes, data.num_classes, device=data.y.device)
            data.P_Y.fill_diagonal_(1.0)

        return data


class GRRFS:
    def __init__(self, x_eps=np.inf, m=None):
        self.eps = x_eps
        self.m = m

    @staticmethod
    def __one_feature(grr_index, x_tuple, p, q, gamma_i):
        # print(grr_index)
        # print(gamma_i)
        # print(x_tuple)
        x_mat = F.one_hot(x_tuple.to(torch.int64), num_classes=gamma_i)
        # print(x_mat)
        # print(p)
        # print(q)

        p_grr = x_mat * p + (1 - x_mat) * q
        pr_unif = 1/gamma_i
        p_unif = torch.full(x_mat.shape, fill_value=pr_unif)

        # if torch.isinf(p_grr).any() or torch.isnan(p_grr).any():
        #     # 处理这些无穷大或非法值
        #     # 或者通过将它们替换为合适的值来修复它们
        #     p_grr[torch.isinf(p_grr)] = p_grr[0].clone()
        #     p_grr[torch.isnan(p_grr)] = p_grr[0].clone()

        grr_response = torch.multinomial(p_grr, num_samples=1).squeeze().to(torch.float)
        unif_response = torch.multinomial(p_unif, num_samples=1).squeeze().to(torch.float)

        x_tuple[grr_index] = grr_response[grr_index]
        x_tuple[~grr_index] = unif_response[~grr_index]

        del x_mat, p_grr, p_unif

        return x_tuple

    def __call__(self, data):
        x = data.x

        n, d = x.shape
        # print(x.shape)

        gamma = torch.max(x, dim=1)[0] + 1

        data.gamma = gamma

        data.P_X = torch.empty((d, 2))

        self.m = d if self.m is None else self.m
        self.m = min(self.m, x.size(1))
        index = torch.rand_like(x).topk(self.m, dim=1).indices
        bool_grr_index = torch.full(x.shape, fill_value=0).to(torch.bool)
        bool_grr_index = bool_grr_index.scatter(1, index, True)

        for i in range(d):
            if not np.isinf(self.eps):
                # 因为lastfm最后一个gamma为1
                if gamma[i] != 2:
                    gamma[i] = 2
                # print(gamma[i])
                # print(i)
                p = math.exp(self.eps) / (math.exp(self.eps) + gamma[i] - 1)
            else:
                p = 1

            q = (1 - p) / (gamma[i] - 1)

            data.P_X[i][0] = p
            data.P_X[i][1] = q

            # #因为lastfm最后一个gamma为1
            # if gamma[i] != 2:
            #     gamma[i] = 2

            rnd_x_tuple = self.__one_feature(bool_grr_index[:, i], x[:, i], p, q, int(gamma[i]))

            data.x[:, i] = rnd_x_tuple.to(torch.float)



        del bool_grr_index

        return data

