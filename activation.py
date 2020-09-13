import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class Activate(nn.Module):

    def __init__(self, act_id, init=1.0):

        super(Activate, self).__init__()

        if act_id == -1:
            self.act_id = -1
            self.weight = Parameter(torch.Tensor(8).fill_(init))
            # self.weight = Parameter(torch.Tensor(8))
            # self.weight.data.uniform_(0, 2)
        if act_id == -2:
            self.act_id = -2
            self.weight = Parameter(torch.Tensor(8).fill_(init))
            # self.weight = Parameter(torch.Tensor(8))
            # self.weight.data.uniform_(0, 2)
        if act_id == -3:
            self.act_id = -3
            # self.weight = Parameter(torch.Tensor(8).fill_(init))
            self.weight = Parameter(torch.Tensor(8))
            self.weight.data.uniform_(0, 2)
        elif act_id == -4:
            self.act_id = -4
            # self.weight = Parameter(torch.Tensor(4).fill_(init))
            self.weight = Parameter(torch.Tensor(4))
            self.weight.data.uniform_(0, 2)
        elif act_id == -5:
            self.act_id = -5
            self.weight = Parameter(torch.Tensor(4).fill_(init))
            # self.weight = Parameter(torch.Tensor(4))
            # self.weight.data.uniform_(0, 2)
        elif act_id == -6:
            self.act_id = -6
            self.weight = Parameter(torch.Tensor(4).fill_(init))
            # self.weight = Parameter(torch.Tensor(4))
            # self.weight.data.uniform_(0.125, 0.35)
        elif act_id == -7:
            self.act_id = -7
            self.weight = Parameter(torch.Tensor(4).fill_(init))
            # self.weight = Parameter(torch.Tensor(4))
            # self.weight.data.uniform_(0.125, 0.35)
        elif act_id == -8:
            self.act_id = -8
            self.weight = Parameter(torch.Tensor(4).fill_(init))
            # self.weight = Parameter(torch.Tensor(4))
            # self.weight.data.uniform_(0.125, 0.35)
        elif act_id == -9:
            self.act_id = -9
            self.weight = Parameter(torch.Tensor(8).fill_(init))
            # self.weight = Parameter(torch.Tensor(4))
            # self.weight.data.uniform_(0.125, 0.35)
        elif act_id == -10:
            self.act_id = -10
            self.weight = Parameter(torch.Tensor(4).fill_(init))
            # self.weight = Parameter(torch.Tensor(4))
            # self.weight.data.uniform_(0.125, 0.35)
        elif act_id == 1:
            self.act_id = 1
            self.activate = nn.ReLU()
        elif act_id == 2:
            self.act_id = 2
            self.activate = nn.LeakyReLU()
        elif act_id == 3:
            self.act_id = 3
            self.activate = nn.PReLU()
        elif act_id == 4:
            self.act_id = 4
            self.activate = nn.RReLU()
        elif act_id == 5:
            self.act_id = 5
            self.activate = nn.ELU()
        elif act_id == 6:
            self.act_id = 6
            self.activate = nn.SELU()
        elif act_id == 7:
            self.act_id = 7
            self.activate = nn.Sigmoid()
        elif act_id == 8:
            self.act_id = 8
            self.activate = nn.ReLU6()
        elif act_id == 91:
            self.act_id = 91
            self.weight = Parameter(torch.Tensor(6))
            self.weight.data.uniform_(0, 2)
        elif act_id == 92:
            self.act_id = 92
            self.weight = Parameter(torch.Tensor(6))
            self.weight.data.uniform_(0, 2)
        elif act_id == 93:
            self.act_id = 93
            self.weight = Parameter(torch.Tensor(6))
            self.weight.data.uniform_(0, 2)

        assert (self.act_id == act_id)

    def forward(self, x_input):

        if self.act_id == -1:
            # B C W H
            x_input = torch.where((x_input < -7), x_input * self.weight[0],
                                  torch.where((-7 <= x_input) & (x_input < -3), x_input * self.weight[1],
                                              torch.where((-3 <= x_input) & (x_input < -1), x_input * self.weight[2],
                                                          torch.where((-1 <= x_input) & (x_input < 0),
                                                                      x_input * self.weight[3],
                                                                      torch.where((0 <= x_input) & (x_input < 1),
                                                                                  x_input * self.weight[4],
                                                                                  torch.where(
                                                                                      (1 <= x_input) & (x_input < 3),
                                                                                      x_input * self.weight[5],
                                                                                      torch.where((3 <= x_input) & (
                                                                                                  x_input < 7),
                                                                                                  x_input * self.weight[
                                                                                                      6],
                                                                                                  x_input * self.weight[
                                                                                                      7])))))))
            return x_input
        elif self.act_id == -2:
            # B C W H
            x_input = torch.where((x_input < -15), x_input * self.weight[0],
                                  torch.where((-15 <= x_input) & (x_input < -7), x_input * self.weight[1],
                                              torch.where((-7 <= x_input) & (x_input < -3), x_input * self.weight[2],
                                                          torch.where((-3 <= x_input) & (x_input < 0),
                                                                      x_input * self.weight[3],
                                                                      torch.where((0 <= x_input) & (x_input < 3),
                                                                                  x_input * self.weight[4],
                                                                                  torch.where(
                                                                                      (3 <= x_input) & (x_input < 7),
                                                                                      x_input * self.weight[5],
                                                                                      torch.where((7 <= x_input) & (
                                                                                                  x_input < 15),
                                                                                                  x_input * self.weight[
                                                                                                      6],
                                                                                                  x_input * self.weight[
                                                                                                      7])))))))
            return x_input
        elif self.act_id == -3:
            # B C W H
            x_input = torch.where((x_input < -3), x_input * self.weight[0],
                                  torch.where((-3 <= x_input) & (x_input < -2), x_input * self.weight[1],
                                              torch.where((-2 <= x_input) & (x_input < -1), x_input * self.weight[2],
                                                          torch.where((-1 <= x_input) & (x_input < 0),
                                                                      x_input * self.weight[3],
                                                                      torch.where((0 <= x_input) & (x_input < 1),
                                                                                  x_input * self.weight[4],
                                                                                  torch.where(
                                                                                      (1 <= x_input) & (x_input < 2),
                                                                                      x_input * self.weight[5],
                                                                                      torch.where((2 <= x_input) & (
                                                                                                  x_input < 3),
                                                                                                  x_input * self.weight[
                                                                                                      6],
                                                                                                  x_input * self.weight[
                                                                                                      7])))))))
            return x_input
        elif self.act_id == -4:
            # -7, -3, -1
            x_input = torch.where((x_input < -7), x_input * self.weight[0],
                                  torch.where((-7 <= x_input) & (x_input < -3), x_input * self.weight[1],
                                              torch.where((-3 <= x_input) & (x_input < -1), x_input * self.weight[2],
                                                          torch.where((-1 <= x_input) & (x_input < 0),
                                                                      x_input * self.weight[3],
                                                                      x_input))))
            return x_input
        elif self.act_id == -5:
            # -3, -2, -1
            x_input = torch.where((x_input < -3), x_input * self.weight[0],
                                  torch.where((-3 <= x_input) & (x_input < -2), x_input * self.weight[1],
                                              torch.where((-2 <= x_input) & (x_input < -1), x_input * self.weight[2],
                                                          torch.where((-1 <= x_input) & (x_input < 0),
                                                                      x_input * self.weight[3],
                                                                      x_input))))
            return x_input
        elif self.act_id == -6:
            # -15, -7, -3
            x_input = torch.where((x_input < -15), x_input * self.weight[0],
                                  torch.where((-15 <= x_input) & (x_input < -7), x_input * self.weight[1],
                                              torch.where((-7 <= x_input) & (x_input < -3), x_input * self.weight[2],
                                                          torch.where((-3 <= x_input) & (x_input < 0),
                                                                      x_input * self.weight[3],
                                                                      x_input))))
            return x_input
        elif self.act_id == -7:
            # -31, -15, -7
            x_input = torch.where((x_input < -31), x_input * self.weight[0],
                                  torch.where((-31 <= x_input) & (x_input < -15), x_input * self.weight[1],
                                              torch.where((-15 <= x_input) & (x_input < -7), x_input * self.weight[2],
                                                          torch.where((-7 <= x_input) & (x_input < 0),
                                                                      x_input * self.weight[3],
                                                                      x_input))))
            return x_input
        elif self.act_id == -8:
            # -63, -31, -15
            x_input = torch.where((x_input < -63), x_input * self.weight[0],
                                  torch.where((-63 <= x_input) & (x_input < -31), x_input * self.weight[1],
                                              torch.where((-31 <= x_input) & (x_input < -15), x_input * self.weight[2],
                                                          torch.where((-15 <= x_input) & (x_input < 0),
                                                                      x_input * self.weight[3],
                                                                      x_input))))
            return x_input
        elif self.act_id == -9:
            # -40, -31, -22, -15, -7, -3
            x_input = torch.where((x_input < -40), x_input * self.weight[0],
                                  torch.where((-40 <= x_input) & (x_input < -31), x_input * self.weight[1],
                                              torch.where((-31 <= x_input) & (x_input < -22), x_input * self.weight[2],
                                                          torch.where((-22 <= x_input) & (x_input < -15),
                                                                      x_input * self.weight[3],
                                                                      torch.where((-15 <= x_input) & (x_input < -7),
                                                                                  x_input * self.weight[4],
                                                                                  torch.where(
                                                                                      (-7 <= x_input) & (x_input < -3),
                                                                                      x_input * self.weight[5],
                                                                                      torch.where((-3 <= x_input) & (
                                                                                                  x_input < 0),
                                                                                                  x_input * self.weight[
                                                                                                      6],
                                                                                                  x_input * self.weight[
                                                                                                      7])))))))
            return x_input
        elif self.act_id == -10:
            # -45, -30, -15
            x_input = torch.where((x_input < -45), x_input * self.weight[0],
                                  torch.where((-45 <= x_input) & (x_input < -30), x_input * self.weight[1],
                                              torch.where((-30 <= x_input) & (x_input < -15), x_input * self.weight[2],
                                                          torch.where((-15 <= x_input) & (x_input < 0),
                                                                      x_input * self.weight[3],
                                                                      x_input))))
            return x_input
        elif self.act_id == 7:
            return x_input * self.activate(x_input)
        elif self.act_id == 91:
            # B C W H
            x_input = torch.where((x_input < -7),
                                  -1 + self.weight[0] * (-3 + 1) + self.weight[1] * (-7 + 3) + self.weight[2] * (
                                              x_input + 7),
                                  torch.where((-7 <= x_input) & (x_input < -3),
                                              -1 + self.weight[0] * (-3 + 1) + self.weight[1] * (x_input + 3),
                                              torch.where((-3 <= x_input) & (x_input < -1),
                                                          -1 + self.weight[0] * (x_input + 1),
                                                          torch.where((-1 <= x_input) & (x_input < 1), x_input,
                                                                      torch.where((1 <= x_input) & (x_input < 3),
                                                                                  1 + self.weight[3] * (x_input - 1),
                                                                                  torch.where(
                                                                                      (3 <= x_input) & (x_input < 7),
                                                                                      1 + self.weight[3] * (3 - 1) +
                                                                                      self.weight[4] * (x_input - 3),
                                                                                      1 + self.weight[3] * (3 - 1) +
                                                                                      self.weight[4] * (7 - 3) +
                                                                                      self.weight[5] * (
                                                                                                  x_input - 7)))))))
            return x_input
        elif self.act_id == 92:
            # B C W H
            x_input = torch.where((x_input < -15),
                                  -3 + self.weight[0] * (-7 + 3) + self.weight[1] * (-15 + 7) + self.weight[2] * (
                                              x_input + 15),
                                  torch.where((-15 <= x_input) & (x_input < -7),
                                              -3 + self.weight[0] * (-7 + 3) + self.weight[1] * (x_input + 7),
                                              torch.where((-7 <= x_input) & (x_input < -3),
                                                          -3 + self.weight[0] * (x_input + 3),
                                                          torch.where((-3 <= x_input) & (x_input < 3), x_input,
                                                                      torch.where((3 <= x_input) & (x_input < 7),
                                                                                  3 + self.weight[3] * (x_input - 3),
                                                                                  torch.where(
                                                                                      (7 <= x_input) & (x_input < 15),
                                                                                      1 + self.weight[3] * (7 - 3) +
                                                                                      self.weight[4] * (x_input - 7),
                                                                                      1 + self.weight[3] * (7 - 3) +
                                                                                      self.weight[4] * (15 - 7) +
                                                                                      self.weight[5] * (
                                                                                                  x_input - 15)))))))
            return x_input
        elif self.act_id == 93:
            # B C W H
            x_input = torch.where((x_input < -3),
                                  -1 + self.weight[0] * (-2 + 1) + self.weight[1] * (-3 + 2) + self.weight[2] * (
                                              x_input + 3),
                                  torch.where((-3 <= x_input) & (x_input < -2),
                                              -1 + self.weight[0] * (-2 + 1) + self.weight[1] * (x_input + 2),
                                              torch.where((-2 <= x_input) & (x_input < -1),
                                                          -1 + self.weight[0] * (x_input + 1),
                                                          torch.where((-1 <= x_input) & (x_input < 1), x_input,
                                                                      torch.where((1 <= x_input) & (x_input < 2),
                                                                                  1 + self.weight[3] * (x_input - 1),
                                                                                  torch.where(
                                                                                      (2 <= x_input) & (x_input < 3),
                                                                                      1 + self.weight[3] * (2 - 1) +
                                                                                      self.weight[4] * (x_input - 2),
                                                                                      1 + self.weight[3] * (2 - 1) +
                                                                                      self.weight[4] * (3 - 2) +
                                                                                      self.weight[5] * (
                                                                                                  x_input - 3)))))))
            return x_input
        else:
            return self.activate(x_input)