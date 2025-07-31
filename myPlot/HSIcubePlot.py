# -*- coding: utf-8 -*-
import time

import spectral
import numpy as np
import wx
from utils.dataset import load_mat_hsi


if __name__ == "__main__":

    dataset = r"F:\KDL\pycharmProjects\MyHSIC2\datasets"


    image, gt, labels = load_mat_hsi("pu", dataset, n_components=60)
    # 启动应用并设置白色背景
    app = wx.App()
    spectral.settings.WX_GL_DEPTH_SIZE = 16
    # 关键修改：添加 background=(1,1,1) 参数
    # spectral.view_cube(image, bands=[57, 80, 30], background=(1, 1, 1))
    spectral.view_cube(image, bands=[53, 30, 17], background=(1, 1, 1))
    app.MainLoop()
