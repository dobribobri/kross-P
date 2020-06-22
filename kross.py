# dobri@bobri:~$ sudo apt-get install python3-tk
# dobri@bobri:~$ sudo apt-get install python3-pil.imagetk


import PIL.Image
from PIL import ImageTk
from tkinter import filedialog
from tkinter import *
import struct
import numpy as np
import math
import os
import dill
from scipy.stats.stats import pearsonr
import threading
from matplotlib import pyplot as plt
import random

WIDTH = 1500
HEIGHT = 3840


class ImageField(Frame):
    def __init__(self, master, _image_path: str, width_: int, height_: int):
        Frame.__init__(self, master=None)
        self.x = self.y = 0
        self.canvas = Canvas(self, cursor="cross", width=width_, height=height_)

        self.sbarv = Scrollbar(self, orient=VERTICAL)
        self.sbarh = Scrollbar(self, orient=HORIZONTAL)
        self.sbarv.config(command=self.canvas.yview)
        self.sbarh.config(command=self.canvas.xview)

        self.canvas.config(yscrollcommand=self.sbarv.set)
        self.canvas.config(xscrollcommand=self.sbarh.set)

        self.canvas.grid(row=0, column=0, sticky=N + S + E + W)
        self.sbarv.grid(row=0, column=1, stick=N + S)
        self.sbarh.grid(row=1, column=0, sticky=E + W)

        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_move_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

        self.rect = None

        self.start_x = None
        self.start_y = None
        self.curX = None
        self.curY = None

        self.im = PIL.Image.open(_image_path)
        self.w_im, self.h_im = self.im.size
        self.resize_coeff = width_ / self.w_im
        self.h_im_resized = int(self.resize_coeff * self.h_im)
        self.w_im_resized = int(self.resize_coeff * self.w_im)
        self.im = self.im.resize((self.w_im_resized, self.h_im_resized))
        self.canvas.config(scrollregion=(0, 0, self.w_im_resized, self.h_im_resized))
        self.tk_im = ImageTk.PhotoImage(self.im)
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_im)

        self.imf_connected = None

    def on_button_press(self, event):
        # save mouse drag start position
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)

        # create rectangle if not yet exist
        if not self.rect:
            self.rect = self.canvas.create_rectangle(self.x, self.y, 1, 1, outline='red')
        if self.imf_connected:
            self.imf_connected.start_x = self.start_x
            self.imf_connected.start_y = self.start_y

            if not self.imf_connected.rect:
                self.imf_connected.rect = \
                    self.imf_connected.canvas.create_rectangle(
                        self.imf_connected.x, self.imf_connected.y, 1, 1, outline='red')

    def on_move_press(self, event):
        self.curX = self.canvas.canvasx(event.x)
        self.curY = self.canvas.canvasy(event.y)
        if self.imf_connected:
            self.imf_connected.curX = self.canvas.canvasx(event.x)
            self.imf_connected.curY = self.canvas.canvasy(event.y)

        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
        if event.x > 0.9 * w:
            self.canvas.xview_scroll(1, 'units')
        elif event.x < 0.1 * w:
            self.canvas.xview_scroll(-1, 'units')
        if event.y > 0.9 * h:
            self.canvas.yview_scroll(1, 'units')
        elif event.y < 0.1 * h:
            self.canvas.yview_scroll(-1, 'units')
        if self.imf_connected:
            w, h = self.imf_connected.canvas.winfo_width(), self.imf_connected.canvas.winfo_height()
            if event.x > 0.9 * w:
                self.imf_connected.canvas.xview_scroll(1, 'units')
            elif event.x < 0.1 * w:
                self.imf_connected.canvas.xview_scroll(-1, 'units')
            if event.y > 0.9 * h:
                self.imf_connected.canvas.yview_scroll(1, 'units')
            elif event.y < 0.1 * h:
                self.imf_connected.canvas.yview_scroll(-1, 'units')

        if self.curX > self.w_im_resized:
            self.curX = self.w_im_resized
        if self.curY > self.h_im_resized:
            self.curY = self.h_im_resized
        if self.curX < 0:
            self.curX = self.start_x
            self.start_x = 0
        if self.curY < 0:
            self.curY = self.start_y
            self.start_y = 0
        if self.imf_connected:
            if self.imf_connected.curX > self.imf_connected.w_im_resized:
                self.imf_connected.curX = self.imf_connected.w_im_resized
            if self.imf_connected.curY > self.imf_connected.h_im_resized:
                self.imf_connected.curY = self.imf_connected.h_im_resized
            if self.imf_connected.curX < 0:
                self.imf_connected.curX = self.imf_connected.start_x
                self.imf_connected.start_x = 0
            if self.imf_connected.curY < 0:
                self.imf_connected.curY = self.imf_connected.start_y
                self.imf_connected.start_y = 0

        # expand rectangle as you drag the mouse
        self.canvas.coords(self.rect, self.start_x, self.start_y, self.curX, self.curY)
        if self.imf_connected:
            self.imf_connected.canvas.coords(self.imf_connected.rect,
                                             self.imf_connected.start_x, self.imf_connected.start_y,
                                             self.imf_connected.curX, self.imf_connected.curY)

    def on_button_release(self, event):
        if self.start_x > self.curX:
            self.start_x, self.curX = self.curX, self.start_x
        if self.start_y > self.curY:
            self.start_y, self.curY = self.curY, self.start_y
        if self.imf_connected:
            if self.imf_connected.start_x > self.imf_connected.curX:
                self.imf_connected.start_x, self.imf_connected.curX = \
                    self.imf_connected.curX, self.imf_connected.start_x
            if self.imf_connected.start_y > self.imf_connected.curY:
                self.imf_connected.start_y, self.imf_connected.curY = \
                    self.imf_connected.curY, self.imf_connected.start_y

    def connect(self, imf: 'ImageField'):
        self.imf_connected = imf


def read_cmp(path_: str, mode: str = 'cmp'):
    data = np.zeros((HEIGHT, WIDTH, 2), dtype=float)
    if os.path.exists('{}.dump'.format(path_)):
        with open('{}.dump'.format(path_), 'rb') as dump:
            data = dill.load(dump)
    else:
        with open(path_, 'rb') as file:
            if mode == 'cmp':
                for i in range(HEIGHT - 1, -1, -1):
                    for j in range(WIDTH):
                        re_ = file.read(4)
                        im_ = file.read(4)
                        data[i][j][0] = struct.unpack('f', re_)[0]
                        data[i][j][1] = struct.unpack('f', im_)[0]
            elif mode == 'flt':
                for i in range(HEIGHT - 1, -1, -1):
                    for j in range(WIDTH):
                        faza = file.read(4)
                        data[i][j][0] = struct.unpack('f', faza)[0]

    # print(data.shape)
    if not os.path.exists('{}.dump'.format(path_)):
        with open('{}.dump'.format(path_), 'wb') as dump:
            dill.dump(data, dump)
    return data


def cmp2intensity(cmp_data: np.ndarray, save_path_: str):
    data = []
    if os.path.exists(save_path_):
        with open(save_path_, 'rb') as dump:
            data = dill.load(dump)
    else:
        for i in range(HEIGHT):
            for j in range(WIDTH):
                data.append(math.sqrt(cmp_data[i][j][0] ** 2 + cmp_data[i][j][1] ** 2))

    if not os.path.exists(save_path_):
        with open(save_path_, 'wb') as dump:
            dill.dump(data, dump)
    return data


def cmp2img(cmp_data: np.ndarray, save_img_path_: str, check_existence=True):
    if os.path.exists(save_img_path_) and check_existence:
        return

    data = []
    for i in range(HEIGHT):
        for j in range(WIDTH):
            data.append(math.sqrt(cmp_data[i][j][0] ** 2 + cmp_data[i][j][1] ** 2))
    im = PIL.Image.new('L', (WIDTH, HEIGHT))
    im.putdata(data)
    im.save(fp=save_img_path_)
    return


def intensity2img(intensity_data, save_img_path_: str, check_existence=True):
    if os.path.exists(save_img_path_) and check_existence:
        return
    im = PIL.Image.new('L', (WIDTH, HEIGHT))
    im.putdata(intensity_data)
    im.save(fp=save_img_path_)
    return


def get_coords(imf_: ImageField):
    try:
        x0 = int(1 / imf_.resize_coeff * imf_.start_x)
        y0 = int(1 / imf_.resize_coeff * imf_.start_y)
        x1 = int(1 / imf_.resize_coeff * imf_.curX) - 1
        y1 = int(1 / imf_.resize_coeff * imf_.curY) - 1
    except TypeError:
        print('Выделите прямоугольную область')
        return [-1] * 4

    if x0 == 0:
        x0 += 3
    if y0 == 0:
        y0 += 3
    if x1 == WIDTH - 1:
        x1 -= 3
    if y1 == HEIGHT - 1:
        y1 -= 3

    return x0, y0, x1, y1


# def plot(imf_: ImageField, cmp1_: np.ndarray, cmp2_: np.ndarray, kernel: str = 'II', direction: str = 'X',
#          x_shift_: int = 0, y_shift_: int = 0):
#     mainmenu.entryconfig('Построить графики', state="disabled")
#     print('\n--------------------------------------------------')
#     t = threading.Thread(target=plot_, args=[imf_, cmp1_, cmp2_, kernel, direction, x_shift_, y_shift_])
#     t.setDaemon(True)
#     t.start()


def plot_(imf_: ImageField, cmp1_: np.ndarray, cmp2_: np.ndarray, kernel: str = 'II', direction: str = 'X',
          x_shift_: int = 0, y_shift_: int = 0):
    x0, y0, x1, y1 = get_coords(imf_)
    if x0 == -1:
        mainmenu.entryconfig('Построить графики', state="normal")
        return
    print('Выбран прямоугольник [({}, {}) -> ({}, {})]'.format(x0, y0, x1, y1))
    print('Сдвиг изображения II по оси X: {}'.format(x_shift_))
    print('Сдвиг изображения II по оси Y: {}'.format(y_shift_))

    try:
        _ = cmp2_[y1 - y_shift_][x1 - x_shift_][0]
    except IndexError:
        print('Невозможно вычислить.')
        mainmenu.entryconfig('Построить графики', state="normal")
        return
    if (y0 - y_shift_ < 0) or (x0 - x_shift_ < 0):
        print('Невозможно вычислить.')
        mainmenu.entryconfig('Построить графики', state="normal")
        return

    # plt.close()

    if kernel == 'II':
        if direction == 'X':
            print('Построение графика (корреляция II с I по X) Подождите....')

            cmp20 = cmp2_[y0-y_shift_:y1 + 1 - y_shift_, x0-x_shift_:x1 + 1 - x_shift_, 0]
            cmp21 = cmp2_[y0-y_shift_:y1 + 1 - y_shift_, x0-x_shift_:x1 + 1 - x_shift_, 1]
            I2 = np.sqrt(cmp20 ** 2 + cmp21 ** 2).flatten()

            _y_, _x_ = y1 + 1 - y0, x1 + 1 - x0

            R, P, X = [], [], []
            for b in range(0, WIDTH - _x_):
                cmp10 = cmp1_[y0:y1 + 1, b:b + _x_, 0]
                cmp11 = cmp1_[y0:y1 + 1, b:b + _x_, 1]
                I1 = np.sqrt(cmp10 ** 2 + cmp11 ** 2).flatten()
                r, p = pearsonr(I1, I2)
                R.append(r)
                P.append(p)
                X.append(-x0+b)

            plt.figure(num='[{}] Корреляция II с I по X. Сдвиг II отн. I: ({}, {})'.format(random.randint(1, 100000),
                                                                                           x_shift_, y_shift_))
            plt.xlabel('Ось X')
            plt.ylabel('Значения')
            plt.plot(X, R, label='Коэфф. корреляции', color='crimson', linestyle='-')
            plt.plot(X, P, label='P-value', color='darkblue', marker='*', linewidth=0, markersize='0.8')
            plt.legend(loc='best')

            # mainmenu.entryconfig('Построить графики', state="normal")
            # menu0.entryconfig('Корреляция II с I по X', state="disabled")
            # if lock.locked():
                # print('Пожалуйста, завершите работу с предыдущим графиком.')
            # lock.acquire()
            plt.show()
            # lock.release()
            # menu0.entryconfig('Корреляция II с I по X', state="normal")
        elif direction == 'Y':
            print('Построение графика (корреляция II с I по Y) Подождите....')

            cmp20 = cmp2_[y0 - y_shift_:y1 + 1 - y_shift_, x0 - x_shift_:x1 + 1 - x_shift_, 0]
            cmp21 = cmp2_[y0 - y_shift_:y1 + 1 - y_shift_, x0 - x_shift_:x1 + 1 - x_shift_, 1]
            I2 = np.sqrt(cmp20 ** 2 + cmp21 ** 2).flatten()

            _y_, _x_ = y1 + 1 - y0, x1 + 1 - x0

            R, P, Y = [], [], []
            for b in range(0, HEIGHT - _y_):
                cmp10 = cmp1_[b:b + _y_, x0:x1 + 1, 0]
                cmp11 = cmp1_[b:b + _y_, x0:x1 + 1, 1]
                I1 = np.sqrt(cmp10 ** 2 + cmp11 ** 2).flatten()
                r, p = pearsonr(I1, I2)
                R.append(r)
                P.append(p)
                Y.append(-y0 + b)

            plt.figure(num='[{}] Корреляция II с I по Y. Сдвиг II отн. I: ({}, {})'.format(random.randint(1, 100000),
                                                                                           x_shift_, y_shift_))
            plt.xlabel('Ось Y')
            plt.ylabel('Значения')
            plt.plot(Y, R, label='Коэфф. корреляции', color='crimson', linestyle='-')
            plt.plot(Y, P, label='P-value', color='darkblue', marker='*', linewidth=0, markersize='0.8')
            plt.legend(loc='best')

            # mainmenu.entryconfig('Построить графики', state="normal")
            # menu0.entryconfig('Корреляция II с I по Y', state="disabled")
            # if lock.locked():
                # print('Пожалуйста, завершите работу с предыдущим графиком.')
            # lock.acquire()
            plt.show()
            # lock.release()
            # menu0.entryconfig('Корреляция II с I по Y', state="normal")
    elif kernel == 'I':
        if direction == 'X':
            print('Построение графика (корреляция I с II по X) Подождите....')

            cmp10 = cmp1_[y0:y1 + 1, x0:x1 + 1, 0]
            cmp11 = cmp1_[y0:y1 + 1, x0:x1 + 1, 1]
            I1 = np.sqrt(cmp10 ** 2 + cmp11 ** 2).flatten()

            _y_, _x_ = y1 + 1 - y0, x1 + 1 - x0

            R, P, X = [], [], []
            for b in range(3, WIDTH - _x_ - 3):
                cmp20 = cmp2_[y0 - y_shift_:y1 + 1 - y_shift_, b - x_shift_:b + _x_ - x_shift_, 0]
                cmp21 = cmp2_[y0 - y_shift_:y1 + 1 - y_shift_, b - x_shift_:b + _x_ - x_shift_, 1]
                I2 = np.sqrt(cmp20 ** 2 + cmp21 ** 2).flatten()
                r, p = pearsonr(I1, I2)
                R.append(r)
                P.append(p)
                X.append(-x0+b)

            plt.figure(num='[{}] Корреляция I с II по X. Сдвиг II отн. I: ({}, {})'.format(random.randint(1, 100000),
                                                                                           x_shift_, y_shift_))
            plt.xlabel('Ось X')
            plt.ylabel('Значения')
            plt.plot(X, R, label='Коэфф. корреляции', color='crimson', linestyle='-')
            plt.plot(X, P, label='P-value', color='darkblue', marker='*', linewidth=0, markersize='0.8')
            plt.legend(loc='best')

            # mainmenu.entryconfig('Построить графики', state="normal")
            # menu0.entryconfig('Корреляция I с II по X', state="disabled")
            # if lock.locked():
                # print('Пожалуйста, завершите работу с предыдущим графиком.')
            # lock.acquire()
            plt.show()
            # lock.release()
            # menu0.entryconfig('Корреляция I с II по X', state="normal")
        elif direction == 'Y':
            print('Построение графика (корреляция I с II по Y) Подождите....')

            cmp10 = cmp1_[y0:y1 + 1, x0:x1 + 1, 0]
            cmp11 = cmp1_[y0:y1 + 1, x0:x1 + 1, 1]
            I1 = np.sqrt(cmp10 ** 2 + cmp11 ** 2).flatten()

            _y_, _x_ = y1 + 1 - y0, x1 + 1 - x0

            R, P, Y = [], [], []
            for b in range(3, HEIGHT - _y_ - 3):
                cmp20 = cmp2_[b - y_shift_:b + _y_ - y_shift_, x0 - x_shift_:x1 + 1 - x_shift_, 0]
                cmp21 = cmp2_[b - y_shift_:b + _y_ - y_shift_, x0 - x_shift_:x1 + 1 - x_shift_, 1]
                I2 = np.sqrt(cmp20 ** 2 + cmp21 ** 2).flatten()
                r, p = pearsonr(I1, I2)
                R.append(r)
                P.append(p)
                Y.append(-y0 + b)

            plt.figure(num='[{}] Корреляция I с II по Y. Сдвиг II отн. I: ({}, {})'.format(random.randint(1, 100000),
                                                                                           x_shift_, y_shift_))
            plt.xlabel('Ось Y')
            plt.ylabel('Значения')
            plt.plot(Y, R, label='Коэфф. корреляции', color='crimson', linestyle='-')
            plt.plot(Y, P, label='P-value', color='darkblue', marker='*', linewidth=0, markersize='0.8')
            plt.legend(loc='best')

            # mainmenu.entryconfig('Построить графики', state="normal")
            # menu0.entryconfig('Корреляция I с II по Y', state="disabled")
            # if lock.locked():
                # print('Пожалуйста, завершите работу с предыдущим графиком.')
            # lock.acquire()
            plt.show()
            # lock.release()
            # menu0.entryconfig('Корреляция I с II по Y', state="normal")
    elif kernel == 'both':
        if direction == 'X':
            print('Построение графика (корреляция II с I по X) Подождите....')

            cmp20 = cmp2_[y0 - y_shift_:y1 + 1 - y_shift_, x0 - x_shift_:x1 + 1 - x_shift_, 0]
            cmp21 = cmp2_[y0 - y_shift_:y1 + 1 - y_shift_, x0 - x_shift_:x1 + 1 - x_shift_, 1]
            I2 = np.sqrt(cmp20 ** 2 + cmp21 ** 2).flatten()

            _y_, _x_ = y1 + 1 - y0, x1 + 1 - x0

            R, P, X = [], [], []
            for b in range(0, WIDTH - _x_):
                cmp10 = cmp1_[y0:y1 + 1, b:b + _x_, 0]
                cmp11 = cmp1_[y0:y1 + 1, b:b + _x_, 1]
                I1 = np.sqrt(cmp10 ** 2 + cmp11 ** 2).flatten()
                r, p = pearsonr(I1, I2)
                R.append(r)
                P.append(p)
                X.append(-x0 + b)

            plt.figure(num='[{}] Корреляция II с I по X. Сдвиг II отн. I: ({}, {})'.format(random.randint(1, 100000),
                                                                                           x_shift_, y_shift_))
            plt.xlabel('Ось X')
            plt.ylabel('Значения')
            plt.plot(X, R, label='Коэфф. корреляции', color='crimson', linestyle='-')
            plt.plot(X, P, label='P-value', color='darkblue', marker='*', linewidth=0, markersize='0.8')
            plt.legend(loc='best')

            print('Построение графика (корреляция I с II по X) Подождите....')

            cmp10 = cmp1_[y0:y1 + 1, x0:x1 + 1, 0]
            cmp11 = cmp1_[y0:y1 + 1, x0:x1 + 1, 1]
            I1 = np.sqrt(cmp10 ** 2 + cmp11 ** 2).flatten()

            _y_, _x_ = y1 + 1 - y0, x1 + 1 - x0

            R, P, X = [], [], []
            for b in range(3, WIDTH - _x_ - 3):
                cmp20 = cmp2_[y0 - y_shift_:y1 + 1 - y_shift_, b - x_shift_:b + _x_ - x_shift_, 0]
                cmp21 = cmp2_[y0 - y_shift_:y1 + 1 - y_shift_, b - x_shift_:b + _x_ - x_shift_, 1]
                I2 = np.sqrt(cmp20 ** 2 + cmp21 ** 2).flatten()
                r, p = pearsonr(I1, I2)
                R.append(r)
                P.append(p)
                X.append(-x0 + b)

            plt.figure(num='[{}] Корреляция I с II по X. Сдвиг II отн. I: ({}, {})'.format(random.randint(1, 100000),
                                                                                           x_shift_, y_shift_))
            plt.xlabel('Ось X')
            plt.ylabel('Значения')
            plt.plot(X, R, label='Коэфф. корреляции', color='crimson', linestyle='-')
            plt.plot(X, P, label='P-value', color='darkblue', marker='*', linewidth=0, markersize='0.8')
            plt.legend(loc='best')

            # mainmenu.entryconfig('Построить графики', state="normal")
            # menu0.entryconfig('Оба графика по X', state="disabled")
            # if lock.locked():
                # print('Пожалуйста, завершите работу с предыдущим графиком.')
            # lock.acquire()
            plt.show()
            # lock.release()
            # menu0.entryconfig('Оба графика по X', state="normal")
        elif direction == 'Y':
            print('Построение графика (корреляция II с I по Y) Подождите....')

            cmp20 = cmp2_[y0 - y_shift_:y1 + 1 - y_shift_, x0 - x_shift_:x1 + 1 - x_shift_, 0]
            cmp21 = cmp2_[y0 - y_shift_:y1 + 1 - y_shift_, x0 - x_shift_:x1 + 1 - x_shift_, 1]
            I2 = np.sqrt(cmp20 ** 2 + cmp21 ** 2).flatten()

            _y_, _x_ = y1 + 1 - y0, x1 + 1 - x0

            R, P, Y = [], [], []
            for b in range(0, HEIGHT - _y_):
                cmp10 = cmp1_[b:b + _y_, x0:x1 + 1, 0]
                cmp11 = cmp1_[b:b + _y_, x0:x1 + 1, 1]
                I1 = np.sqrt(cmp10 ** 2 + cmp11 ** 2).flatten()
                r, p = pearsonr(I1, I2)
                R.append(r)
                P.append(p)
                Y.append(-y0 + b)

            plt.figure(num='[{}] Корреляция II с I по Y. Сдвиг II отн. I: ({}, {})'.format(random.randint(1, 100000),
                                                                                           x_shift_, y_shift_))
            plt.xlabel('Ось Y')
            plt.ylabel('Значения')
            plt.plot(Y, R, label='Коэфф. корреляции', color='crimson', linestyle='-')
            plt.plot(Y, P, label='P-value', color='darkblue', marker='*', linewidth=0, markersize='0.8')
            plt.legend(loc='best')

            print('Построение графика (корреляция I с II по Y) Подождите....')

            cmp10 = cmp1_[y0:y1 + 1, x0:x1 + 1, 0]
            cmp11 = cmp1_[y0:y1 + 1, x0:x1 + 1, 1]
            I1 = np.sqrt(cmp10 ** 2 + cmp11 ** 2).flatten()

            _y_, _x_ = y1 + 1 - y0, x1 + 1 - x0

            R, P, Y = [], [], []
            for b in range(3, HEIGHT - _y_ - 3):
                cmp20 = cmp2_[b - y_shift_:b + _y_ - y_shift_, x0 - x_shift_:x1 + 1 - x_shift_, 0]
                cmp21 = cmp2_[b - y_shift_:b + _y_ - y_shift_, x0 - x_shift_:x1 + 1 - x_shift_, 1]
                I2 = np.sqrt(cmp20 ** 2 + cmp21 ** 2).flatten()
                r, p = pearsonr(I1, I2)
                R.append(r)
                P.append(p)
                Y.append(-y0 + b)

            plt.figure(num='[{}] Корреляция I с II по Y. Сдвиг II отн. I: ({}, {})'.format(random.randint(1, 100000),
                                                                                           x_shift_, y_shift_))
            plt.xlabel('Ось Y')
            plt.ylabel('Значения')
            plt.plot(Y, R, label='Коэфф. корреляции', color='crimson', linestyle='-')
            plt.plot(Y, P, label='P-value', color='darkblue', marker='*', linewidth=0, markersize='0.8')
            plt.legend(loc='best')

            # mainmenu.entryconfig('Построить графики', state="normal")
            # menu0.entryconfig('Оба графика по Y', state="disabled")
            # if lock.locked():
            #     print('Пожалуйста, завершите работу с предыдущим графиком.')
            # lock.acquire()
            plt.show()
            # lock.release()
            # menu0.entryconfig('Оба графика по Y', state="normal")
    return


def compute(imf_: ImageField, cmp1_: np.ndarray, cmp2_: np.ndarray, x_shift_: int = 0, y_shift_: int = 0):
    b_compute['state'] = DISABLED
    print('\n--------------------------------------------------')
    t = threading.Thread(target=compute_, args=[imf_, cmp1_, cmp2_, x_shift_, y_shift_])
    t.setDaemon(True)
    t.start()


def compute_(imf_: ImageField, cmp1_: np.ndarray, cmp2_: np.ndarray, x_shift_: int = 0, y_shift_: int = 0):
    x0, y0, x1, y1 = get_coords(imf_)
    if x0 == -1:
        b_compute['state'] = NORMAL
        return
    print('Выбран прямоугольник [({}, {}) -> ({}, {})]'.format(x0, y0, x1, y1))
    print('Сдвиг изображения II по оси X: {}'.format(x_shift_))
    print('Сдвиг изображения II по оси Y: {}'.format(y_shift_))

    try:
        _ = cmp2_[y1 - y_shift_][x1 - x_shift_][0]
    except IndexError:
        print('Невозможно вычислить.')
        b_compute['state'] = NORMAL
        return
    if (y0 - y_shift_ < 0) or (x0 - x_shift_ < 0):
        print('Невозможно вычислить.')
        b_compute['state'] = NORMAL
        return

    cmp10 = cmp1_[y0:y1+1, x0:x1+1, 0]
    cmp11 = cmp1_[y0:y1+1, x0:x1+1, 1]
    cmp20 = cmp2_[y0-y_shift_:y1+1-y_shift_, x0-x_shift_:x1+1-x_shift_, 0]
    cmp21 = cmp2_[y0-y_shift_:y1+1-y_shift_, x0-x_shift_:x1+1-x_shift_, 1]
    I1 = np.sqrt(cmp10 ** 2 + cmp11 ** 2).flatten()
    I2 = np.sqrt(cmp20 ** 2 + cmp21 ** 2).flatten()

    print('Средняя интенсивность:')
    print('Изображение I\t-\t{:.5f}'.format(np.average(I1)))
    print('Изображение II\t-\t{:.5f}'.format(np.average(I2)))

    print('Корреляция:')
    r, p = pearsonr(I1, I2)
    print('Коэффициент корреляции\t-\t{:.5f}'.format(r))
    print('P-value\t-\t{:.5f}'.format(p))

    b_compute['state'] = NORMAL


imf_width = 300
imf_height = 430


if __name__ == "__main__":

    root = Tk()
    root.title('Kross/P [dev]')
    root.filename = filedialog.askopenfilename(initialdir=".", title="Выберите первое изображение",
                                               filetypes=(("CMP files (комплексная матрица РЛИ)", "*.cmp"),
                                                          ("FLT files (действительная матрица фазы)", "*.flt")))
    data_path_1 = root.filename
    if data_path_1 is '':
        exit(1)
    root.filename = filedialog.askopenfilename(initialdir=".", title="Выберите второе изображение",
                                               filetypes=(("CMP files (комплексная матрица РЛИ)", "*.cmp"),
                                                          ("FLT files (действительная матрица фазы)", "*.flt")))
    data_path_2 = root.filename
    if data_path_2 is '':
        exit(1)

    mode_1 = 'cmp'
    if data_path_1[len(data_path_1)-3:] == 'flt':
        mode_1 = 'flt'
    cmp_1 = read_cmp(data_path_1, mode=mode_1)
    lbl1 = re.split(r'[\\/]', data_path_1[:len(data_path_1)-4])[-1]
    image_path_1 = '{}.png'.format(data_path_1[:len(data_path_1)-4])
    cmp2img(cmp_1, image_path_1)

    mode_2 = 'cmp'
    if data_path_2[len(data_path_2)-3:] == 'flt':
        mode_2 = 'flt'
    cmp_2 = read_cmp(data_path_2, mode=mode_2)
    lbl2 = re.split(r'[\\/]', data_path_2[:len(data_path_2)-4])[-1]
    image_path_2 = '{}.png'.format(data_path_2[:len(data_path_2)-4])
    cmp2img(cmp_2, image_path_2)

    root.title('Kross/P [dev] - Изображения {} и {}'.format(lbl1, lbl2))
    root.geometry('{:.0f}x{:.0f}'.format(imf_width * 1.08 * 2, imf_height * 1.35))
    root.resizable(width=False, height=False)

    mainmenu = Menu(root)
    root.config(menu=mainmenu)
    # mainmenu.add_command(label='Построить графики')
    menu0 = Menu(mainmenu, tearoff=0)

    imf1 = ImageField(root, image_path_1, width_=imf_width, height_=imf_height)
    imf2 = ImageField(root, image_path_2, width_=imf_width, height_=imf_height)
    imf1.connect(imf2)
    imf2.connect(imf1)
    imf1.grid(row=0, column=0)
    imf2.grid(row=0, column=1, columnspan=2)

    # plt.ion()

    # lock = threading.Lock()
    # N = 0
    menu0.add_command(label="Корреляция II с I по X",
                      command=lambda imf_=imf1, cmp1_=cmp_1, cmp2_=cmp_2: plot_(imf_, cmp1_, cmp2_,
                                                                               kernel='II', direction='X',
                                                                               x_shift_=x_shift_scale.get(),
                                                                               y_shift_=y_shift_scale.get()
                                                                               ))
    menu0.add_command(label="Корреляция II с I по Y",
                      command=lambda imf_=imf1, cmp1_=cmp_1, cmp2_=cmp_2: plot_(imf_, cmp1_, cmp2_,
                                                                               kernel='II', direction='Y',
                                                                               x_shift_=x_shift_scale.get(),
                                                                               y_shift_=y_shift_scale.get()
                                                                               ))
    menu0.add_command(label="Корреляция I с II по X",
                      command=lambda imf_=imf1, cmp1_=cmp_1, cmp2_=cmp_2: plot_(imf_, cmp1_, cmp2_,
                                                                               kernel='I', direction='X',
                                                                               x_shift_=x_shift_scale.get(),
                                                                               y_shift_=y_shift_scale.get()
                                                                               ))
    menu0.add_command(label="Корреляция I с II по Y",
                      command=lambda imf_=imf1, cmp1_=cmp_1, cmp2_=cmp_2: plot_(imf_, cmp1_, cmp2_,
                                                                               kernel='I', direction='Y',
                                                                               x_shift_=x_shift_scale.get(),
                                                                               y_shift_=y_shift_scale.get()
                                                                               ))
    menu0.add_command(label="Оба графика по X",
                      command=lambda imf_=imf1, cmp1_=cmp_1, cmp2_=cmp_2: plot_(imf_, cmp1_, cmp2_,
                                                                               kernel='both', direction='X',
                                                                               x_shift_=x_shift_scale.get(),
                                                                               y_shift_=y_shift_scale.get()
                                                                               ))
    menu0.add_command(label="Оба графика по Y",
                      command=lambda imf_=imf1, cmp1_=cmp_1, cmp2_=cmp_2: plot_(imf_, cmp1_, cmp2_,
                                                                               kernel='both', direction='Y',
                                                                               x_shift_=x_shift_scale.get(),
                                                                               y_shift_=y_shift_scale.get()
                                                                               ))
    mainmenu.add_cascade(label="Построить графики", menu=menu0)

    x_shift_scale = Scale(root, orient=HORIZONTAL, from_=-3, to=3, tickinterval=3, resolution=1)
    x_shift_scale.grid(row=1, column=1)
    y_shift_scale = Scale(root, orient=VERTICAL, from_=-3, to=3, tickinterval=3, resolution=1)
    y_shift_scale.grid(row=1, column=2)

    b_compute = Button(text="Вычислить", width=20, height=1)
    command = (lambda imf_=imf1, cmp1_=cmp_1, cmp2_=cmp_2: compute(imf_, cmp1_, cmp2_,
                                                                   x_shift_=x_shift_scale.get(),
                                                                   y_shift_=y_shift_scale.get()))
    b_compute.config(command=command)
    b_compute.grid(row=1, column=0)

    root.mainloop()
