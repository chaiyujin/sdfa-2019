import os
import cv2
import sys
import math
import torch
import numpy as np
import matplotlib
DEFAULT_FONT_SIZE = 12
matplotlib.rcParams.update({'font.size': DEFAULT_FONT_SIZE})

DEFAULT_CMAP = "viridis"
_title_height = None


def _get_title_height():
    """ return font height in figsize unit """
    global _title_height
    if _title_height is None:
        import matplotlib.pyplot as plt
        f = plt.figure(figsize=(1,1))  # must be 1,1
        r = f.canvas.get_renderer()
        t = plt.text(0.0, 0.0, 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz')
        bb = t.get_tightbbox(renderer=r)
        _title_height = (bb.height * 1.4) / f.bbox.bounds[-1]
        plt.close(f)
    return _title_height


class plot_item(dict):
    def __init__(
        self, item, title="",
        vmin=None, vmax=None,
        draw_fn=None,
        cmap=DEFAULT_CMAP,
        **kwargs
    ):
        def to_numpy(t):
            if torch.is_tensor(t):
                t = t.detach().cpu().numpy()
            return t
        x_length = 0
        y_length = 0
        if isinstance(item, (list, tuple)):
            # x, y
            item = list(to_numpy(x) for x in item)
            assert len(item) == 2
            for x in item:
                assert item.ndim == 1, "[plot_item]: given (x, y), should both be 1d."
            x_length = item[0].max() - item[0].min()
            y_length = 1
            if vmin is None:
                vmin = item[1].min()
            if vmax is None:
                vmax = item[1].max()
        else:
            item = to_numpy(item)
            if item.ndim == 1:
                x_length = len(item)
                y_length = 1
            elif 2 <= item.ndim <= 3:
                if item.ndim == 3:
                    if item.shape[0] in [1, 3, 4]:
                        item = item.transpose(1, 2, 0)
                    assert item.shape[2] in [1, 3, 4],\
                        "invalid image shape of '{}': {}".format(title, item.shape)
                    # into gray image
                    if item.shape[2] == 1:
                        item = np.tile(item, (1, 1, 3))
                x_length = item.shape[1]
                y_length = item.shape[0]
            else:
                raise NotImplementedError("[plot_item]: {} dim is not supported".format(item.ndim))
            if vmin is None:
                vmin = item.min()
            if vmax is None:
                vmax = item.max()
        # set key, val
        if draw_fn is None:
            draw_fn = plot_item.__default_draw
        assert x_length > 0 and y_length > 0
        assert vmin is not None
        assert vmax is not None
        plot_item.__check_draw_fn(draw_fn)
        # wrap title
        if len(title) == 0:
            title_lines = 0
        else:
            title_lines = len(title.split("\n"))
        super().__init__(
            item=item,
            title=title,
            title_lines=title_lines,
            vmin=vmin, vmax=vmax,
            x_length=x_length,
            y_length=y_length,
            draw_fn=draw_fn,
            cmap=cmap,
            **kwargs
        )

    def __getattr__(self, attr):
        if attr in self:
            return super().__getitem__(attr)
        else:
            return super().__getattr__(attr)

    def draw(self, ax, cax):
        # default settings here
        ax.tick_params(labelsize=DEFAULT_FONT_SIZE * 0.8)
        cax.tick_params(labelsize=DEFAULT_FONT_SIZE * 0.8)
        self.draw_fn(self, ax, cax)

    @staticmethod
    def __default_draw(self, ax, cax):
        import matplotlib.pyplot as plt
        if isinstance(self.item, (tuple, list)):
            ax.set_title(self.title)
            ax.set_xlim(0, self.x_length)
            ax.set_ylim(self.vmin, self.vmax)
            ax.plot(self.item[0], self.item[1])
            cax.axis('off')
        elif self.item.ndim == 1:
            if hasattr(self, "aligned_transcription") and (hasattr(self, "sample_rate") or hasattr(self, "sr")):
                # automaticly plot aligned text
                draw_fn_aligned_audio_transcription(self, ax, cax)
            elif self.get("labels") is not None and self.get("rectangle", False):
                draw_fn_rectangles(self, ax, cax)
            else:
                # normal plot
                ax.set_title(self.title)
                ax.set_xlim(0, self.x_length)
                if self.vmin != self.vmax:
                    ax.set_ylim(self.vmin, self.vmax)
                ax.plot(np.arange(0, self.x_length), self.item)
                cax.axis('off')
        elif self.item.ndim == 2:
            ax.set_title(self.title)
            im = ax.imshow(self.item, vmin=self.vmin, vmax=self.vmax, cmap=self.cmap, aspect='auto')
            plt.colorbar(im, cax=cax)
            ax.invert_yaxis()
            # draw index labels
            if self.get("index_labels") is not None:
                plot_item.__draw_index_labels(
                    self, ax, self.get('index_labels'),
                    ymin=0, ymax=self.item.shape[0]
                )
        elif self.item.ndim == 3:
            ax.set_title(self.title)
            ax.imshow(self.item)
            ax.axis('off')
            cax.axis('off')

    @staticmethod
    def __draw_index_labels(self, ax, labels, ymin, ymax):
        import matplotlib.lines as mlines
        assert len(labels) == self.x_length, f"data length is {self.x_length}, but {len(labels)} labels."
        line_color = self.get("line_color", "black")
        text_color = self.get("text_color", "white")
        each_index = self.get("each_index", False)
        # draw transcripts
        yheight = ymax - ymin
        yrange = [ymin, ymax]
        height_percent = 0.10
        ymin = ymin + height_percent * yheight / 10
        ymax = ymax - height_percent * yheight / 5
        delta = 1
        y_pos = ymin
        last_word = ""
        todo_text = []
        for i, the_word in enumerate(labels):
            start = int(i)
            if (not each_index) and len(the_word) and the_word == last_word:
                continue
            ax.add_line(mlines.Line2D([start, start], yrange, linewidth=1, linestyle="-", c=line_color, alpha=0.5))
            todo_text.append((start, y_pos, str(the_word)))
            y_pos += delta * height_percent * yheight
            if y_pos > ymax:
                y_pos = ymin
            # cache last word
            last_word = the_word
        for todo in todo_text:
            ax.text(*todo, fontsize=8, color=text_color)

    @staticmethod
    def __check_draw_fn(draw_fn):
        import inspect
        assert callable(draw_fn), "given 'draw_fn' is not callable"
        params = [k for k in inspect.signature(draw_fn).parameters]
        assert params == ["self", "ax", "cax"]


class plot_grid(object):
    def __init__(self, items):
        super().__init__()
        assert isinstance(items, (list, tuple))
        items = list(items)
        rows, cols = len(items), 1
        for r in range(rows):
            if isinstance(items[r], (list, tuple)):
                cols = max(cols, len(items[r]))
            else:
                items[r] = [items[r]]
            for item in items[r]:
                assert type(item) is plot_item, "given item {} is not 'plot_item'".format(type(item))
        # allocate grid
        self.__grid = []
        self.__max_xlen = 0
        self.__max_ylen = 0
        for r in range(rows):
            self.__grid.append([])
            for c in range(cols):
                if c < len(items[r]):
                    self.__grid[r].append(items[r][c])
                    self.__max_xlen = max(self.__max_xlen, items[r][c].x_length)
                    self.__max_ylen = max(self.__max_ylen, items[r][c].y_length)
                else:
                    self.__grid[r].append(None)

    def set_value_range(self, vmin, vmax, mode):
        assert mode in ["auto", "same"]
        _vmin = float('inf')
        _vmax = float('-inf')
        for row in self.__grid:
            for item in row:
                if item is None:
                    continue
                if vmin is not None:
                    item.vmin = vmin
                if vmax is not None:
                    item.vmax = vmax
                _vmin = min(_vmin, item.vmin)
                _vmax = max(_vmax, item.vmax)
        if mode == "same":
            for row in self.__grid:
                for item in row:
                    if item is None:
                        continue
                    item.vmin = _vmin
                    item.vmax = _vmax

    @property
    def rows(self):
        return len(self.__grid)

    @property
    def cols(self):
        return len(self.__grid[0])

    @property
    def max_xlen(self):
        return self.__max_xlen

    @property
    def max_ylen(self):
        return self.__max_ylen

    def __iter__(self):
        self.__r = 0
        self.__c = 0
        return self

    def __next__(self):
        if self.__r < self.rows and self.__c < self.cols:
            ret = self.__grid[self.__r][self.__c]
            self.__c += 1
            if self.__c >= self.cols:
                self.__r += 1
                self.__c = 0
            return ret
        else:
            raise StopIteration

    def __call__(self, r, c):
        return self.__grid[r][c]


def color_mapping(arr, vmin=None, vmax=None, cmap=DEFAULT_CMAP, flip_rows=False):
    import matplotlib.pyplot as plt
    arr = np.asarray(arr)
    assert arr.ndim == 2, "color_mapping() only work for 2d array"
    if vmin is None:
        vmin = np.min(arr)
    if vmax is None:
        vmax = np.max(arr)
    cm = plt.get_cmap(cmap)
    img_data = np.uint8(cm(np.clip((arr-vmin)/(vmax-vmin+1e-10), 0, 1))*255)
    if flip_rows:
        img_data = np.flip(img_data, axis=0)
    return img_data


def figure_to_numpy(fig):
    # save it to a numpy array.
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


def draw_figure(fig, file_path=None, show=False, onclick=None):
    import matplotlib.pyplot as plt
    if file_path is not None:
        dirname = os.path.dirname(file_path)
        if len(dirname) > 0:
            os.makedirs(dirname, exist_ok=True)
        if os.path.splitext(file_path)[1] != ".png":
            file_path += ".png"
        plt.savefig('{}'.format(file_path), format='png')
    if show:
        if onclick is not None:
            fig.canvas.mpl_connect("button_press_event", onclick)
        plt.show()
        data = None
    else:
        data = figure_to_numpy(fig)
    plt.close(fig)
    return data


def plot(
    *items, file_path=None, val_mode="auto", aspect="wide", suptitle="",
    vmin=None, vmax=None, fig_scaling=1, show=False, onclick=None, **kwargs
):
    import matplotlib.pyplot as plt
    # allocate grid
    grid = plot_grid(items)
    grid.set_value_range(vmin, vmax, mode=val_mode)
    # get title lines for each row
    title_lines = [
        max(grid(r,c).title_lines for c in range(grid.cols))
        for r in range(grid.rows)
    ]
    # alloocate fig and axes
    fig, axes = __allocate_figure(
        num_rows= grid.rows,
        num_cols= grid.cols,
        title_lines=title_lines,
        x_length=grid.max_xlen,
        y_length=grid.max_ylen,
        scaling=fig_scaling,
        aspect=aspect,
        **kwargs
    )

    for r in range(grid.rows):
        for c in range(grid.cols):
            ax, cax = axes[r][c]
            if grid(r, c) is None:
                ax.axis('off')
                cax.axis('off')
            else:
                grid(r, c).draw(ax, cax)
    plt.suptitle(suptitle)
    return draw_figure(fig, file_path=file_path, show=show, onclick=onclick)


def __allocate_figure(num_rows, num_cols, title_lines, x_length, y_length, scaling, aspect, **kwargs):
    import matplotlib.pyplot as plt
    img_h = 2.5
    if type(aspect) is str:
        assert aspect in ["auto", "wide"], "[plot]: 'aspect' should be 'auto' or 'wide' not {}".format(aspect)
        img_w = (
            max(img_h, min(img_h*4, (x_length*img_h/y_length)))
            if aspect == "auto" else
            img_h * (16.0 / 9.0)
        )
    else:
        assert isinstance(aspect, (float, int))
        img_w = img_h * aspect

    assert num_rows == len(title_lines)
    # scaling
    scaling = scaling or 1  # guard None or 0
    img_h *= scaling
    img_w *= scaling
    mar_h, mar_w = 0.2 * scaling, 0.5 * scaling
    gap_h, gap_w = 0.35 * scaling, 0.8 * scaling
    gap_b = 0.1 * scaling
    bar_w = 0.1 * scaling
    th = _get_title_height()  # get title height
    th_acc_from_btm = [0]
    for i in range(len(title_lines) - 1, 0, -1):
        last = th_acc_from_btm[-1]
        th_acc_from_btm.append(last + title_lines[i] * th)
    # print(th_acc_from_btm)

    # alloc
    fig_h = img_h * num_rows + gap_h * num_rows + mar_h * 2\
          + th * sum(title_lines)
    fig_w = img_w * num_cols + gap_w * (num_cols - 1) + mar_w * 2 + bar_w * num_cols + gap_b * num_cols
    img_hr, img_wr = img_h / fig_h, img_w / fig_w
    bar_hr, bar_wr = img_h / fig_h, bar_w / fig_w
    bar_sr = (img_w + gap_b) / fig_w
    fig = plt.figure(1, figsize=(fig_w, fig_h))
    # definitions for first group

    def bttm_ratio(row):
        row = num_rows - row - 1
        h = mar_h + gap_h + (img_h + gap_h) * row + th_acc_from_btm[row]
        return h / fig_h

    def left_ratio(col):
        w = mar_w + (img_w + gap_b + bar_w + gap_w) * col
        return w / fig_w

    axes = [
        [
            (
                plt.axes([left_ratio(c),          bttm_ratio(r), img_wr, img_hr]),
                plt.axes([left_ratio(c) + bar_sr, bttm_ratio(r), bar_wr, bar_hr])
            )
            for c in range(num_cols)
        ]
        for r in range(num_rows)
    ]

    return fig, axes


def draw_fn_aligned_audio_transcription(self: plot_item, ax, cax):
    import matplotlib.lines as mlines
    from ..forced_alignment.transcription import Transcription
    assert self.item.ndim == 1, "given item should be 1dim signal, not {}".format(self.item.ndim)
    assert hasattr(self, "aligned_transcription")
    assert hasattr(self, "sample_rate") or hasattr(self, "sr")
    sr = self.get("sample_rate", self.get("sr"))
    aligned: Transcription = self.get("aligned_transcription")
    assert isinstance(aligned, Transcription)
    # set title
    if len(self.title) == 0:
        title = aligned.transcript
    else:
        title = "{}: {}".format(self.title, aligned.transcript)
    ax.set_title(title)
    # draw audio
    ax.set_xlim(0, len(self.item))
    ax.set_ylim(self.vmin, self.vmax)
    ax.plot(np.arange(0, len(self.item)), self.item)
    # draw transcripts
    yrange = self.vmax - self.vmin
    height_percent = 0.15
    ymin = self.vmin + height_percent * yrange / 10
    ymax = self.vmax - height_percent * yrange / 5
    delta = 1
    y_pos = ymin
    for i, the_word in enumerate(aligned.words):
        word = "({}) {}".format(i, the_word.word)
        start = int(the_word.start * sr)
        # end   = int(the_word.end * sr)
        ax.add_line(mlines.Line2D([start, start], [self.vmin, self.vmax], linewidth=1, linestyle="-.", c='g'))
        # ax.add_line(mlines.Line2D([end, end],     [self.vmin, self.vmax], linewidth=1, linestyle="-.", c='g'))
        ax.text(start, y_pos, str(word))
        y_pos += delta * height_percent * yrange
        if y_pos > ymax:
            y_pos = ymin
    cax.axis('off')


def draw_fn_rectangles(self: plot_item, ax, cax):
    from matplotlib.patches import Rectangle
    labels = self.get("labels")
    assert self.item.ndim == 1
    assert labels is not None
    assert len(labels) == len(self.item)
    ax.set_title(self.title)
    ax.set_ylim((self.vmin, self.vmax))
    ax.set_xlim((-0.5, len(labels)-0.5))
    ax.set_xticks(list(range(len(labels))))
    ax.set_xticklabels(labels)
    # Create a Rectangle patch
    for i, val in enumerate(self.item):
        xy = (-0.4 + i, self.vmin)
        height = val - self.vmin
        rect = Rectangle(xy, 0.8, height)
        ax.add_patch(rect)
        ax.text(-0.3+i, self.vmin, "{:.2f}".format(val), fontsize=8)
    cax.axis('off')
