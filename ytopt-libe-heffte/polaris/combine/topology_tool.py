# encoding: utf-8
import argparse
from copy import deepcopy as dcpy
import typing

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.widgets

def build():
    prs = argparse.ArgumentParser()

    fft = prs.add_argument_group("FFT parameters (X/Y/Z dimension of source and target FFT; # MPI Processes for FFT)")
    fft.add_argument('-f1', '--f1-size', dest='f1', type=int, default=None, help="Default size for unspecified dimensions of FFT #1")
    fft.add_argument('-f1x', '--f1-x-size', dest='f1x', type=int, default=None, help="X-dimension size for FFT #1 (default: --f1-size value)")
    fft.add_argument('-f1y', '--f1-y-size', dest='f1y', type=int, default=None, help="Y-dimension size for FFT #1 (default: --f1-size value)")
    fft.add_argument('-f1z', '--f1-z-size', dest='f1z', type=int, default=None, help="Z-dimension size for FFT #1 (default: --f1-size value)")
    fft.add_argument('-p1', '--procs-1', dest='p1', type=int, required=True, help="Number of MPI processes for FFT #1")
    fft.add_argument('-f2', '--f2-size', dest='f2', type=int, default=None, help="Default size for unspecified dimensions of FFT #2")
    fft.add_argument('-f2x', '--f2-x-size', dest='f2x', type=int, default=None, help="X-dimension size for FFT #2 (default: --f2-size value)")
    fft.add_argument('-f2y', '--f2-y-size', dest='f2y', type=int, default=None, help="Y-dimension size for FFT #2 (default: --f2-size value)")
    fft.add_argument('-f2z', '--f2-z-size', dest='f2z', type=int, default=None, help="Z-dimension size for FFT #2 (default: --f2-size value)")
    fft.add_argument('-p2', '--procs-2', dest='p2', type=int, required=True, help="Number of MPI processes for FFT #2")

    control = prs.add_argument_group("Control program behavior")
    callbacks = list(get_callbacks().keys())
    control.add_argument('--callback', choices=callbacks, default=callbacks[0], help="Determines how mappings are performed")

    plot = prs.add_argument_group("Plotting parameters (set of colors must be unique!)")
    plot.add_argument("-dc", "--default-color", dest='dc', default="y", help="Color for default configuration highlights (default: %(default)s)")
    plot.add_argument("-hc", "--highlight-color", dest='hc', default="red", help="Color for transfer configuration highlights (default: %(default)s)")
    plot.add_argument("-uc", "--unused-color", dest='uc', default="black", help="Color for inactive configurations (default: %(default)s)")

    return prs

def parse(args=None, prs=None):
    if prs is None:
        prs = build()
    if args is None:
        args = prs.parse_args()
    # Require all F1 and F2 X/Y/Z to be specified, and set defaults appropriately
    dims = ['x','y','z']
    unspecified_f1 = [getattr(args,f"f1{dim}") is None for dim in dims]
    if any(unspecified_f1) and args.f1 is None:
        raise ValueError("Cannot form fully-specified X/Y/Z dimensions with given arguments to F1\n"+\
                         "Define --f1-[xyz]-size and fill any unspecified gaps with --f1-size")
    for value, dim in zip(unspecified_f1, dims):
        if value:
            setattr(args, f"f1{dim}", args.f1)
    unspecified_f2 = [getattr(args,f"f2{dim}") is None for dim in dims]
    if any(unspecified_f2) and args.f2 is None:
        raise ValueError("Cannot form fully-specified X/Y/Z dimensions with given arguments to F2\n"+\
                         "Define --f2-[xyz]-size and fill any unspecified gaps with --f2-size")
    for value, dim in zip(unspecified_f2, dims):
        if value:
            setattr(args, f"f2{dim}", args.f2)

    # Set callback
    callback_dict = get_callbacks()
    args.callback = callback_dict[args.callback]

    # Check colors
    if len(set([args.dc, args.hc, args.uc])) != 3:
        ColorsRepeated = f"Highlight color {args.hc}, default color {args.dc} and unused color {args.uc} must all be different!"
        raise ValueError(ColorsRepeated)
    return args

def get_callbacks():
    gs = globals()
    return dict((_[len('callback_'):], gs[_]) for _ in gs.keys() if _.startswith('callback') and callable(gs[_]))

"""
    Namespace object designed to make it easy to define an API and adjust it,
    with validators to ensure you use it as intended

    Derivatives should subclass this, set the self.expect attribute to an appropriate dictionary:
        {attribute name:
            (string: human-description,
             validator: \
                lambda self, attrvalue: \
                    return True if attrvalue is appropriate
            )
        }
    If a particular order is required to validate attributes, also supply an attribute ordering in the self.validation_order attribute as an interable of string attribute names.
    Derivatives are guaranteed in this order or the default ordering of iteration over self.expect, so using an OrderedDict is an alternative to this specification

    Then define do_concretize() to set any implied attributes and complete initialization when all expected attributes are present.

    Usage of the class/subclass is expected to follow this pattern:
    class rn1(RegisteredNamespace):
        def __init__(self, **kwargs):
            self.expect = {
                'key1': ('first key', lambda self, x: hasattr(x,'__add__')),
                'key2': ('second key', lambda self, x: hasattr(x,'__add__') and type(x) is type(self.key1))
            }
            self.validation_order = ['key1','key2']
            super().__init__(**kwargs)
        def do_concretize(self):
            self.key3 = self.key1 + self.key2

    abc = rn1()
    abc.register(key1=11, key2=22)
    abc.concretize()
    print(abc.key3) # 33
"""
class RegisteredNamespace:
    is_concrete = False
    validation_order = None
    def __init__(self, **kwargs):
        if not hasattr(self, 'expect'):
            self.expect = {
                'is_concrete':
                    ("Basic functionality",
                     lambda self, x: True
                    )
            }
        self.register(**kwargs)

    def __regStr__(self, other):
        if id(self) == id(other):
            return "--You Are Here--"
        elif isinstance(other, type(self)):
            return "--Another Registration Object--"
        elif isinstance(other, np.ndarray):
            return f"ndarray with Shape {other.shape}"
        else:
            return str(other)

    def __str__(self):
        ret = "EXPECT:\n"
        for attr in self.expect.keys():
            ret += "\t"+f"{attr}: "
            if not hasattr(self, attr):
                ret += "--Not Registered Yet--\n"
            else:
                value = getattr(self, attr)
                ret += self.__regStr__(value)+"\n"
        ret += "Additional:\n"
        for attr in self.__dict__.keys():
            if attr in self.expect.keys():
                continue
            ret += "\t"+f"{attr}: "
            value = getattr(self, attr)
            ret += self.__regStr__(value)+"\n"
        return ret

    def register(self, **kwargs):
        for (k,v) in kwargs.items():
            setattr(self, k, v)
        return self

    def check_concrete_ok(self):
        if not hasattr(self, 'expect') or not isinstance(self.expect, typing.Mapping):
            raise ValueError("self.expect must remain a mapping type for concretization")
        if not hasattr(self, 'validation_order') or (self.validation_order is not None and not isinstance(self.validation_order, typing.Iterable)):
            raise ValueError("self.validation_order must be None (default) or an iterable type")
        not_avail = []
        if self.validation_order is None:
            self.validation_order = list(self.expect.keys())
        for attr in self.validation_order:
            if not hasattr(self, attr):
                not_avil.append(attr)
        if len(not_avail) == 0:
            # Check that all lambdas are satisfied
            improper = []
            for key in self.validation_order:
                validator = self.expect[key][1]
                if not validator(self, getattr(self, key)):
                    improper.append(key)
            if len(improper) == 0:
                # All values present and validated -- OK
                self.is_concrete = True
                return
            else:
                improper = "\n".join([f"{i}: {self.expect[i][0]}" for i in improper])
                improper = "The following expected types did not pass validation:\n" + improper
                raise ValueError(improper)
        not_avail = "\n".join([f"{na}: {self.expect[na][0]}" for na in not_avail])
        not_avail = "The following expected attributes were not registered:\n" + not_avail
        raise ValueError(not_avail)

    def concretize(self):
        if not self.is_concrete:
            self.check_concrete_ok()
        self.do_concretize()

    def do_concretize(self):
        pass

class PlotData(RegisteredNamespace):
    def __init__(self, **kwargs):
        self.expect = {
        'topologies': ("Numpy Array of shape (*,3) with integer type, each entry representing a 3D topology",
                        lambda self, x: type(x) is np.ndarray and len(x.shape) == 2 and x.shape[1] == 3),
        'default': ("Default topology from the topologies array",
                        lambda self, x: type(x) in [np.ndarray, list] and len(x) == 3 and len(np.where((self.topologies == x).all(axis=1))[0]) == 1),
        'callback': ("Callback to update the data",
                        lambda self, x: callable(x)),
        'unused_color': ("Color for labels when not active (must be unique from other colors)",
                        lambda self, x: matplotlib.colors.is_color_like(x)),
        'default_color': ("Color for labels when indicating default label (must be unique from other colors)",
                        lambda self, x: matplotlib.colors.is_color_like(x)),
        'highlight_color': ("Color for labels when indicating active (must be unique from other colors)",
                        lambda self, x: matplotlib.colors.is_color_like(x)),
        }
        self.validation_order = ['topologies', 'default', 'callback', 'unused_color', 'default_color', 'highlight_color']
        super().__init__(**kwargs)

    def do_concretize(self):
        self.default_idx = np.where((self.topologies == self.default).all(axis=1))[0][0]
        self.coords = np.zeros((len(self.topologies),2))
        # OPTIONAL definition, but non-optional attribute name
        if not hasattr(self, 'other'):
            self.other = None
        # Late check on color uniqueness for better error message
        self.color_names = ['UNUSED', 'DEFAULT', 'HIGHLIGHT']
        self.color_values = [self.unused_color, self.default_color, self.highlight_color]
        if len(set(self.color_values)) != 3 or len(set([matplotlib.colors.to_hex(c) for c in self.color_values])) != 3:
            RepeatedColor = f"All colors must be unique: {dict((k,v) for (k,v) in zip(self.color_names, self.color_values))}"
            raise ValueError(RepeatedColor)
        for attr, value in zip(self.color_names, self.color_values):
            setattr(self, attr, value)
        # Form coordinates based on tree shape of the topologies
        counter, x_cond, ymax = 0, self.topologies[0][0], len(set([_[0] for _ in self.topologies]))
        for idx in np.arange(len(self.topologies)):
            if self.topologies[idx][0] != x_cond:
                x_cond = self.topologies[idx][0]
                counter = 0
                ymax -= 1
            self.coords[idx] = (counter, ymax)
            counter += 1
        self.colors = np.array([self.unused_color]*len(self.topologies))
        self.colors[self.default_idx] = self.DEFAULT
        # Transform labels to be more friendly to read in visual format
        self.labels = ["["+",".join([str(v) for v in l])+"]" for l in self.topologies]

class Player(FuncAnimation):
    def __init__(self, fig, ax, plotdata, pos=(0.2, 0.92)):
        self.i = 0
        self.container = plotdata
        self.min=0
        self.max=len(self.container.colors)-1
        self.runs = True
        self.forwards = True
        def wrapper(i):
            self.container.i = i
            self.i = self.container.callback(self.container)
            for t, c in zip(self.container.texts, self.container.colors):
                t.set(color=c)
            return self.i
        self.wrapper_func = wrapper
        import pdb
        pdb.set_trace()
        super().__init__(self, fig=fig, func=self.wrapper_func, frames=np.arange(self.min,self.max), save_count=self.max-self.min)
        self.fig = fig
        self.ax = ax
        self.setup(pos)
        # Initialize plot
        self.ax.plot(self.container.coords[:,0], self.container.coords[:,1], 'ko', alpha=0)
        self.container.register(texts=[])
        for i in range(len(self.container.labels)):
            t = self.ax.text(self.container.coords[i,0], self.container.coords[i,1], self.container.labels[i], color=self.container.colors[i], fontsize='x-small')
            self.container.texts.append(t)

    def step(self, amount=1):
        self.i += self.forwards-(not self.forwards)
        # Loop
        if self.i < self.min:
            self.i = self.max
        elif self.i > self.max:
            self.i = self.min

    def play(self):
        while self.runs:
            self.step()
            yield self.i

    def start(self):
        self.runs=True
        self.event_source.start()

    def stop(self, event=None):
        self.runs = False
        self.event_source.stop()

    def forward(self, event=None):
        self.forwards = True
        self.start()
    def backward(self, event=None):
        self.forwards = False
        self.start()
    def oneforward(self, event=None):
        self.forwards = True
        self.onestep()
    def onebackward(self, event=None):
        self.forwards = False
        self.onestep()

    def onestep(self):
        self.step()
        self.wrapper_func(self.i)
        self.fig.canvas.draw_idle()
        yield self.i

    def default_callback(self):
        self.container.colors[:] = self.container.UNUSED
        self.container.colors[self.container.default_idx] = self.container.DEFAULT
        # Reset self.step()
        self.i = self.container.default_idx
        self.fig.canvas.draw_idle()

    def get_highlighted(self):
        return np.where(self.container.colors != self.container.UNUSED)[0][0]/len(self.container.colors)

    def setup(self, pos):
        # One back
        onebackax = self.fig.add_axes([pos[0],pos[1], 0.1, 0.05])
        self.button_oneback = matplotlib.widgets.Button(onebackax, label=u'$\u29CF$')
        self.button_oneback.on_clicked(self.onebackward)
        # Back
        backax = self.fig.add_axes([pos[0]+0.11,pos[1], 0.1, 0.05])
        self.button_back = matplotlib.widgets.Button(backax, label=u'$\u25C0$')
        self.button_back.on_clicked(self.backward)
        # Stop
        stopax = self.fig.add_axes([pos[0]+0.22,pos[1],0.1,0.05])
        self.button_stop = matplotlib.widgets.Button(stopax, label=u'$\u25A0$')
        self.button_stop.on_clicked(self.stop)
        # Default
        defaultax = self.fig.add_axes([pos[0]+0.33,pos[1],0.1,0.05])
        self.button_default = matplotlib.widgets.Button(defaultax, label='Default')
        self.button_default.on_clicked(self.default_callback)
        # Forward
        forax = self.fig.add_axes([pos[0]+0.44,pos[1],0.1,0.05])
        self.button_forward = matplotlib.widgets.Button(forax, label=u'$\u25B6$')
        self.button_forward.on_clicked(self.forward)
        # One Forward
        oneforax = self.fig.add_axes([pos[0]+0.55,pos[1],0.1,0.05])
        self.button_oneforward = matplotlib.widgets.Button(oneforax, label=u'$\u29D0$')
        self.button_oneforward.on_clicked(self.oneforward)

# Minimum surface splitting solve is used as the default topology for FFT (picked by heFFTe when in-grid and/or out-grid topology == ' ')
def surface(fft_dims, grid):
    # Volume of FFT assigned to each process
    box_size = (np.asarray(fft_dims) / np.asarray(grid)).astype(int)
    # Sum of exchanged surface areas
    return (box_size * np.roll(box_size, -1)).sum()
def minSurfaceSplit3D(X, Y, Z, procs):
    fft_dims = (X, Y, Z)
    best_grid = (1, 1, procs)
    best_surface = surface(fft_dims, best_grid)
    topologies = []
    # Consider other topologies that utilize all ranks
    for i in range(1, procs+1):
        if procs % i == 0:
            remainder = int(procs / float(i))
            for j in range(1, remainder+1):
                candidate_grid = (i, j, int(remainder/j))
                if np.prod(candidate_grid) != procs:
                    continue
                topologies.append(candidate_grid)
                candidate_surface = surface(fft_dims, candidate_grid)
                if candidate_surface < best_surface:
                    best_surface = candidate_surface
                    best_grid = candidate_grid
    # Topologies are reversed such that the topology order is X-1-1 to 1-1-X
    # This matches previous version ordering
    return np.asarray(best_grid), np.asarray(list(reversed(topologies)))

def callback_proportional(container):
    # Unpack
    primary = container.primary
    secondary = container.secondary
    step = container.i
    if secondary is None:
        # Moving from default -- color change
        if primary.colors[primary.default_idx] == primary.default_color:
            primary.colors[primary.default_idx] == primary.highlight_color
        # Perform step
        primary.colors = np.roll(primary.colors, step)
        primary.forwards = step > 0
        primary.step()
        yield primary.i
    else:
        old_index = np.where(primary.colors != primary.unused_color)[0][0]
        primary.colors[old_index] = primary.unused_color
        new_index = int(secondary.get_highlighted() * len(primary.colors))
        primary.colors[new_index] = primary.highlight_color
        # Actual step may be different from indicated step value
        diff = new_index - old_index
        primary.forwards = diff > 0
        primary.step(amount=diff)
        yield primary.i

def main(args=None):
    args = parse(args)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), width_ratios=(1,2))
    # De-spine the axes
    for ax in (ax1, ax2):
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(axis='both', which='both',
                       bottom=False, top=False, left=False, right=False,
                       labelbottom=False, labeltop=False, labelleft=False, labelright=False)

    # Prepare data to hook into animators
    d1, t1 = minSurfaceSplit3D(args.f1x, args.f1y, args.f1z,
                               args.p1)
    plot1 = PlotData(default=d1, topologies=t1)
    d2, t2 = minSurfaceSplit3D(args.f2x, args.f2y, args.f2z,
                               args.p2)
    plot2 = PlotData(default=d2, topologies=t2)
    plotDataArgs = {'default_color': args.dc,
                    'unused_color': args.uc,
                    'highlight_color': args.hc,
                    'callback': args.callback,}
    plot1.register(primary=plot1, **plotDataArgs).concretize()
    plot2.register(primary=plot2, secondary=plot1, **plotDataArgs).concretize()

    anim1 = Player(fig, ax1, plot1)
    anim2 = Player(fig, ax2, plot2)

    plt.show()


if __name__ == '__main__':
    main()
