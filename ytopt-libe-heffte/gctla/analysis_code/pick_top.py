import pandas as pd, numpy as np
import argparse, matplotlib
import matplotlib.pyplot as plt, matplotlib.widgets as widgets
from matplotlib.animation import FuncAnimation

def build():
    prs = argparse.ArgumentParser()
    files = prs.add_argument_group("Files")
    files.add_argument("--output", default="picked.csv",
                        help="Output CSV destination (default: %(default)s)")
    files.add_argument("--inputs", nargs="+", required=True,
                        help="Input CSVs to pick top results from")
    picking = prs.add_argument_group("Selection")
    picking.add_argument("--column-name", default="FLOPS",
                        help="Column in input CSVs to use for objective value (default: %(default)s)")
    picking.add_argument("--bottom-up", action="store_true",
                        help="Pick best results from the lowest objective value rather than highest (default: %(default)s)")
    picking.add_argument("--quantile", type=float, default=None,
                        help="Identify a quantile for each loaded dataset (default: %(default)s)")
    return prs

def parse(args=None, prs=None):
    if prs is None:
        prs = build()
    if args is None:
        args = prs.parse_args()
    if args.quantile < 0 or args.quantile > 1:
        raise ValueError(f"Quantile must be [0-1] inclusive (received: {args.quantile})")
    return args

def load_all(args):
    csv_q, stat_q = [], []
    for fname in args.inputs:
        try:
            frame = pd.read_csv(fname)
            frame = frame.sort_values(by=[args.column_name]).reset_index(drop=True)
        except FileNotFoundError:
            print(f"Omitting input as unloadable (FileNotFound): {fname}")
            continue
        except KeyError:
            print(f"Ommitting input as unparseable (ValueError, column {args.column_name} may be unavailable): {fname}")
            continue
        csv_q.append(frame)
        framelen = len(frame)
        stats = {'records': framelen,
                 'min': frame.loc[0,args.column_name],
                 'max': frame.loc[framelen-1,args.column_name],
                 'mean': np.mean(frame[args.column_name])}
        if args.quantile is not None:
            stats['quantile'] = frame[args.column_name].quantile(args.quantile, interpolation='nearest')
        stat_q.append(stats)
    return csv_q, stat_q

class Player(FuncAnimation):
    def __init__(self, fig, axes, data, statistics, name, args, always_all_artists=False):
        # Copy and set attributes
        self.fig = fig
        self.axes = axes
        self.data = data
        self.stats = statistics
        self.args = args
        self.always_all_artists = always_all_artists
        # Set up a wrapper function to drive frames
        def wrapper(frame, *fargs):
            self.stop()
            return self.updateArtists()
        self.wrapper_func = wrapper
        # For some reason we get advanced a few frames on initialization
        # This + the function wrapping around mean that we should 'halt' after these frames on what frame 0 would be
        self.init_ID = self.frameID = 0 # self.stats['records']-3
        # Initialize plot
        self.line = self.axes.plot(self.data.index, self.data[self.args.column_name],
                                    'ko', alpha=0)
        self.points = self.axes.scatter(self.data.index, self.data[self.args.column_name],
                                    c=['k']*self.stats['records'], alpha=1)
        initial_data = self.data.loc[self.frameID,self.args.column_name]
        self.text = self.axes.text(0,initial_data,f"IDX={self.init_ID}\nObjective: {initial_data}",color='red')
        self.axes.set_xlabel("Use buttons then OK || input x-axis # in text field (negative to omit)")
        self.axes.set_ylabel("Objective")
        self.axes.set_title(name)
        # Initial coloring
        self.activecolor = [1,0,0,0]
        self.inactivecolor = [0,0,0,0.5]
        colors = self.points.get_fc()
        colors[self.frameID] = self.activecolor
        colors[~self.frameID] = self.inactivecolor
        self.points.set_fc(colors)
        if self.args.quantile is not None:
            index = np.where(self.data[self.args.column_name] == self.stats['quantile'])[0]
            assert len(index) > 0
            self.axes.vlines(index[0], self.stats['min'], self.stats['max'])
            self.quantile_idx = index[0]
            if self.quantile_idx > self.stats['mean']:
                self.axes.text(index[0], self.stats['max'], f"Quantile: {self.stats['quantile']}",
                                horizontalalignment='right', verticalalignment='top', rotation='vertical')
            else:
                self.axes.text(index[0], self.stats['min'], f"Quantile: {self.stats['quantile']}",
                                horizontalalignment='right', verticalalignment='bottom', rotation='vertical')
        # FuncAnimation takes care of the rest!
        super().__init__(self.fig, self.wrapper_func, frames=None, save_count=self.stats['records']-1)

    def makeRange(self, idx):
        if self.init_ID == 0:
            return range(0, idx+1)
        else:
            return range(idx, self.stats['records'])

    def start(self, event=None):
        #self.event_source.start()
        pass
    def stop(self, event=None):
        self.event_source.stop()
    def onestep(self, mouseEvent=None):
        self.frameID = (self.frameID+1) % (self.stats['records']-1)
        self.wrapper_func(self.frameID)
        self.fig.canvas.draw_idle()
    def oneback(self, mouseEvent=None):
        self.frameID = (self.frameID-1) % (self.stats['records']-1)
        self.wrapper_func(self.frameID)
        self.fig.canvas.draw_idle()
    def reset_left(self, mouseEvent=None):
        self.frameID = 0
        self.wrapper_func(self.frameID)
        self.fig.canvas.draw_idle()
    def reset_right(self, mouseEvent=None):
        self.frameID = self.stats['records']-1
        self.wrapper_func(self.frameID)
        self.fig.canvas.draw_idle()
    def teleport(self, idx):
        self.frameID = idx
        self.wrapper_func(self.frameID)
        self.fig.canvas.draw_idle()
    def quantile_teleport(self, mouseEvent=None):
        if not hasattr(self, 'quantile_idx'):
            return
        self.teleport(self.quantile_idx)
    def updateArtists(self):
        editedArtists = []
        if self.always_all_artists:
            self.axes.clear()
        # Main data update
        colors = self.points.get_fc()
        # Force representation of alpha and nonalpha
        make_alpha_idx = np.where(colors == self.activecolor)[0]
        colors[make_alpha_idx] = self.inactivecolor
        colors[self.makeRange(self.frameID)] = self.activecolor
        self.points.set_fc(colors)
        editedArtists.append(self.points)
        # Move text and change it
        current_data = self.data.loc[self.frameID,self.args.column_name]
        #self.text.set_position((self.frameID/(self.stats['records']-1), (self.stats['max']-current_data)/(self.stats['max']-self.stats['min'])))
        self.text.set_position((self.frameID-10, current_data-100 if current_data > self.stats['mean'] else current_data+100))
        self.text.set_text(f"IDX={self.frameID}\nObjective: {current_data}")
        editedArtists.append(self.text)
        if self.always_all_artists:
            self.fig.canvas.draw()
        return editedArtists

class AnimationController():
    def __init__(self, position, animation, args):
        self.position = position
        self.offset = 0
        self.animation = animation
        self.fig = self.animation.fig
        self.axes = self.animation.axes
        self.setup()
        #self.fig.set_tight_layout(True)
        plt.show()

    def exit_plot(self, mouseEvent):
        self.resolved_frame_ID = self.animation.frameID
        plt.close(fig=self.fig)

    def teleport_exit(self, expression):
        try:
            idx = int(expression)
        except ValueError:
            print(f"Failed to interpret IDX: {expression}")
            return
        #self.animation.teleport(idx)
        self.resolved_frame_ID = idx
        plt.close(fig=self.fig)

    def make_axes(self, bonus_offset = 0):
        ax = self.fig.add_axes([self.position[0]+self.offset+bonus_offset, self.position[1], 0.1, 0.05])
        self.offset += 0.11
        return ax

    def setup(self):
        skip_left_button_ax = self.make_axes()
        self.button_skip_left = widgets.Button(skip_left_button_ax, label=u"$\u29CF$")
        self.button_skip_left.on_clicked(self.animation.reset_left)

        retreat_button_ax = self.make_axes()
        self.button_retreat = widgets.Button(retreat_button_ax, label=u"$\u25C0$")
        self.button_retreat.on_clicked(self.animation.oneback)

        accept_button_ax = self.make_axes()
        self.button_accept = widgets.Button(accept_button_ax, label="OK")
        self.button_accept.on_clicked(self.exit_plot)

        if hasattr(self.animation, 'quantile_idx'):
            quantile_button_ax = self.make_axes()
            self.button_quantile = widgets.Button(quantile_button_ax, label="$\u2020$")
            self.button_quantile.on_clicked(self.animation.quantile_teleport)

        advance_button_ax = self.make_axes()
        self.button_advance = widgets.Button(advance_button_ax, label=u"$\u25B6$")
        self.button_advance.on_clicked(self.animation.onestep)

        skip_right_button_ax = self.make_axes()
        self.button_skip_right = widgets.Button(skip_right_button_ax, label=u"$\u29D0$")
        self.button_skip_right.on_clicked(self.animation.reset_right)

        direct_ax = self.make_axes()
        self.input_idx = widgets.TextBox(direct_ax,"",textalignment='center')
        self.input_idx.on_submit(self.teleport_exit)

def select_ranges(csvs, stats, args):
    selection_idx = []
    for idx, (csv, stat, name) in enumerate(zip(csvs, stats, args.inputs)):
        if args.bottom_up:
            idx_range = [0, None]
        else:
            idx_range = [None, stat['records']]
        fig, axes = plt.subplots()
        animation = Player(fig, axes, csv, stat, name, args)
        controller = AnimationController((0.2,0.92), animation, args)
        if controller.resolved_frame_ID < 0:
            idx_range = [None, None]
        else:
            idx_range[idx_range.index(None)] = controller.resolved_frame_ID
            if args.bottom_up:
                idx_range[1] += 1
        selection_idx.append(idx_range)
    return selection_idx

def output(csvs, idx_ranges, args):
    common_columns = set(csvs[0].columns)
    for other_csv in csvs[1:]:
        common_columns = set(other_csv.columns).intersection(common_columns)
    common_columns = sorted(common_columns)
    print(f"Using common columns: {common_columns}")
    output_frame = pd.DataFrame([], columns=common_columns)
    for csv, idxes, name in zip(csvs, idx_ranges, args.inputs):
        print(f"Range {idxes} for CSV: {name}")
        if all([_ is None for _ in idxes]):
            print("\t"+"--Omitted--")
            continue
        subset = csv.loc[range(*idxes), common_columns]
        if len(output_frame) > 0:
            output_frame = pd.concat((output_frame, subset))
        else:
            output_frame = subset
    print(f"Saving {len(output_frame)} records to {args.output}")
    output_frame.to_csv(args.output)

def main(args=None):
    args = parse(args)
    csvs, stats = load_all(args)
    idx_ranges = select_ranges(csvs, stats, args)
    output(csvs, idx_ranges, args)

if __name__ == '__main__':
    main()

