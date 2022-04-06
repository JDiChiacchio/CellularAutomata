from Automata import Automata
import numpy as np
import bokeh as bh
from bokeh.plotting import figure, curdoc
from bokeh.transform import linear_cmap, LinearColorMapper
from bokeh.models import Range1d
from scipy.sparse import coo_matrix
import sys
import os


# TODO
# Understand/fix rule change system for generalized mode
# Doubleclick now triggers js callback and sets pointdraw tool as active
#   *Need to overlay text entry box in html and move to mouse location to update state value


class automata_visualizer():
    def __init__(self, alt_func=None):
        if alt_func is None:
            # Default rule is Conways Game of Life
            self.CABackend = Automata({'Ss': [2, 3], 'Bb': [3]})
        else:
            self.CABackend = Automata({'state_func': alt_func},
                                      rule_format='generalized')
        self.current_rule = self.CABackend.rule

        self.gridx = 50
        self.gridy = 50
        self.scale = 12

        self.play_callback_id = None
        self.max_plot_state = self.CABackend.rule['Cc'] - 1

        self.build_layout()

    def tooltip_update(self, atrr, old, new):
        if new != '':
            self.update_max_plot_state(int(new))
            self.point_tool.empty_value = int(new)

    def change_wall_mode(self, event):
        self.CABackend.set_edge_mode(event.item)
        self.edge_mode_selector.label = 'Edge Mode: ' + event.item

    def update_max_plot_state(self, new_max, force=False):
        # This function dynamically updates the maximum state value in the color mapper (useful for custom transition functions with unbounded state)
        # Checks against current max staet value unless force parameter is true
        if new_max > self.max_plot_state or force:
            self.max_plot_state = new_max
            self.color_map['transform'].update(high=self.max_plot_state)

    def make_menu(self, adict):
        return [(k + ': ' + self.format_rule_value(v), k) for k, v in adict.items()]

    ## Functions for rule changes ##

    def edit_rule(self, atrr, old, new):
        if(self.selected_rule_key == 'Bb' or self.selected_rule_key == 'Ss'):
            new = [int(val) for val in new.split(',') if val != '']
        if (self.selected_rule_key == 'Cc'):
            self.update_max_plot_state(int(new) - 1, force=True)
        if (self.selected_rule_key == 'Mm'):
            new = (new.lower() in ['true', '1', 'y'])

        self.current_rule[self.selected_rule_key] = new
        self.CABackend.set_rule(self.current_rule, format='standard')
        self.rule_key_selector.menu = self.make_menu(self.current_rule)
        self.rule_value_setter.visible = False

    def set_rule_key(self, event):
        self.selected_rule_key = event.item
        # set the default value of the setter
        current_val = self.current_rule[self.selected_rule_key]
        self.rule_value_setter.value = self.format_rule_value(current_val)
        self.rule_value_setter.visible = True

    def format_rule_value(self, val):
        if isinstance(val, np.ndarray):
            # Resolbe array values into strings
            val = np.squeeze(val)
            val = str(val.tolist())
            val = val.replace('\'', '')
            output_string = val.strip('[]')
        else:
            output_string = str(val)
        return output_string

    ## Functions for grid changes ##

    def update_grid_view(self):
        self.plot.frame_width = self.scale*self.gridx
        self.plot.frame_height = self.scale*self.gridy

        self.plot.x_range.end = self.gridx - 0.5
        self.plot.y_range.end = self.gridy - 0.5

        self.plot.x_range.reset_end = self.gridx - 0.5
        self.plot.y_range.reset_end = self.gridy - 0.5

        self.plot.x_range.bounds = (-0.5, self.gridx - 0.5)
        self.plot.y_range.bounds = (-0.5, self.gridy - 0.5)

        # Fix vertical grid lines at even intervals
        self.plot.xaxis.ticker = [val - 0.5 for val in range(self.gridx)]
        # Fix horizontal grid lines at even intervals
        self.plot.yaxis.ticker = [val - 0.5 for val in range(self.gridy)]

    def select_grid_prop(self, event):
        self.selected_grid_prop = event.item
        cur_val = 0
        if self.selected_grid_prop == 'Width':
            cur_val = self.gridx
        elif self.selected_grid_prop == 'Height':
            cur_val = self.gridy
        elif self.selected_grid_prop == 'Scale':
            cur_val = self.scale
        self.grid_prop_setter.value = str(cur_val)
        self.grid_prop_setter.visible = True

    def set_grid_prop(self, atrr, old, new):
        if self.selected_grid_prop == 'Width':
            self.gridx = int(new)
        elif self.selected_grid_prop == 'Height':
            self.gridy = int(new)
        elif self.selected_grid_prop == 'Scale':
            self.scale = int(new)
        self.update_grid_view()
        self.grid_prop_selector.menu = self.make_menu(
            {'Width': self.gridx, 'Height': self.gridy, 'Scale': self.scale})
        self.grid_prop_setter.visible = False

    ## Self similarity ##
    def ssim(self):
        # Function for displaying a self similarity matrix for a given configuration's evolution
        size = 100
        src = self.source.data
        arr = self.CABackend.build_self_sim(size)

        self.gridx, self.gridy = size, size
        self.update_grid_view()

        coo_src = coo_matrix(arr)
        x = coo_src.col
        y = coo_src.row
        vals = coo_src.data
        src.update({'x_values': x, 'y_values': y, 'states': vals})
        self.update_max_plot_state(np.amax(arr), True)

    ## Primary Functions ##

    def step_graph(self):
        # Update model grid from bokeh source
        src = self.source.data
        dense = coo_matrix((src['states'], (np.round(src['y_values']), np.round(
            src['x_values']))), dtype=int, shape=(self.gridx, self.gridy)).toarray()
        self.CABackend.set_initial_state(dense)
        new_grid = self.CABackend.evolve_grid()
        if self.CABackend.edge_mode == 'GROW':
            if self.CABackend.grid is not None:
                self.gridx, self.gridy = self.CABackend.grid.shape
                self.update_grid_view()

        # Update bokeh source from model evolution
        coo_src = coo_matrix(new_grid)
        x = coo_src.col
        y = coo_src.row
        vals = coo_src.data
        src.update({'x_values': x, 'y_values': y, 'states': vals})
        try:  # For edge case in which grid is empty
            max_state = np.amax(vals)
        except ValueError:
            max_state = self.max_plot_state
        self.update_max_plot_state(max_state)

    def animate(self):
        if self.play_button.label == '► Play':
            self.play_button.label = '❚❚ Pause'
            self.play_callback_id = curdoc().add_periodic_callback(self.step_graph, 200)
        else:
            self.play_button.label = '► Play'
            curdoc().remove_periodic_callback(self.play_callback_id)

    def build_widgets(self):
        self.step_button = bh.models.Button(label='> Step', width=80)
        self.step_button.on_click(self.step_graph)

        self.play_button = bh.models.Button(label='► Play', width=80)
        self.play_button.on_click(self.animate)

        self.ssim_button = bh.models.Button(label='Self Sim', width=80)
        self.ssim_button.on_click(self.ssim)

        self.state_selector = bh.models.TextInput(
            value='1', placeholder='Draw Tool State:', max_width=80)
        self.state_selector.on_change('value', self.tooltip_update)

        self.rule_key_selector = bh.models.Dropdown(
            label="Edit Rule", button_type="warning", menu=self.make_menu(self.current_rule), max_width=100)
        self.rule_key_selector.on_click(self.set_rule_key)

        self.rule_value_setter = bh.models.TextInput(
            visible=False, max_width=100)
        self.rule_value_setter.on_change('value', self.edit_rule)

        self.grid_prop_selector = bh.models.Dropdown(label="Grid", button_type="warning", menu=self.make_menu(
            {'Width': self.gridx, 'Height': self.gridy, 'Scale': self.scale}), max_width=100)
        self.grid_prop_selector.on_click(self.select_grid_prop)

        self.grid_prop_setter = bh.models.TextInput(
            visible=False, max_width=100)
        self.grid_prop_setter.on_change('value', self.set_grid_prop)

        self.edge_mode_selector = bh.models.Dropdown(
            label="Edges: WALL", button_type="warning", menu=['WALL', 'WRAP', 'GROW'], max_width=100)
        self.edge_mode_selector.on_click(self.change_wall_mode)

    def build_plot(self):
        self.plot = figure(title='Life Visualizer', background_fill_color='#CDCDCD', toolbar_location="below",
                           x_range=Range1d(-0.5, self.gridx - 0.5, bounds='auto'), y_range=Range1d(-0.5, self.gridy - 0.5, bounds='auto'))
        self.update_grid_view()
        # Format cell grid
        self.plot.xaxis.visible = False  # Remove x axis line
        self.plot.yaxis.visible = False  # Remove y axis line
        self.plot.xaxis.major_tick_line_color = None  # Turn off x-axis major ticks
        self.plot.yaxis.major_tick_line_color = None  # Turn off y-axis major ticks
        self.plot.xaxis.major_label_text_font_size = '0pt'  # Turn off x-axis tick labels
        self.plot.yaxis.major_label_text_font_size = '0pt'  # Turn off y-axis tick labels

        # Instantiate data source
        self.source = bh.models.ColumnDataSource(
            data={'x_values': [], 'y_values': [], 'states': []})
        # JS code segment to round point coordinates
        self.source.js_on_change('data', bh.models.CustomJS(args=dict(source=self.source), code="""
        var data = source.data;
        var x = data['x_values']
        var y = data['y_values']
        data['x_values'] = x.map(val => Math.round(val))
        data['y_values'] = y.map(val => Math.round(val))
        source.change.emit();
        """))

        # Initialize cells
        self.color_map = linear_cmap(
            'states', 'Turbo256', 0, self.max_plot_state)
        self.points = self.plot.circle(
            x='x_values', y='y_values', color=self.color_map, radius=0.5, source=self.source)

        # Install tools
        self.point_tool = bh.models.PointDrawTool(
            renderers=[self.points], empty_value=1)
        hover_tool = bh.models.HoverTool(
            tooltips=[('Location', '(@{x_values}, @{y_values})'), ('State', '@{states}')])
        self.plot.add_tools(self.point_tool, hover_tool)

        # Configure state update for draw tool
        self.plot.js_on_event(bh.events.DoubleTap, bh.models.CustomJS(
            args=dict(t=self.plot.select_one(bh.models.PointDrawTool)), code="""
            t.active = true;
            console.log(event);"""))

        # Add additional state/color legend
        color_bar = bh.models.ColorBar(
            color_mapper=self.color_map['transform'], width=8,  location=(0, 0))
        self.plot.add_layout(color_bar, 'right')

    def build_layout(self):
        self.build_widgets()
        self.build_plot()
        app = bh.layouts.layout(
            [self.plot,
             [[self.play_button, self.step_button, self.ssim_button, self.state_selector],
              [self.rule_key_selector, self.rule_value_setter, self.grid_prop_selector, self.grid_prop_setter, self.edge_mode_selector]]])
        curdoc().add_root(app)


made = automata_visualizer()
