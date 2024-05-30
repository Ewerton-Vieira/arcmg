import torch 
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from functools import partial
import itertools
from cycler import cycler
import matplotlib.pyplot as plt

''' filled_hist and stack_hist are taken from the following link: https://matplotlib.org/stable/gallery/lines_bars_and_markers/filled_step.html '''

''' make_edge is from https://github.com/rweng18/midsummer_network/blob/master/midsummer_graph.ipynb '''

def make_arrow(x, y, width):
    return go.layout.Annotation(dict(
                x= x[1] + np.random.rand()*0.02,
                y= y[1],
                xref="x", yref="y",
                text="",
                showarrow=True,
                axref = "x", ayref='y',
                ax= x[0],
                ay= y[0],
                arrowhead = 5,
                arrowwidth=width)
            )
    
def make_edge(x, y, text, width):
    
    '''Creates a scatter trace for the edge between x's and y's with given width

    Parameters
    ----------
    x    : a tuple of the endpoints' x-coordinates in the form, tuple([x0, x1, None])
    
    y    : a tuple of the endpoints' y-coordinates in the form, tuple([y0, y1, None])
    
    width: the width of the line

    Returns
    -------
    An edge trace that goes between x0 and x1 with specified width.
    '''
    return  go.Scatter(x         = x,
                       y         = y,
                       mode="lines+markers",
                       marker      = dict(size = width,
                                   color = 'cornflowerblue', symbol = "arrow-bar-up", angleref="previous"),
                       hoverinfo = 'text',
                       text      = ([text]))

def filled_hist(ax, edges, values, bottoms=None, orientation='v',
                **kwargs):
    """
    Draw a histogram as a stepped patch.

    Parameters
    ----------
    ax : Axes
        The axes to plot to

    edges : array
        A length n+1 array giving the left edges of each bin and the
        right edge of the last bin.

    values : array
        A length n array of bin counts or values

    bottoms : float or array, optional
        A length n array of the bottom of the bars.  If None, zero is used.

    orientation : {'v', 'h'}
       Orientation of the histogram.  'v' (default) has
       the bars increasing in the positive y-direction.

    **kwargs
        Extra keyword arguments are passed through to `.fill_between`.

    Returns
    -------
    ret : PolyCollection
        Artist added to the Axes
    """
    print(orientation)
    if orientation not in 'hv':
        raise ValueError(f"orientation must be in {{'h', 'v'}} "
                         f"not {orientation}")

    kwargs.setdefault('step', 'post')
    kwargs.setdefault('alpha', 0.7)
    edges = np.asarray(edges)
    values = np.asarray(values)
    if len(edges) - 1 != len(values):
        raise ValueError(f'Must provide one more bin edge than value not: '
                         f'{len(edges)=} {len(values)=}')

    if bottoms is None:
        bottoms = 0
    bottoms = np.broadcast_to(bottoms, values.shape)

    values = np.append(values, values[-1])
    bottoms = np.append(bottoms, bottoms[-1])
    if orientation == 'h':
        return ax.fill_betweenx(edges, values, bottoms,
                                **kwargs)
    elif orientation == 'v':
        return ax.fill_between(edges, values, bottoms,
                               **kwargs)
    else:
        raise AssertionError("you should never be here")


def stack_hist(ax, stacked_data, sty_cycle, bottoms=None,
               hist_func=None, labels=None,
               plot_func=None, plot_kwargs=None):
    """
    Parameters
    ----------
    ax : axes.Axes
        The axes to add artists too

    stacked_data : array or Mapping
        A (M, N) shaped array.  The first dimension will be iterated over to
        compute histograms row-wise

    sty_cycle : Cycler or operable of dict
        Style to apply to each set

    bottoms : array, default: 0
        The initial positions of the bottoms.

    hist_func : callable, optional
        Must have signature `bin_vals, bin_edges = f(data)`.
        `bin_edges` expected to be one longer than `bin_vals`

    labels : list of str, optional
        The label for each set.

        If not given and stacked data is an array defaults to 'default set {n}'

        If *stacked_data* is a mapping, and *labels* is None, default to the
        keys.

        If *stacked_data* is a mapping and *labels* is given then only the
        columns listed will be plotted.

    plot_func : callable, optional
        Function to call to draw the histogram must have signature:

          ret = plot_func(ax, edges, top, bottoms=bottoms,
                          label=label, **kwargs)

    plot_kwargs : dict, optional
        Any extra keyword arguments to pass through to the plotting function.
        This will be the same for all calls to the plotting function and will
        override the values in *sty_cycle*.

    Returns
    -------
    arts : dict
        Dictionary of artists keyed on their labels
    """
    # deal with default binning function
    if hist_func is None:
        hist_func = np.histogram

    # deal with default plotting function
    if plot_func is None:
        plot_func = filled_hist

    # deal with default
    if plot_kwargs is None:
        plot_kwargs = {}
    print(plot_kwargs)
    try:
        l_keys = stacked_data.keys()
        label_data = True
        if labels is None:
            labels = l_keys

    except AttributeError:
        label_data = False
        if labels is None:
            labels = itertools.repeat(None)

    if label_data:
        loop_iter = enumerate((stacked_data[lab], lab, s)
                              for lab, s in zip(labels, sty_cycle))
    else:
        loop_iter = enumerate(zip(stacked_data, labels, sty_cycle))

    arts = {}
    for j, (data, label, sty) in loop_iter:
        if label is None:
            label = f'dflt set {j}'
        label = sty.pop('label', label)
        vals, edges = hist_func(data)
        if bottoms is None:
            bottoms = np.zeros_like(vals)
        top = bottoms + vals
        print(sty)
        sty.update(plot_kwargs)
        print(sty)
        ret = plot_func(ax, edges, top, bottoms=bottoms,
                        label=label, **sty)
        bottoms = top
        arts[label] = ret
    ax.legend(fontsize=15)
    return arts

def plot_loss_histogram(loss_success_list, loss_failure_list, train_or_test_str, path_file):
    loss_array = np.array([np.array(loss_success_list).transpose(), np.array(loss_failure_list).transpose()], dtype = object)

    # get the min and max of x and y
    if len(loss_success_list) > 0 and len(loss_failure_list) > 0:
        x_min = min(loss_success_list)
        x_max = max(loss_success_list)
        y_min = min(loss_failure_list)
        y_max = max(loss_failure_list)
        hist_min = x_min if x_min < y_min else y_min
        hist_max = x_max if x_max > y_max else y_max
    
    if len(loss_success_list) == 0:
        hist_min = min(loss_failure_list)
        hist_max = max(loss_failure_list)
    
    if len(loss_failure_list) == 0:
        hist_min = min(loss_success_list)
        hist_max = max(loss_success_list)

    # set up histogram function to fixed bins
    edges = np.linspace(hist_min, hist_max, 30, endpoint=True)
    hist_func = partial(np.histogram, bins=edges)

    # set up style cycles
    color_cycle = cycler(facecolor=plt.rcParams['axes.prop_cycle'][:2])
    label_cycle = cycler(label=['Successful trials', 'Failed trials'])
    hatch_cycle = cycler(hatch=['/', '.'])

    dict_data = dict(zip((c['label'] for c in label_cycle), loss_array))

    fig = plt.figure(figsize=(15,5))
    ax = fig.add_subplot(111)

    stack_hist(ax, dict_data, color_cycle + label_cycle + hatch_cycle, hist_func=hist_func)

    ax.set_ylabel('Stacked frequency')
    ax.set_xlabel('Loss')

    if train_or_test_str == 'train':
        ax.set_title('Train Loss Distribution for Successful and Unsuccessful Trials')
    if train_or_test_str == 'test':
        ax.set_title('Test Loss Distribution for Successful and Unsuccessful Trials')

    plt.savefig(path_file + f'{train_or_test_str}_loss_dist.png')
    plt.show()
    plt.close()


def plot_loss(config, train_loss_list, test_loss_list):
    fig = plt.figure(figsize=(15,5))
    ax = fig.add_subplot(111)

    ax.set_title('Test and Train Loss')
    ax.set_xlabel('Epoch Number')
    ax.set_ylabel('Loss')

    final_epoch_number = len(train_loss_list)+1

    timelist=list(range(1, final_epoch_number))

    plt.subplots_adjust(left=0.2, bottom=0.2, top=0.9, right = 0.9)

    step = max(final_epoch_number//10, 1)

    ax.set(xticks=np.arange(0, final_epoch_number, step=step))
    ax.plot(timelist, test_loss_list, linewidth = 1.2, c = 'blueviolet', linestyle = 'dashed', label = 'Test Loss')
    ax.plot(timelist, train_loss_list, linewidth = 1, c = 'k', label = 'Train Loss')
    ax.legend()

    plt.savefig(f'{config.output_dir}loss.png')
    # plt.show()
    plt.close()

# Needs to be re-written for higher dimensions 
def make_point_mesh(domain, num_points_in_mesh):
    x_lower = domain[0][0]
    x_upper = domain[0][1]
    x = torch.linspace(x_lower, x_upper, num_points_in_mesh)
    mesh = torch.meshgrid(x)
    return mesh

def compute_num_classes_found(model, domain, input_dimension, num_points_in_mesh):
    mesh = make_point_mesh(domain, num_points_in_mesh)
    classes_found_list = []
    for i in range(0, num_points_in_mesh):
        point = torch.zeros(input_dimension, requires_grad = False)
        for dim in range(input_dimension):
            point[dim] = mesh[dim][i]
        probability, classification = model.classification_of_point(point)
        classes_found_list.append(classification)
        print('classification: ', classification)
    classes_found_list = list(set(classes_found_list))
    return len(classes_found_list)

def sample_state(self, num_pts=1, region = False):
    if region is False:
        region = self.get_true_bounds()
    sample_ = np.random.uniform(region[:,0], region[:,1], size=(num_pts, self.dimension()))
    return self.transform(sample_)[0]
    

def plot_classes_2D(model, config, name = ""):
    num_classes = model.output_layer[0].out_features
    X = np.random.uniform(config.domain[0], config.domain[1], size=(config.num_data_points, config.input_dimension))
    X=torch.Tensor(X)
    Z = model.vector_of_probabilities(X)
    Z = torch.argmax(Z, dim=1)
    
    X = X.detach().numpy()
    for i in range(num_classes):
        X_temp = []
        for k, x in enumerate(X):
            if int(Z[k]) == i:
                X_temp.append(x)

        X_temp = np.array(X_temp)
        if len(X_temp) == 0:
            continue
        plt.scatter(X_temp[:,0], X_temp[:,1], marker=".", label="class:"+str(i))


    # fig, ax = plt.subplots()
    # scatter = ax.scatter(X[:,0].detach().numpy(), X[:,1].detach().numpy(), c=Z.detach().numpy(), marker=".")
    # # legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
    # # ax.add_artist(legend1)

    # # ax.xlabel('x_0')
    # # ax.ylabel('x_1')
    plt.legend()
    plt.title(f'Classes')

    plt.savefig(f'{config.output_dir}plot{name}.png')
    # plt.show()
    plt.close()

def plot_classes(model, config, name = ""):
    num_classes = model.output_layer[0].out_features
    for label in range(num_classes):
        mesh = make_point_mesh(config.domain, 100)
        Z = torch.zeros(100, requires_grad = False)

        for i in range(0, 100):
            point = torch.zeros(config.input_dimension, requires_grad = False)
            point[0] = mesh[0][i]

            # Tried calling model.classification_of_point(point) to get the probabilities, but then only one class gets plotted and I don't know why
            point = point.unsqueeze(0)
            output = model.vector_of_probabilities(point)
            
            output = output[0, label]
            output = output.cpu()

            Z[i] = output

        plot = plt.plot(mesh[0].detach().numpy(), Z.detach().numpy(), label = f'Class {label+config.num_attractors}')
        plt.legend()
        plt.xlabel('Activation space')
        plt.ylabel('Network Output')
        plt.title(f'Network Probabilities')
        if label == num_classes - 1:
            plt.savefig(f'{config.output_dir}plot{name}.png')
            # plt.show()
            plt.close()

def make_parallel_coordinates_plot():
    pass