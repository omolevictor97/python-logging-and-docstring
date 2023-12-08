import matplotlib.pyplot as plt

import os
import logging


def prepare_data(df, target=None):
    
    '''
    
    '''

    if target is not None:
        X = df.drop(target, axis=1).values
        y = df[target].values
        logging.info(f'Data {df} \t has been prepared')
    return X, y


def plot_graph(df, plot_dir=None, filename=None):
    X = df[['X1', 'X2']].values
    y = df['y'].values

    plt.plot(
        X[y == 0, 0],
        X[y == 0, 1],
        marker = '^',
        linestyle = '',
        color = 'orange',
        label = 'Class 0',
        markersize = 10
    )

    plt.plot(
        X[y == 1,0],
        X[y == 1,1],
        marker = 'D',
        linestyle = '',
        color = 'royalblue',
        label = 'Class 1',
        markersize = 12
    )

    if plot_dir is not None and filename is not None:
        os.makedirs(plot_dir, exist_ok=True)
        graph_path = os.path.join(plot_dir, filename)

    #plt.xlim([-1,1])
    #plt.ylim([-1,1])
    plt.grid()
    plt.legend(loc='center')
    plt.xlabel('X1')
    plt.ylabel('X2')

    plt.savefig(graph_path)