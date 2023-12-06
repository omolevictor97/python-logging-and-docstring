import pandas as pd
import numpy as np

from utils.all_utils import prepare_data, plot_graph
from utils.models import Perceptron

def main(data, eta:float, epochs:int, plot_dir:str, filename, model_dir:str):
    df_OR = pd.DataFrame(data)
    X, y = prepare_data(df_OR, target='y')
    plot_graph(df_OR, plot_dir, filename)

    model = Perceptron(eta, epochs)

    model.fit(X=X, y=y)
    model.save(filename+str('.pkl'), model_dir)
    total_loss = model.total_loss()
    print(f'Total Loss is {total_loss}')
if __name__ == '__main__':
    AND_GATE = {
        "X1" : [0,1,0,1],
        "X2" : [0,0,1,1],
        "y" :  [0,0,0,1]
    }
    ETA = 0.01
    EPOCHS = 20
    plot_dir = 'Model_graph'
    Filename = 'and_model'
    model_name = 'MODELS'
    main(data=AND_GATE, eta=ETA, epochs=EPOCHS, plot_dir=plot_dir, filename=Filename, model_dir=model_name)




#### Plotting graph to see if data is linearly separable



