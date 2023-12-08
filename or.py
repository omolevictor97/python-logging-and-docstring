import pandas as pd
import numpy as np
import logging
import os

gate = 'OR GATE'
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    filename= os.path.join('logs', 'running_logs.log'),
    level = logging.INFO,
    format= '[%(asctime)s: %(levelname)s: %(module)s]: %(message)s ',
    filemode = 'a'
)

from utils.all_utils import prepare_data, plot_graph
from utils.models import Perceptron

def main(data, eta:float, epochs:int, plot_dir:str, filename, model_dir:str):
    df_OR = pd.DataFrame(data)
    logging.info(f'Dataframe \n {df_OR} \t has been created')
    X, y = prepare_data(df_OR, target='y')
    plot_graph(df_OR, plot_dir, filename)

    model = Perceptron(eta, epochs)

    model.fit(X=X, y=y)
    model.save(filename+str('.pkl'), model_dir)
    total_loss = model.total_loss()
    print(f'Total Loss is {total_loss}')
if __name__ == '__main__':
    OR_GATE = {
        "X1" : [0,1,0,1],
        "X2" : [0,0,1,1],
        "y" :  [0,1,1,1]
    }
    ETA = 0.01
    EPOCHS = 20
    plot_dir = 'Model_graph'
    Filename = 'or_model'
    model_name = 'MODELS'
    try:
        logging.info(f'>>>>>>>> Training For {gate} Has Statrted Already >>>>>>>>')
        main(data=OR_GATE, eta=ETA, epochs=EPOCHS, plot_dir=plot_dir, filename=Filename, model_dir=model_name)
        logging.info(f'<<<<<<<< Training For {gate} Has Been Completed <<<<<<<< \n\n')
    except Exception as e:
        logging.exception(e)
        raise e




#### Plotting graph to see if data is linearly separable



