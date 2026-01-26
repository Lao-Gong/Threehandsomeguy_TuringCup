import pandas as pd
from utils import *


########Your Model
class MyModel:
    def __init__(self):
        pass

    def predict(self, input_frame):
        pre= input_frame['AskVolume1'] + input_frame['BidVolume1']
        return pre


def do_test():
    target_name_list = get_target_name("./data")
    
    ########Load Your Model
    model = MyModel()
    for target_name in target_name_list:
        ##########Load Test Data
        input_frame = pd.read_csv("./data/"+target_name)

        ##########Your Predict
        pre_frame = model.predict(input_frame)

        ##########Save Your Data
        out_frame = pd.concat([input_frame['Time'], pre_frame], axis=1, ignore_index=True)
        columns = ['Time', 'Predict']
        out_frame.columns = columns
        out_frame.to_csv("./output/"+target_name, index=False)

        print ("Predict", target_name)

    
if __name__ == '__main__':
    do_test()
