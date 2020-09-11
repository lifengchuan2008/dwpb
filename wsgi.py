
import os

from DWPB_denoising import DWPB
from flask_executor import Executor
import pandas as pd
from threading import Thread

from sqlalchemy import create_engine


from flask import Flask
application = Flask(__name__)
executor = Executor(application)

env = os.environ
# MYSQL_URI   mysql+pymysql://test:test@172.30.238.185:3306/test
mysql_uri = env.get('MYSQL_URI')

sqlEngine = create_engine(mysql_uri, pool_recycle=3600)

print ('=== mysql uri: ' + mysql_uri)



# rest  api
@application.route('/')
def hello():

    executor.submit(threaded_task,'data')
    return b'DWPB '


@application.route('/api')
def api():
    executor.submit(threaded_task, 'data')
    return b'DWPB '

if __name__ == '__main__':

    application.run()


def threaded_task(data):
    try:

        print ('===== run task')

        # load imbalance2  from  mysql 
        data_2 = pd.read_sql("select * from  imbalance2 ", con=sqlEngine, index_col='Time')

        print ('data2 to dataframe')
       # load imbalance14  from  mysql 
        data_14 = pd.read_sql("select * from  imbalance14 ", con=sqlEngine, index_col='Time')

        print ('data14 to dataframe')

        Wavelet_basis = 'db8'
        Max_layers = 3
        data_2_denoised = DWPB(Wavelet_basis, Max_layers).denoising_process(data_2)
        data_14_denoised = DWPB(Wavelet_basis, Max_layers).denoising_process(data_14)


        # dataframe to  mysql 
        data_2_denoised.to_sql('imbalance2_denoised', con=sqlEngine, if_exists='append', index=True)
        # dataframe to  mysql 
        data_14_denoised.to_sql('imbalance14_denoised', con=sqlEngine, if_exists='append', index=True)


    except Exception as e:
        print ('===error===')
        print (e)
        raise e
    return True
