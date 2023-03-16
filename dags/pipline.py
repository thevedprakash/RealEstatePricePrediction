from airflow import DAG
from datetime import datetime, timedelta
from airflow.operators.python import PythonOperator
from utils.load import load_data
from utils.pre_processing import pre_process
from utils.train import train
import os
import sys

#current_path = os.path.dirname(os.path.abspath(__file__))
#sys.path.append(os.path.join(current_path, "utils"))
sys.path.insert(0, 'utils')

default_args = {
    "owner": "RISHAB",
    "retries":5,
    "retry_delay": timedelta(minutes=2),

}

with DAG (
    dag_id = "house_usa",
    default_args=default_args,
    description = 'Predict price of house',
    start_date=datetime(2023, 3, 1, 14),
    schedule='@hourly'



) as dag:
    load_train_data = PythonOperator(
        task_id='load_data_train',
        python_callable=load_data
    )

    pre_processing = PythonOperator(
        task_id='preprocessing',
        python_callable=pre_process
    )

    training = PythonOperator(
        task_id='training',
        python_callable=train
    )



load_train_data >> pre_processing >> training




