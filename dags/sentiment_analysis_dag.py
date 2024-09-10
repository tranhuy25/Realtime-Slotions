# dags/sentiment_analysis_dag.py

from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
import subprocess

def run_data_preprocessing():
    subprocess.run(["python", "/app/src/data_preprocessing.py"], check=True)

def run_model_training():
    subprocess.run(["python", "/app/src/model_training.py"], check=True)

def run_model_inference():
    subprocess.run(["python", "/app/src/model_inference.py"], check=True)

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'sentiment_analysis_dag',
    default_args=default_args,
    description='DAG for Sentiment Analysis Project',
    schedule_interval=timedelta(days=1),
)

task1 = PythonOperator(
    task_id='data_preprocessing',
    python_callable=run_data_preprocessing,
    dag=dag,
)

task2 = PythonOperator(
    task_id='model_training',
    python_callable=run_model_training,
    dag=dag,
)

task3 = PythonOperator(
    task_id='model_inference',
    python_callable=run_model_inference,
    dag=dag,
)

task1 >> task2 >> task3
