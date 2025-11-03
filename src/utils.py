import mlflow
def log_run_infos(script_name= "train_model.py"):
    mlflow.set_tag("script_name", script_name)
    mlflow.set_tag("run_id", mlflow.active_run().info.run_id)
    mlflow.set_tag('pipeline_step', 'training')