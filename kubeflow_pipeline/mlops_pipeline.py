import kfp
from kfp import dsl
from kfp.dsl import component

# --------------------------
# Define pipeline components
# --------------------------

@component(base_image="python:3.9")
def data_processing_op():
    import subprocess
    subprocess.run(["python", "src/data_processing.py"], check=True)

@component(base_image="python:3.9")
def model_training_op():
    import subprocess
    subprocess.run(["python", "src/model_training.py"], check=True)

# ----------------------
# Define the KFP pipeline
# ----------------------

@dsl.pipeline(
    name="MLops Pipeline",
    description="My first Kubeflow pipeline"
)
def mlops_pipeline():
    data_task = data_processing_op()
    training_task = model_training_op().after(data_task)

# ----------------------
# Compile the pipeline YAML
# ----------------------

if __name__ == "__main__":
    kfp.compiler.Compiler().compile(
        pipeline_func=mlops_pipeline,
        package_path="mlops_pipeline.yaml"
    )
