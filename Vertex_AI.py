# We are import the require library
from google.cloud import aiplatform

# We are initialize Vertex AI client
client = aiplatform.gapic.JobServiceClient()

# We are define fine-tuneing jobs parameters
job = {
    "display_name": "fine-tuning-job",
    "model": "projects/<project-id>/locations/<location>/models/<model-id>",
    "training_input": {
        "scale_tier": "BASIC",
        "python_package_spec":{
            "executor_image_uri": "gcr.io/cloud-aiplatform/training/tf-cpu.2-3:latest",
            "package_uris": ["gs://<bucket-name>/<package-name>.tar.gz"],
            "python_module": "<python-module-name>",
            "args": ["--train_data_path", "gs://<bucket-name>/train.tfrecord",
                     "--eval_data_path", "gs://<bucket-name>/eval.tfrecord",
                     "--num_train_steps", "1000",
                     "--output_dir", "gs://<bucket-name>/output"]
        }
    }
}

# We are submit the fine-tuning job
operation = client.create_custom_job(parent="projects/<project-id>/locations/<location>", custom_job=job)
print("Job submitted:",operation.result())
