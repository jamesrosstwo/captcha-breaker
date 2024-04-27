import hydra
import sys
import os

# Add the root directory to the Python module search path (otherwise can't find src)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# set manual cache directory for hugging face models
project_dir = "."

cache_dir = f"{project_dir}/.cache"
os.makedirs(cache_dir,exist_ok=True)
os.environ['XDG_CACHE_HOME'] = cache_dir # works
os.environ['MPLCONFIGDIR'] = cache_dir # define before matplotlib is imported
os.environ['HF_HOME'] = cache_dir
os.environ['HUGGINGFACE_HUB_CACHE'] = cache_dir
os.environ['WANDB_DATA_DIR'] = f"{project_dir}/wandb"
xformers_package = os.environ.get('XFORMERS_PACKAGE', 'xformers==0.0.20')


from src.definitions import CONFIG_PATH, BASE_CONFIG_NAME
from src.experiment.experiment import CaptchaExperiment


@hydra.main(version_base=None, config_path=str(CONFIG_PATH.resolve()), config_name=BASE_CONFIG_NAME)
def launch_experiment(cfg):
    experiment = CaptchaExperiment(**cfg)
    return experiment.run()


if __name__ == "__main__":
    launch_experiment()


# run commands
# python src/__main__.py architecture=albert_tinyvit_linear use_wandb=False
# python src/__main__.py architecture=albert_tinyvit_large_linear use_wandb=False
# python src/__main__.py architecture=albert_tinyvit_crossattention use_wandb=False
# python src/__main__.py architecture=albert_tinyvit_canonical architecture.fusion_head.shared_canon=False use_wandb=False

