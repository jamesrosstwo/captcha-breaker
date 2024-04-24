import hydra
import sys
import os

# Add the root directory to the Python module search path (otherwise can't find src)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# set manual cache directory for stable diffusion
project_dir = "/scratch/dajisafe/course_project/cpsc_540/captcha-breaker"

cache_dir = f"{project_dir}/.cache"
os.makedirs(cache_dir,exist_ok=True)
os.environ['XDG_CACHE_HOME'] = cache_dir # works
os.environ['MPLCONFIGDIR'] = cache_dir # define before matplotlib is imported
os.environ['HF_HOME'] = cache_dir
os.environ['HUGGINGFACE_HUB_CACHE'] = cache_dir
os.environ['WANDB_DATA_DIR'] = f"{project_dir}/wandb"
xformers_package = os.environ.get('XFORMERS_PACKAGE', 'xformers==0.0.20')


from src.definitions import CONFIG_PATH, BASE_CONFIG_NAME
from src.experiment.captcha import CaptchaExperiment


@hydra.main(version_base=None, config_path=str(CONFIG_PATH.resolve()), config_name=BASE_CONFIG_NAME)
def launch_experiment(cfg):
    experiment = CaptchaExperiment(**cfg)
    experiment.run()


if __name__ == "__main__":
    launch_experiment()


# run commands
# python src/__main__.py loader.num_workers=4 loader.batch_size=4 architecture.fusion_head=linear_head
# python src/__main__.py loader.num_workers=4 loader.batch_size=4 +architecture.fusion_head.shared_canon=True
#