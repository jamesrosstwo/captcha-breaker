import hydra
import sys
import os

# Add the root directory to the Python module search path (otherwise can't find src)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.definitions import CONFIG_PATH, BASE_CONFIG_NAME
from src.experiment.captcha import CaptchaExperiment


@hydra.main(version_base=None, config_path=str(CONFIG_PATH.resolve()), config_name=BASE_CONFIG_NAME)
def launch_experiment(cfg):
    experiment = CaptchaExperiment(**cfg)
    experiment.run()


if __name__ == "__main__":
    launch_experiment()