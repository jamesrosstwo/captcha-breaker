import hydra

from src.definitions import CONFIG_PATH, BASE_CONFIG_NAME
from src.experiment.captcha import CaptchaExperiment


@hydra.main(version_base=None, config_path=str(CONFIG_PATH.resolve()), config_name=BASE_CONFIG_NAME)
def launch_experiment(cfg):
    experiment = CaptchaExperiment(**cfg)
    experiment.run()


if __name__ == "__main__":
    launch_experiment()