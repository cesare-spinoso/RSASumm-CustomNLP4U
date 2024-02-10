import hydra
from omegaconf import DictConfig
from src.utils.decorators import main_decorator


@hydra.main(version_base=None)
@main_decorator
def main(run_name: str, cfg: DictConfig):
    pass


if __name__ == "__main__":
    main()
