import logging
from pathlib import Path
from typing import Union, Literal, Optional

import neptune

from utils.misc import PathUtils, env_get


class NeptuneFetcher:
    """
    A utility class to fetch Neptune project and run information.
    It caches projects and runs to avoid repeated API calls.
    """
    PROJECTS_CACHE = {}
    RUNS_CACHE = {}

    def __init__(self, project_name: Optional[str] = None):
        self._project = NeptuneUtils.get_project(project_name)
        self.project_shortcode = NeptuneUtils.get_project_id(self._project)

    def download_checkpoint(self, run_idx: str) -> None:
        pass


class NeptuneUtils:
    PROJECTS_CACHE = {}
    RUNS_CACHE = {}

    @classmethod
    def get_project(cls, project_name: Optional[str] = None) -> neptune.Project:
        """
        Get the Neptune project.

        Parameters
        ----------
        project_name : Optional[str]
            The name of the Neptune project. If not provided, it will use the one from the environment variable NEPTUNE_PROJECT.

        Returns
        -------
        neptune.Project
            The Neptune project instance.
        """
        project_name = project_name if project_name is not None else env_get('NEPTUNE_PROJECT')
        if project_name in cls.PROJECTS_CACHE:
            return cls.PROJECTS_CACHE[project_name]
        logging.getLogger('neptune').setLevel(logging.ERROR)
        project = neptune.init_project(
            project=project_name,
            api_token=env_get('NEPTUNE_API_TOKEN'),
            mode="read-only",
        )
        project._mode = neptune.types.mode.Mode.OFFLINE
        cls.PROJECTS_CACHE[project_name] = project
        return project

    @classmethod
    def get_project_id(cls, project: Optional[neptune.Project] = None, project_name: Optional[str] = None) -> str:
        """
        Get the Neptune project ID.

        Parameters
        ----------
        project : Optional[neptune.Project]
            The Neptune project instance. If not provided, it will be fetched using the project_name.
        project_name : Optional[str]
            The name of the Neptune project. If not provided, it will use the one from the environment variable NEPTUNE_PROJECT.

        Returns
        -------
        str
            The Neptune project ID.
        """
        if project is None:
            project = cls.get_project(project_name=project_name)
        return project['sys/id'].fetch()

    @classmethod
    def get_project_name(cls, project: Optional[neptune.Project] = None, project_name: Optional[str] = None) -> str:
        """
        Get the Neptune project name.

        Parameters
        ----------
        project : Optional[neptune.Project]
            The Neptune project instance. If not provided, it will be fetched using the project_name.
        project_name : Optional[str]
            The name of the Neptune project. If not provided, it will use the one from the environment variable NEPTUNE_PROJECT.

        Returns
        -------
        str
            The Neptune project ID.
        """
        if project is None:
            project = cls.get_project(project_name=project_name)
        return project['sys/name'].fetch()

    @classmethod
    def get_run(cls, run_id: str, project_name: Optional[str] = None) -> neptune.Run:
        """
        Get a Neptune run by name.

        Parameters
        ----------
        run_id : str
            The name of the Neptune run.
        project_name : Optional[str]
            The name of the Neptune project. If not provided, it will use the one from the environment variable NEPTUNE_PROJECT.

        Returns
        -------
        neptune.Run
            The Neptune project run.
        """
        if run_id in cls.RUNS_CACHE:
            return cls.RUNS_CACHE[run_id]
        project_name = project_name if project_name is not None else env_get('NEPTUNE_PROJECT')
        logging.getLogger('neptune').setLevel(logging.ERROR)
        run = neptune.init_run(
            with_id=run_id,
            project=project_name,
            api_token=env_get('NEPTUNE_API_TOKEN'),
            mode="read-only",
        )
        run._mode = neptune.types.mode.Mode.OFFLINE
        cls.RUNS_CACHE[run_id] = run
        return run

    @classmethod
    def get_run_by_index(cls, run_index: int, project_name: Optional[str] = None) -> neptune.Run:
        """
        Get a Neptune project run by its index.

        Parameters
        ----------
        run_index : int
            The index of the Neptune project run.
        project_name : Optional[str]
            The name of the Neptune project. If not provided, it will use the one from the environment variable NEPTUNE_PROJECT.

        Returns
        -------
        neptune.Run
            The Neptune project run.
        """
        run_prefix = cls.get_project_id(project_name=project_name)
        run_id = f'{run_prefix}-{run_index}'
        return cls.get_run(run_id=run_id, project_name=project_name)

    @classmethod
    def get_run_id(cls, run: neptune.Run) -> str:
        """
        Get the ID of a Neptune run.

        Parameters
        ----------
        run : neptune.Run
            The Neptune run from which to get the name.

        Returns
        -------
        str
            The ID of the Neptune run.
        """
        return run["sys/id"].fetch()

    @classmethod
    def download_checkpoint_from_run(cls, run: neptune.Run, checkpoint: Union[Literal['best'], str], download_path: Union[Path, str] = '__CHECKPOINTS_PATH__/__PROJECT_NAME__', overwrite: bool = False) -> Path:
        """
        Download a checkpoint file from a Neptune run.

        Parameters
        ----------
        run : neptune.Run
            The Neptune run from which to download the checkpoint.
        checkpoint : Union[Literal['best'], str]
            The name of the checkpoint file to download. If 'best', it will download the best checkpoint.
        download_path : Path
            The local path where the checkpoint will be saved.
        overwrite : bool
            If True, overwrite the existing file if it exists. Default is False.

        Returns
        -------
        Path
            The local path where the checkpoint was saved.
        """
        # get checkpoint path
        if checkpoint == 'best':
            neptune_file_path = Path(f'train/model/checkpoints{(run["train/model/best_model_path"].fetch()).split(run["sys/id"].fetch())[-1]}')
        else:
            neptune_file_path = Path(f'train/model/checkpoints/{checkpoint}')
        neptune_attr = str(neptune_file_path.with_suffix(''))
        neptune_file = run[neptune_attr]
        # prepare download path
        # noinspection PyProtectedMember
        download_path = Path(str(download_path).replace('__CHECKPOINTS_PATH__', str(PathUtils.checkpoints_path()).rstrip('/')).replace('__PROJECT_NAME__', run._project_name))
        if download_path.is_file():
            download_path_parent = download_path.parent
            download_path_parent.mkdir(parents=True, exist_ok=True)
        else:
            download_path.mkdir(parents=True, exist_ok=True)
            # create filename: <run_id>.<original_extension>
            download_path = download_path / cls.get_run_id(run).lower()
            if checkpoint == 'best':
                download_path = f'{str(download_path)}-best'
            download_path = Path(download_path).with_suffix(neptune_file_path.suffix)
        # check if file already exists
        if download_path.exists() and not overwrite:
            return download_path
        # download file
        neptune_file.download(str(download_path), progress_bar=False)
        return download_path

    @classmethod
    def download_checkpoint_from_run_index(cls, run_index: int, *arg, project_name: Optional[str] = None, **kwargs) -> Path:
        """
        Download a checkpoint file from a Neptune run identified by its index.

        Parameters
        ----------
        run_index : int
            The index of the Neptune run from which to download the checkpoint.
        project_name : Optional[str]
            The name of the Neptune project. If not provided, it will use the one from the environment variable NEPTUNE_PROJECT.
        *arg : tuple
            Additional positional arguments to pass to the download function.
        **kwargs : dict
            Additional keyword arguments to pass to the download function, including:
            - checkpoint_name: Union[Literal['best'], str] - The name of the checkpoint file to download.
            - download_path: Union[Path, str] - The local path where the checkpoint will be saved.

        Returns
        -------
        Path
            The local path where the checkpoint was saved.
        """
        # get run
        run = cls.get_run_by_index(run_index, project_name=project_name)
        return cls.download_checkpoint_from_run(run=run, *arg, **kwargs)

    @classmethod
    def decode_checkpoint_path(cls, checkpoint_url: str, overwrite: bool = False, convert_to_pth: bool = True) -> Path:
        """
        Decode a checkpoint path from a Neptune URL.

        Parameters
        ----------
        checkpoint_url : str
            The Neptune URL of the checkpoint, e.g. "neptune://<project_name>/<run_id:int or str>/<neptune_path.ckpt>" or "neptune://<run_idx:int>/best".
        overwrite : bool
            If True, overwrite the existing file if it exists.
        convert_to_pth : bool
            If True, convert the checkpoint file to .pth format to be used by `sft.gps_impl.GPS`. Otherwise, the .ckpt file has to be loaded in `sft.gps_impl.GPSLightningModule`. Default is True.

        Returns
        -------
        Path
            The local path where the checkpoint was saved.
        """
        assert checkpoint_url.startswith('neptune://'), "Checkpoint URL must start with 'neptune://'."
        run_info = checkpoint_url[10:].split('/')
        if len(run_info) == 3:
            project_name, run_id, neptune_path = run_info
        elif len(run_info) == 2:
            run_id, neptune_path = run_info
            project_name = None
        else:
            raise ValueError(f"Invalid Neptune URL format: {checkpoint_url}")
        if run_id.isdigit():
            run_id = int(run_id)
            checkpoint_path = cls.download_checkpoint_from_run_index(run_index=run_id, project_name=project_name, checkpoint=neptune_path, overwrite=overwrite)
        else:
            run = cls.get_run(run_id=run_id, project_name=project_name)
            checkpoint_path = cls.download_checkpoint_from_run(run=run, checkpoint=neptune_path, overwrite=overwrite)
        if convert_to_pth and checkpoint_path.suffix == '.ckpt':
            from sft.gps_impl import GPSLightningModule
            checkpoint_path = GPSLightningModule.convert_ckpt(checkpoint_path, checkpoint_path.with_suffix('.pth'), overwrite=overwrite, return_sd=False)
        return checkpoint_path


if __name__ == '__main__':
    run_ = NeptuneUtils.get_run_by_index(85)
    saved_path_ = NeptuneUtils.download_checkpoint_from_run_index(85, checkpoint='best', overwrite=False)
    print(saved_path_, f'{saved_path_.stat().st_size / (1024 * 1024):.2f} MB')
    saved_path_2_ = NeptuneUtils.decode_checkpoint_path('neptune://85/best', overwrite=False)
    print(saved_path_2_, f'{saved_path_2_.stat().st_size / (1024 * 1024):.2f} MB')
