import abc
import io
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Union, List

from utils.misc import env_get, log

Pathlike = Union[str, Path]


class IFilesystem(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_root(self) -> Optional[Path]:
        raise NotImplementedError

    @abc.abstractmethod
    def exists(self, path: Pathlike) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def listdir(self, path: Pathlike):
        raise NotImplementedError

    @abc.abstractmethod
    def mkdir(self, path: Pathlike, parents: bool = True) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def remove(self, path: Pathlike) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def retrieve(self, path: Pathlike, local_path: Optional[Pathlike] = None) -> Union[Pathlike, bytes]:
        raise NotImplementedError

    @abc.abstractmethod
    def rmdir(self, path: Pathlike) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def set_root(self, path: Pathlike, mkdir: bool = True, **mkdir_kwargs) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def store(self, local_file: Union[Pathlike, bytes], path: Pathlike) -> bool:
        raise NotImplementedError

    def download_file(self, path: Pathlike, local_path: Optional[Pathlike] = None) -> Union[Pathlike, bytes]:
        return self.retrieve(path, local_path)

    def upload_file(self, local_file: Pathlike, path: Pathlike) -> bool:
        return self.store(local_file, path)


class LocalFilesystem(IFilesystem):
    def __init__(self, root: Optional[Pathlike] = None):
        self.root = None if root is None else Path(root)

    def get_root(self) -> Optional[Path]:
        return self.root

    def exists(self, path: Pathlike) -> bool:
        if self.root is not None and not str(path).startswith(str(self.root)):
            path = self.root / path
        return path.exists()

    def listdir(self, path: Pathlike) -> List[str]:
        return os.listdir(path)

    def mkdir(self, path: Pathlike, parents: bool = True) -> None:
        if self.root is not None and not str(path).startswith(str(self.root)):
            path = self.root / path
        Path(path).mkdir(parents=parents, exist_ok=True)

    def remove(self, path: Pathlike) -> None:
        if self.root is not None and not str(path).startswith(str(self.root)):
            path = self.root / path
        Path(path).unlink(missing_ok=True)

    def retrieve(self, path: Pathlike, local_path: Optional[Pathlike] = None) -> Union[Pathlike, bytes]:
        if self.root is not None and not str(path).startswith(str(self.root)):
            path = self.root / path
        if local_path is None:
            with open(path, 'rb') as bin_fp:
                out = bin_fp.read()
            return out
        local_path = Path(local_path)
        path = Path(path)
        if not local_path.exists():
            local_path.mkdir(parents=True, exist_ok=True)
        if local_path.is_dir():
            local_path = local_path / path.name
        shutil.copyfile(path, local_path)

    def rmdir(self, path: Pathlike) -> None:
        if self.root is not None and not str(path).startswith(str(self.root)):
            path = self.root / path
        shutil.rmtree(path)

    def set_root(self, path: Pathlike, mkdir: bool = True, **mkdir_kwargs) -> None:
        if self.root is not None and not str(path).startswith(str(self.root)):
            path = self.root / path
        if mkdir:
            self.mkdir(path, **mkdir_kwargs)
        self.root = path

    def store(self, local_file: Union[Pathlike, bytes], path: Pathlike) -> bool:
        if self.root is not None and not str(path).startswith(str(self.root)):
            path = self.root / path
        if isinstance(local_file, (str, Path)):
            shutil.copyfile(local_file, path)
            return True
        with open(path, 'wb') as bin_fp:
            n_bytes_written = bin_fp.write(local_file)
        return n_bytes_written == len(local_file)


class SMBClient:
    def __init__(self, ip: str, username: str, password: str, share_name: str, remote_name: str = 'server',
                 port: int = 445):
        self._ip = ip
        self._port = port
        self._username = username
        self._password = password
        self._share_name = share_name
        self._remote_name = remote_name
        self._localhost = None
        self._connection = None

    @property
    def connection(self):
        if self._connection is None:
            from smb.SMBConnection import SMBConnection
            self._connection = SMBConnection(
                username=self._username,
                password=self._password,
                my_name=self.localhost,
                remote_name=self._remote_name,
                use_ntlm_v2=True
            )
            self._connection.connect(self._ip, port=self._port)
        return self._connection

    def download(self, remote_path: str, local_path: Optional[Path] = None, show_progress: bool = False) -> io.BytesIO:
        """ Download a file from the remote share. """
        if local_path is None:
            # download in memory
            local_fp = local_path = io.BytesIO()
            local_fp.close = lambda: local_fp.seek(0)
        else:
            local_path = Path(local_path)
            if local_path.suffix != Path(remote_path).suffix:
                local_path = local_path / Path(remote_path).name
            local_fp = open(local_path, 'wb')
        self.connection.retrieveFile(service_name=self._share_name, path=remote_path, file_obj=local_fp,
                                     show_progress=show_progress)
        local_fp.close()
        return local_path

    @classmethod
    def from_env(cls) -> 'SMBClient':
        return cls(
            ip=env_get('NAS_IP'),
            username=env_get('NAS_USERNAME'),
            password=env_get('NAS_PASSWORD'),
            share_name=env_get('NAS_SHARE'),
        )

    @property
    def localhost(self) -> str:
        if self._localhost is None:
            self._localhost = str(subprocess.Popen(['hostname'], stdout=subprocess.PIPE).communicate()[0].strip())
        return self._localhost

    def upload(self, local_path: Union[Path, io.BytesIO], remote_path: str, show_progress: bool = False) -> bool:
        if isinstance(local_path, (Path, str)):
            local_fp = open(local_path, mode='rb')
        else:
            if hasattr(local_path, 'seek'):
                local_path.seek(0)
            local_fp = local_path
            show_progress = False  # not supported
        n_bytes = self.connection.storeFile(service_name=self._share_name, path=remote_path, file_obj=local_fp,
                                            show_progress=show_progress)
        local_fp.close()
        return n_bytes > 0


class SMBFilesystem(SMBClient, IFilesystem):
    def __init__(self, root: Optional[Pathlike] = None, show_progress: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.root = None if root is None else Path(Path(root).as_posix())
        self._show_progress = show_progress

    def exists(self, path: Pathlike) -> bool:
        path = Path(path).as_posix()
        if self.root is not None and not path.startswith(str(self.root)):
            path = os.path.join(self.root.as_posix(), path)
        from smb.smb_structs import OperationFailure
        try:
            self.connection.getAttributes(service_name=self._share_name, path=path)
            return True
        except OperationFailure:
            return False

    @classmethod
    def from_env(cls, root: Optional[Pathlike] = None, show_progress: bool = False) -> 'SMBFilesystem':
        return cls(
            ip=env_get('NAS_IP'),
            username=env_get('NAS_USERNAME'),
            password=env_get('NAS_PASSWORD'),
            share_name=env_get('NAS_SHARE'),
            root=root,
            show_progress=show_progress,
        )

    def get_root(self) -> Optional[Path]:
        return self.root

    def listdir(self, path: Pathlike) -> List:
        """
        Return:
        -------
        List[smb.base.SharedFile]: List of SharedFile objects
        """
        path = Path(path).as_posix()
        if self.root is not None and not path.startswith(str(self.root)):
            path = os.path.join(self.root.as_posix(), path)
        return self.connection.listPath(service_name=self._share_name, path=str(path))

    def mkdir(self, path: Pathlike, parents: bool = True) -> None:
        path = Path(path).as_posix()
        if self.root is not None and not path.startswith(self.root.as_posix()):
            path = os.path.join(self.root.as_posix(), path)
        path = Path(path)
        log(f'[{self.__class__.__name__}::mkdir] Creating directory "{str(path.as_posix())}"')
        acc_path = Path('/')
        path_parts = path.parts if parents else [str(path)]
        for path_part in path_parts:
            if path_part in ['.', '..', '/', '\\']:
                continue
            # check if it exists
            dirs = [d.filename for d in self.listdir(acc_path) if d.filename not in ['.', '..']]
            acc_path = acc_path / path_part
            if path_part not in dirs:
                # create new directory
                self.connection.createDirectory(service_name=self._share_name, path=acc_path.as_posix())

    def remove(self, path: Pathlike) -> None:
        path = Path(path).as_posix()
        if self.root is not None and not path.startswith(str(self.root)):
            path = os.path.join(self.root.as_posix(), path)
        self.connection.deleteFiles(service_name=self._share_name, path_file_pattern=str(path))

    def retrieve(self, path: Pathlike, local_path: Optional[Pathlike] = None) -> Union[Pathlike, bytes]:
        path = Path(path).as_posix()
        if self.root is not None and not path.startswith(str(self.root)):
            path = os.path.join(self.root.as_posix(), path)
        out = self.download(remote_path=str(path), local_path=local_path, show_progress=self._show_progress)
        if isinstance(out, io.BytesIO):
            out.seek(0)
            out = out.read()
        return out

    def rmdir(self, path: Pathlike) -> None:
        path = Path(path).as_posix()
        if self.root is not None and not path.startswith(str(self.root)):
            path = os.path.join(self.root.as_posix(), path)
        for p in self.connection.listPath(service_name=self._share_name, path=str(path)):
            if p.filename != '.' and p.filename != '..':
                parentPath = path
                if not parentPath.endswith('/'):
                    parentPath += '/'
                if p.isDirectory:
                    self.rmdir(parentPath + p.filename)
                    self.connection.deleteDirectory(service_name=self._share_name, path=parentPath + p.filename)
                else:
                    self.connection.deleteFiles(service_name=self._share_name,
                                                path_file_pattern=str(parentPath + p.filename))

    def set_root(self, path: Pathlike, mkdir: bool = True, **mkdir_kwargs) -> None:
        path = Path(path).as_posix()
        if self.root is not None and not path.startswith(str(self.root)):
            path = os.path.join(self.root.as_posix(), path)
        path = Path(path)
        if mkdir:
            self.mkdir(path, **mkdir_kwargs)
        self.root = path

    def store(self, local_file: Union[Pathlike, bytes], path: Pathlike) -> bool:
        path = Path(path).as_posix()
        if self.root is not None and not path.startswith(str(self.root)):
            path = os.path.join(self.root.as_posix(), path)
        if isinstance(local_file, bytes):
            local_file = io.BytesIO(local_file)
        return self.upload(local_path=local_file, remote_path=str(path), show_progress=self._show_progress)


if __name__ == '__main__':
    from misc import PathUtils

    _smb = SMBFilesystem.from_env()
    # exists
    print(_smb.exists(Path('/Thanos/Captures/Tue__13_08_2024__19_26_42/raw_depth/000/1723570010959.npy')))
    exit(0)
    # download
    mb = 540463 / 1024
    start = time.time()
    _bytes = _smb.download(remote_path='Thanos/Torch/hub/checkpoints/vgg16-397923af.pth',
                           local_path=PathUtils.resources_path(),
                           show_progress=True)
    end = time.time()
    print(f'Download speed: {mb / (end - start):.2f} MB/s')
    # upload
    start = time.time()
    _smb.upload(remote_path='Thanos/Torch/hub/checkpoints/vgg16-397923af2.pth',
                local_path=_bytes,
                show_progress=True)
    end = time.time()
    print(f'Download speed: {mb / (end - start):.2f} MB/s')
