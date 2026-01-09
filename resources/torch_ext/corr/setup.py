import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

from utils.misc import env_get, PathUtils

os.environ["CC"] = env_get("CC", "gcc-7")
os.environ["CXX"] = env_get("CXX", "g++-7")
USE_NINJA = env_get('USE_NINJA', '1') == '1'

setup(
    name='corr_sampler',
    ext_modules=[
        CUDAExtension('corr_sampler', [
            str(PathUtils.torch_extension_path('corr') / 'sampler.cpp'),
            str(PathUtils.torch_extension_path('corr') / 'sampler_kernel.cu'),
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension.with_options(use_ninja=USE_NINJA)
    })
