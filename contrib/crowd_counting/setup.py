from setuptools import setup, find_packages

setup(
    name="crowdcounting",
    version="0.0.1",
    description="A crowd counting package",
    author="Lixun Zhang",
    author_email="lxzhang@gmail.com",
    packages=find_packages(),
    install_requires=[
        "h5py==2.8.0", "flask==1.0.2", "scikit-image", "bokeh", "opencv-python>=4.2.0.32", "pillow", "Cython", "contextlib2"
    ]
)