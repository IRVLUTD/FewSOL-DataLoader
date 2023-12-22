import setuptools

with open("README.md", "r", encoding = "utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name = "FewSOLDataLoader",
    version = "0.0.4",
    author = "Jesse Musa at IRVLUTD",
    author_email = "JOM210001@utdallas.edu",
    description = "Pytorch Dataloader for FewSOL",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/IRVLUTD/FewSOL-DataLoader",
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir = {"": "src"},
    packages = setuptools.find_packages(where="src"),
    python_requires = ">=3.0",
    install_requires=['matplotlib', 'numpy', 'torch', 'torchvision', 'scipy', 'pyyaml', 'easydict', 'transforms3d'],
    package_data={'FewSOLDataLoader': ['syn_google_scenes_data_mapper.json']}
)