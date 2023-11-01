import setuptools

with open("README.md", "r", encoding = "utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name = "FewSOLDataLoader",
    version = "0.0.1",
    author = "Jesse Musa at IRVLUTD",
    author_email = "JOM210001@utdallas.edu",
    description = "Pytorch Dataloader for FewSOL",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "package URL",
    project_urls = {
        "Bug Tracker": "package issues URL",
    },
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir = {"": "src"},
    packages = setuptools.find_packages(where="src"),
    python_requires = ">=3.0",
    install_requires=['matplotlib', 'numpy', 'torch', 'torchvision', 'scipy', 'pyyaml', 'easydict', 'transforms3d']
)