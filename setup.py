import setuptools

with open("README.md", "r", encoding = "utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name = "FewSOLDataLoader",
    version = "0.0.14",
    author = "Jesse Musa, Jishnu P",
    author_email = "jesse.musa@utdallas.edu, jishnu.p@utdallas.edu",
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
    install_requires=['matplotlib', 'numpy', 'torch', 'torchvision', 'scipy', 'pyyaml', 'easydict', 'transforms3d', 'opencv-python'],
    package_data={'FewSOLDataLoader': ['syn_google_scenes_data_mapper.json', 'file_class_mapper.json']}
)