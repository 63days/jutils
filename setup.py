from setuptools import setup

setup(
    name="jutils",
    version=0.1,
    packages=["jutils"],
    zip_safe=False,
    install_requires = [
        "numpy<2", # numpy version 1
        "pillow",
        "opencv-python",
        "torch",
    ]
)
