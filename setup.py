from setuptools import setup, find_packages

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()
setup(
    name="yolo_model",
    version="0.1.0",
    author="Shin",
    author_email="NON-PUBLIC",
    description="YOLOv11 object detection with Zenoh integration",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/SlayM4yd4y/yolo_model",
    packages=find_packages(exclude=["tests*", "docs*", "examples*"]),
    install_requires=requirements,  
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU GENERAL PUBLIC LICENSE Version 3",
        "Operating System :: OS Independent",
    ],
    entry_points={
        "console_scripts": [
            "yolo_train=src.train:main",
            "yolo_detect=src.detect:main",
        ]
    },
)
