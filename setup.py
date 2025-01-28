from setuptools import setup, find_packages

setup(
    name="segmentation",  # Nazwa pakietu
    version="0.1.0",  # Wersja pakietu

    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],

    entry_points={
        'console_scripts': [
            'run_training=segmentation.scripts.run_training:main',
            'run_evaluation=segmentation.scripts.run_evaluation:main',
        ],
    },
)
