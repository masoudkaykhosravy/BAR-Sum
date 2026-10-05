"""
=============================================================================
Package Setup: BAR-Sum
Project: Multi-Objective Deep Reinforcement Learning with Bounded Adaptive Rewards 
         for Factually Faithful Extractive Text Summarization
Author: Masoud Keikhosravi
Affiliation: Islamic Azad University, Mashhad Branch
Email: masoud.keikhosravi@iau.ac.ir
=============================================================================
"""

import os
from setuptools import setup, find_packages

# خواندن توضیحات README در صورت وجود
readme_path = os.path.join(os.path.dirname(__file__), "README.md")
long_description = ""
if os.path.exists(readme_path):
    with open(readme_path, "r", encoding="utf-8") as f:
        long_description = f.read()

setup(
    name="bar_sum",
    version="1.0.0",
    author="Masoud Keikhosravi",
    author_email="masoud.keikhosravi@iau.ac.ir",
    description="Multi-Objective Deep Reinforcement Learning with Bounded Adaptive Rewards for Extractive Summarization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/masoud-keikhosravi/BAR-Sum",
    
    # شناسایی پکیج‌های درون src و همچنین ماژول‌های مستقر در ریشه
    package_dir={"": "src"} if os.path.exists("src") else {},
    packages=find_packages(where="src") if os.path.exists("src") else find_packages(),
    py_modules=["plot_weights"],  # ثبت مستقیم ماژول‌های موجود در ریشه اصلی
    
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12.0",
        "transformers>=4.25.0",
        "datasets>=2.8.0",
        "rouge-score>=0.1.2",
        "scipy>=1.9.0",
        "numpy>=1.23.0",
        "pyyaml>=6.0",
        "matplotlib>=3.6.0",
        "tqdm>=4.64.0",
        "nltk>=3.7",
    ],
    classifiers=[
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
