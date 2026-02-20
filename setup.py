"""
Setup configuration for Satellite Change Detection System
"""
from setuptools import setup, find_packages
import os

# Read the README file
def read_file(filename):
    filepath = os.path.join(os.path.dirname(__file__), filename)
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    return ''

setup(
    name='changedetect',
    version='1.0.0',
    description='Satellite Image Change Detection using Siamese U-Net',
    long_description=read_file('README.md'),
    long_description_content_type='text/markdown',
    author='Harsha',
    author_email='',
    url='https://github.com/Harsha2318/Satellite-Change-Detection-System',
    license='MIT',
    
    packages=find_packages(),
    
    python_requires='>=3.8',
    
    install_requires=[
        'numpy>=1.24.0',
        'pillow>=10.0.0',
    ],
    
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'flake8>=4.0.0',
            'black>=22.0.0',
        ],
        'full': [
            'pandas>=1.4.0',
            'scipy>=1.8.0',
            'matplotlib>=3.5.0',
            'scikit-image>=0.19.0',
            'scikit-learn>=1.0.0',
            'tqdm>=4.62.0',
            'joblib>=1.1.0',
            'PyYAML>=6.0',
            'rasterio>=1.3.0',
            'geopandas>=0.12.0',
            'shapely>=2.0.0',
            'pyproj>=3.3.0',
            'fiona>=1.8.0',
            'rtree>=1.0.0',
            'torch>=2.0.0',
            'torchvision>=0.15.0',
            'tensorboard>=2.10.0',
            'albumentations>=1.1.0',
            'opencv-python>=4.5.0',
            'timm>=0.5.4',
            'gdal>=3.4.0',
        ]
    },
    
    entry_points={
        'console_scripts': [
            'changedetect=changedetect.src.main:main',
        ],
    },
    
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Image Recognition',
    ],
    
    keywords='satellite imagery change detection deep learning',
)
