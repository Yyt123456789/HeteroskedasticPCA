from setuptools import setup, find_packages

setup(
    name='special_pca',
    version='0.1',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=[
        # 这里列出项目的依赖库
        'numpy',
        'pandas',
        'scikit-learn',
        # ...
    ],
)
