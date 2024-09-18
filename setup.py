from setuptools import setup, find_packages

setup(
    name='special_pca',
    version='0.1',
    packages=find_packages('src'),  # 这里会找到src目录下的所有包
    package_dir={'': 'src'},
    install_requires=[
        # 列出项目的依赖库
        'numpy',
        'pandas',
        'scikit-learn',
        # ...
    ],
)
