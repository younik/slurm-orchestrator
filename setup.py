import setuptools

install_requires = [
    'wandb',
]


setuptools.setup(
    name='slurm-orchestrator',
    version='0.0.1a',
    description='Slurm launcher for Machine Learning',
    url='https://github.com/younik/slurm-orchestrator',
    author='Omar G. Younis',
    author_email='omar.younis98@gmail.com',
    keywords='Machine Learning, Slurm, Cluster, Computing, Experiments',
    packages=setuptools.find_packages(),
    python_requires=">=3.8",
    include_package_data=True,
    install_requires=install_requires,
    extras_require={},
    cmdclass={},
)
