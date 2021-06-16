from setuptools import setup, find_namespace_packages

setup(
    name='vsdkx-model-mobilenet',
    url='https://gitlab.com/natix/visiondeploy/aiconnector',
    author='Helmut',
    author_email='helmut@natix.io',
    namespace_packages=['vsdkx', 'vsdkx.model'],
    packages=find_namespace_packages(include=['vsdkx*']),
    dependency_links=[
        'git+https://gitlab+deploy-token-485942:VJtus51fGR59sMGhxHUF@gitlab.com/natix/cvison/vsdkx/vsdkx-core.git#egg=vsdkx-core'
    ],
    install_requires=[
        'vsdkx-core',
        'opencv-python~=4.2.0.34',
        'numpy==1.18.5',
        'tensorflow==2.3.1',
    ],
    version='1.0',
)
