from setuptools import find_packages,setup
from typing import List

HYPEN_E_DOT='-e .'

def get_requirements(file_path:str)->List[str]:
    requires=[]

    with open(file_path) as file_obj:
        requires=file_obj.readlines()
        requires=[req.replace("\n","") for req in requires]

        if HYPEN_E_DOT in requires:
            requires.remove(HYPEN_E_DOT)

    return(requires)

setup(
    name='DS_PRO',
    version='0.0.1',
    author='NIRAJ_SHAW',
    author_email='Niraj123@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements("requirement.txt")
)
