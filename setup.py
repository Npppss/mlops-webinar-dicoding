from setuptools import find_packages, setup
from typing import List


HYPEN_E_DOT='-e .'
def get_requirements(file_path:str) -> List[str]:
    '''This function will return the list of requirements'''
    requirments=[]
    with open(file_path) as file_obj:
        requirments=file_obj.readlines()
        requirments=[req.replace("\n","") for req in requirments]
    if HYPEN_E_DOT in requirments:
        requirments.remove(HYPEN_E_DOT)

        return requirments

setup(
    name="Mlops-Project",
    version="0.0.1",
    author="Novan",
    author_email="novandrae7980@gmail.com",
    package=find_packages(),
    install_requires=get_requirements('requirements.txt')
)