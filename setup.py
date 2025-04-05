from setuptools import setup

setup(name='touchsim',
      version='0.1.1',
      description='TouchSim: Simulating tactile signals from the whole hand with millisecond precision',
      url='https://github.com/hsaal/touchsim',
      classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3.6',
      ],
      author='Hannes Saal',
      author_email='h.saal@sheffield.ac.uk',
      packages=['touchsim'],
      install_requires=[
          'numpy','scipy','numba','matplotlib','scikit-image'
      ],
      zip_safe=False,
      data_files=[('surfaces',['surfaces/hand.png'])])
