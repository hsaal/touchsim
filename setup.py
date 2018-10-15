from setuptools import setup

setup(name='touchsim',
      version='0.1',
      description='TouchSim: Simulating tactile signals from the whole hand with millisecond precision',
      url='https://bitbucket.org/hsaal/touch-sim',
      classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3.6',
      ],
      author='Hannes Saal',
      author_email='h.saal@sheffield.ac.uk',
      packages=['touchsim'],
      install_requires=[
          'numpy','scipy','numba','matplotlib','holoviews','scikit-image'
      ],
      zip_safe=False,
      data_files=[('surfaces',['surfaces/hand.tiff'])])
