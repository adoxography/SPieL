from setuptools import setup

setup(name='spiel',
      version='0.1.0',
      description='Segmentation of polysynthetic languages',
      url='https://github.com/adoxography/SPieL',
      author='Graham Still',
      author_email='gstill@uw.edu',
      license='MIT',
      packages=['spiel'],
      entry_points={
          'console_scripts': ['spiel=spiel.command_line:main']
      },
      setup_requires=[
          'nose==1.3.7',
          'noseOfYeti==1.8',
          'nose-pathmunge'
      ],
      install_requires=[
          'numpy==1.15.4',
          'scikit-learn==0.20.1',
          'sklearn-crfsuite',
      ],
      test_suite='nose.collector',
      tests_require=[
          'coverage==4.5.2'
      ],
      zip_safe=False)
