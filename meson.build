project('stLENS', 'cpp',
  version : '0.1.0.dev10',
  default_options : ['warning_level=3',
                     'cpp_std=c++11'])

# Find OpenMP for perturb_omp.cpp
openmp = dependency('openmp', required : false)

# Get Python installation
py = import('python').find_installation(pure: false)

# Define the source directory
src_dir = 'src/stLENS'

# Define C++ sources and their corresponding shared libraries
cpp_sources = [
  {
    'name': 'random_matrix',
    'source': src_dir / 'random_matrix.cpp',
    'dependencies': []
  },
  {
    'name': 'perturb_omp', 
    'source': src_dir / 'perturb_omp.cpp',
    'dependencies': openmp.found() ? [openmp] : []
  }
]

# Build shared libraries
foreach cpp_source : cpp_sources
  shared_library(
    cpp_source['name'],
    cpp_source['source'],
    dependencies: cpp_source['dependencies'],
    name_suffix: 'so',
    install: true,
    install_dir: py.get_install_dir() / 'stLENS',
    cpp_args: ['-fPIC', '-O3']
  )
endforeach

# Install Python modules
py.install_sources(
  [
    src_dir / '__init__.py',
    src_dir / 'stLENS.py', 
    src_dir / 'PCA.py',
    src_dir / 'calc.py'
  ],
  subdir: 'stLENS'
)