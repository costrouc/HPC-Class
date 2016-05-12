README - Strassen Parallel Matrix Multiply

Written by Chris Ostrouchov for CPSC 594.
CAPS (Communication-Avoiding Parallel Strassen) is implemented with respect to algorithm described in (1).
It works on 7^k processors ((1) does it for {1, 2, 3, 6}\*7^k processors) and matricies of size $n = 2^{s}\*7^{k/2}$.
This algorithm has a __HUGE__ importance as it has been shown to be 2x faster than scalapack implementation (growing larger as P increases).
See refernce (1) for a detailed explanation of the algorithm or alternatively read my writup for the project in the `doc/` directory.

# Dependencies
 - CBlas (depends on cblas for local matrix multiplication)
 - PAPI for timing
 - R (ggplot) for creating graphs `make plot`

# Install
To install simply type
`make`

# Make Commands
 - make (Compile Program)
 - make test (Run __Short__ Tests for quick demonstration only uses 7 processors)
 - make test_long (You will need a super computer for this with 7^k where k = 1, 2, 3, 4. processors: 2401. And the tests will take a long time). 
 - make debug (you will not need to use this... used when making the code. The MACROS have been removed from the code for readability)
 - make plot  (uses gglot2 to create nice graphs of data. Images go to `img/`)
 
# Directory Structure
src - source files for strassen 
data - results from tests (`make test_long` *CAREFUL NEEDS SUPERCOMPUTER with 2401 processors*) in csv(tab) format. Use `make test` for demonstration (uses 7 processors).
img - R plots to create images from data (`make plot`)
doc - Writeup on Project (available in markdown format + pdf)
tools - helper scripts to create (images and run tests etc.)

# References
(1) Grey Ballard, James Demmel, Olga Holtz, Benjamin Lipshitz, Oded Schwartz, __Communication-Optimal Parallel Algorithm for Strassen's Matrix Multiplication__, *2012*
(2) Grey Ballard, James Demmel, Olga Holtz, Benjamin Lipshitz, Oded Schwartz, __Communication-Avoiding Parallel Strassen: Implementation and Performace__, *2012*
