# Some-Name-Here

INTENT

WHAT THE PROJECT DOES





## Table of Contents

- [Acknowledgements](#acknowledgements)
- [Environment Setup](#environment-setup)
- [General Project Process](#general-project-process)
- [Objectives](#objectives)
- [Known Issues](#known-issues)





## Acknowledgements

[Back to Table of Contents](#table-of-contents)

- **DATASET:** [DEEPSIG RADIOML 2018.01A](https://www.deepsig.ai/datasets/)
    - **LICENSE:** [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
    - **CHANGES MADE:**
        - Dataset reduced down to only SNR 30 for baseline establishment
        - This reduced dataset was then used to feature engineer a multitude of seemingly typical information of a signals intelligence anaylsts' pipeline





## Environment Setup

[Back to Table of Contents](#table-of-contents)

Python version in both environments: `VERSION HERE`

You have two options for setting up your Python environment:

### Option 1: Conda (Recommended)

**Conda** is an open-source environment and package manager that makes it easy to manage Python versions and dependencies. If you do not already use an environment manager, you may want to familiarize yourself with one since it helps avoid conflicts and makes reproducibility easier.  I use Conda and I think it's the easiest (Though I haven't used other packages)

**Steps:**
1. Install [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
2. Clone this repository (Or just download ```environment.yml```).
3. Create the environment using the provided `environment.yml`:
	```bash
	conda env create -f environment.yml
	conda activate RF_HACKING_AI
	```

### Option 2: pip (Use with Caution)

You can also use `pip` with the `environment.txt` file. Using pip does not manage Python versions, so you must ensure your Python version matches the requirements.

**Steps:**
1. Ensure you are using a compatible Python version (see above).
2. Clone this repository (Or just download environment.txt).
3. Install dependencies:
	```bash
	pip install -r environment.txt
    ```





## General Project Process

[Back to Table of Contents](#table-of-contents)

STUFF





## Objectives

[Back to Table of Contents](#table-of-contents)

- [ ] Acquire Dataset
- [ ] Prepare Dataset
- [ ] Explore Dataset
- [ ] Feature Engineering
- [ ] Select Models
- [ ] Establish Baseline Models
- [ ] Optimize Baseline Models
- [ ] Establish "Better" Models
- [ ] Optimize "Better" Models
- [ ] Presentation and Report Creation
- [ ] Future Implementations





## Known Issues

[Back to Table of Contents](#table-of-contents)

- `environment.yml` and `requirements.txt` is not created yet
