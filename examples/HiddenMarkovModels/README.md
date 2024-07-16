# Getting Started

In general, when using the HiddenMarkovModels package, you need to configure your Bonsai workflow to run the correct python environment which has the [ssm](https://github.com/lindermanlab/ssm) package installed.

If you followed the installation guide, you will have a basic Bonsai-python environment for using the LinearDynamicalSystems package. You will also have installed the basic dependencies required to build and run the Bonsai workflows. However, if you wish to run the examples, we highly recommend following this guide to create and configure the environments needed for each example. This guide has been made for users starting from scratch to get started. Alternatively, you can use the environment you created in the previous guide and install the necessary package dependencies required to run each workflow.

All of the examples are self-contained and can be found by going to the [Bonsai.ML - Examples](https://github.com/bonsai-rx/machinelearning-examples) repo. For each example, you can bootstrap the python and Bonsai environments from within each example directory. Below, we demonstrate how to get started using the examples by bootstrapping the example environments that are needed to run the example workflow.

### Windows

#### Dependencies

You must have the following installed on your PC in order to run the example workflows:

- [Python (v3.10)](https://www.python.org/downloads/) *Note: python must be installed using the windows installer and must be added to the system PATH*
- [Git](https://git-scm.com/downloads)

#### Instructions

Open the terminal or powershell. Start by cloning the [Bonsai.ML - Examples](https://github.com/bonsai-rx/machinelearning-examples) repo with:

```cmd
git clone https://github.com/bonsai-rx/machinelearning-examples.git
```

Change directory to one of the examples:

```cmd
cd .\machinelearning-examples\examples\HiddenMarkovModels\ForagingMouse
```

To create the python virtual environment and install the package, run the following:

```cmd
python -m venv .venv 
.\.venv\Scripts\activate
pip install numpy cython
pip install ssm@git+https://github.com/lindermanlab/ssm@6c856ad3967941d176eb348bcd490cfaaa08ba60
```

> [!WARNING]
> The latest version of numpy (v2.0) is incompatible with the ssm package. To install the latest compatible version of numpy, use:
> ```cmd
> pip install numpy==1.26.4
> ```

To create the bonsai environment and install the packages, run the powershell script:

```cmd
cd .bonsai
powershell .\Setup.ps1
```

Once installed, run the bonsai executable with:

```cmd
.\Bonsai.exe
```

Open the workflow example and start the bonsai workflow.

### Linux

#### Notes on Running Bonsai in Linux

Currently, the examples have only been tested on Ubuntu 22.04. Running Bonsai on Linux is still being tested and should be used with caution. We cannot guarantee that these instructions will work for all Linux distributions or versions. It is important that you consult the general [Bonsai installation guide on Linux](https://github.com/orgs/bonsai-rx/discussions/1101) to ensure that the underlying Bonsai package dependencies are met and installed properly.

#### Dependencies

You must have the following installed on your PC in order to run the example workflows:

- [Python (v3.10)](https://www.python.org/downloads/) *comes installed with the latest version of Ubuntu 22.04*
- [Git](https://git-scm.com/downloads)
- [Mono](https://www.mono-project.com/download/stable/#download-lin)
- [OpenCV and OpenGL binaries](https://github.com/orgs/bonsai-rx/discussions/1101) 

#### Instructions

Open up a terminal and start by cloning the [Bonsai.ML - Examples](https://github.com/bonsai-rx/machinelearning-examples) repo:

```cmd
git clone https://github.com/bonsai-rx/machinelearning-examples.git
```

Change directory to one of the examples. For instance:

```cmd
cd machinelearning-examples/examples/LinearDynamicalSystems/Kinematics/SimulatedData
```

Create the python virtual environment and install the lds python package with the following:

```cmd
python3 -m venv .venv 
source .venv/bin/activate
pip install numpy cython
pip install ssm@git+https://github.com/lindermanlab/ssm@6c856ad3967941d176eb348bcd490cfaaa08ba60
```

> [!WARNING]
> The latest version of numpy (v2.0) is incompatible with the ssm package. To install the latest compatible version of numpy, use:
> ```cmd
> pip install numpy==1.26.4
> ```

Run the `Setup.sh` script using the following:

```cmd
cd .bonsai
./Setup.sh
```

> [!TIP]
> The `Setup.sh` script uses the `xmllint` and `xmlstarlet` packages to convert the assembly location paths in the bonsai config file from Windows to Linux paths. If you do not have these packages already installed on your computer, you can install the `xmllint` and the `xmlstarlet` packages using:
> ```cmd
> sudo apt install -y libxml2-utils xmlstarlet
> ```

Once the bonsai environment has been created, activate the bonsai environment and run the Bonsai executable:

```cmd
source ./activate
bonsai
```

> [!TIP]
> If your desktop theme is set to dark mode, the Bonsai GUI can display in weird ways due to the way that mono tries to use the desktop theme in applications. If you use the `bonsai-clean` command instead of the `bonsai` command, this will reset the theme that mono uses to the default theme, which can imporove the appearance of the Bonsai GUI.

Open the workflow example and start the bonsai workflow.
