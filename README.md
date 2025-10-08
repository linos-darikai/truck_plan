# Python Package Installation and Requirements Guide

## Installing Packages with pip

### Basic Installation

To install a single package:

```bash
pip install package-name
```

Example:
```bash
pip install requests
```

### Installing Specific Versions

Install a specific version:
```bash
pip install package-name==1.2.3
```

Install a minimum version:
```bash
pip install package-name>=1.2.0
```

Install within a version range:
```bash
pip install "package-name>=1.2.0,<2.0.0"
```

### Installing Multiple Packages

Install multiple packages at once:
```bash
pip install package1 package2 package3
```

### Installing from requirements.txt

Install all packages listed in a requirements file:
```bash
pip install -r requirements.txt
```

## Creating and Managing requirements.txt

### Manual Creation

Create a `requirements.txt` file and list packages line by line:

```text
requests==2.31.0
numpy>=1.24.0
pandas==2.0.3
flask>=2.3.0,<3.0.0
```

### Automatic Generation

Generate a requirements file from your current environment:

```bash
pip freeze > requirements.txt
```

This captures all installed packages with their exact versions.

### Best Practices for requirements.txt

**Option 1: Pin exact versions (most reproducible)**
```text
requests==2.31.0
numpy==1.24.3
pandas==2.0.3
```

**Option 2: Specify minimum versions (more flexible)**
```text
requests>=2.31.0
numpy>=1.24.0
pandas>=2.0.0
```

**Option 3: Version ranges (balanced approach)**
```text
requests>=2.31.0,<3.0.0
numpy>=1.24.0,<2.0.0
pandas>=2.0.0,<3.0.0
```

### Comments in requirements.txt

Add comments to organize your dependencies:

```text
# Core dependencies
requests==2.31.0
numpy==1.24.3

# Data processing
pandas==2.0.3
openpyxl==3.1.2

# Development dependencies
pytest==7.4.0
black==23.7.0
```

## Additional Tips

### Upgrade Packages

Upgrade a specific package:
```bash
pip install --upgrade package-name
```

Upgrade all packages in requirements.txt:
```bash
pip install --upgrade -r requirements.txt
```

### Uninstall Packages

Remove a package:
```bash
pip uninstall package-name
```

### List Installed Packages

View all installed packages:
```bash
pip list
```

View in requirements format:
```bash
pip freeze
```

### Using Virtual Environments

Always use virtual environments to isolate project dependencies:

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate

# Install packages
pip install -r requirements.txt
```

## Common Workflow

1. Create and activate virtual environment
2. Install packages as needed: `pip install package-name`
3. Generate requirements file: `pip freeze > requirements.txt`
4. Commit `requirements.txt` to version control
5. Team members clone repo and run: `pip install -r requirements.txt`

## Troubleshooting

If you encounter installation issues:

```bash
# Upgrade pip itself
pip install --upgrade pip

# Clear pip cache
pip cache purge

# Install with verbose output
pip install -v package-name
```