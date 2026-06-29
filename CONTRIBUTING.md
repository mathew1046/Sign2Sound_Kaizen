# Contributing to GloveTalk

Thank you for your interest in contributing to GloveTalk! This is an open-source assistive communication project, and we welcome contributions of all kinds — bug reports, feature suggestions, documentation improvements, code changes, and hardware designs.

## How to Contribute

### 1. Reporting Issues

Check the [issue tracker](https://github.com/mathew1046/Sign2Sound_Kaizen/issues) first to avoid duplicates. When opening a new issue, include:
- A clear title and description
- Steps to reproduce (for bugs)
- Relevant logs, screenshots, or error messages
- Environment details (OS, Python version, hardware)

### 2. Suggesting Features

Open a feature request issue describing the problem you want to solve and your proposed solution. We especially welcome ideas around:
- New sign language vocabulary and model improvements
- Hardware sensor enhancements
- Deployment and edge computing optimizations
- Accessibility and UX improvements

### 3. Code Contributions

#### Getting Started

```bash
git clone https://github.com/mathew1046/Sign2Sound_Kaizen.git
cd Sign2Sound_Kaizen
pip install -r requirements.txt
```

#### Making Changes

1. Create a branch: `git checkout -b your-feature-name`
2. Make your changes following the existing code style
3. Test your changes locally
4. Commit with a clear message describing what and why

#### Guidelines

- Follow the existing project structure and naming conventions
- Keep code modular and well-organized within existing directories
- Update `requirements.txt` if adding new dependencies
- Add or update tests in `tests/` when applicable
- Update docs if changing public interfaces or adding features

### 4. Hardware Contributions

For hardware-related contributions (ESP32 firmware, sensor configurations, PCB designs), please include:
- Schematics or wiring diagrams
- Bill of materials for any new components
- Calibration procedures if applicable

### 5. Pull Request Process

1. Push your branch and open a Pull Request
2. Describe what your PR does and link any related issues
3. Ensure the PR passes any existing tests
4. We'll review and provide feedback — expect constructive discussion

## Code of Conduct

Be respectful and inclusive. This project is built by a small team for a competition, but we welcome collaborators from all backgrounds. Harassment or disrespectful behavior will not be tolerated.

## Need Help?

Open a discussion or issue if you have questions about the codebase, hardware setup, or want to discuss a contribution before diving in.
