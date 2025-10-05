# GitHub Workflows for Protpardelle-1c

This directory contains GitHub Actions workflows for the Protpardelle-1c project. These workflows provide comprehensive CI/CD, testing, code quality, and deployment automation.

## Workflow Overview

### 1. CI (`ci.yml`)

**Purpose**: Main continuous integration pipeline
**Triggers**: Push to main/develop, pull requests
**Features**:

- Multi-Python version testing (3.10, 3.11, 3.12)
- Linting with ruff, black, isort
- Type checking with mypy
- Unit and integration tests
- GPU testing (on main branch)
- Security scanning with Trivy
- Package building and validation

### 2. Release (`release.yml`)

**Purpose**: Automated package releases
**Triggers**: Version tags, manual dispatch
**Features**:

- Package building with setuptools
- PyPI publishing (requires PYPI_API_TOKEN secret)
- GitHub release creation
- Pre-release detection

### 3. Dependency Update (`dependency-update.yml`)

**Purpose**: Automated dependency updates
**Triggers**: Weekly schedule, manual dispatch
**Features**:

- Dependency version checking
- Automated PR creation for updates
- Cache optimization

### 4. Model Validation (`model-validation.yml`)

**Purpose**: Validate model files and functionality
**Triggers**: Changes to model files, core modules, sampling code
**Features**:

- Model loading validation
- Sampling pipeline testing
- Configuration validation
- Smoke tests for core functionality

### 5. Code Quality (`code-quality.yml`)

**Purpose**: Code quality and security analysis
**Triggers**: Push/PR to main/develop
**Features**:

- Linting with ruff, black, isort
- Type checking with mypy
- Security scanning with Bandit and Trivy
- Code complexity analysis
- Auto-formatting PR creation

### 6. Documentation (`docs.yml`)

**Purpose**: Documentation building and deployment
**Triggers**: Changes to docs, README, source code
**Features**:

- Sphinx documentation generation
- Markdown linting
- GitHub Pages deployment
- API documentation from docstrings

### 7. Status (`status.yml`)

**Purpose**: Daily health checks and monitoring
**Triggers**: Daily schedule, manual dispatch
**Features**:

- Basic health checks
- Test coverage reporting
- Codecov integration

## Required Secrets

To enable full functionality, configure these repository secrets:

### For PyPI Publishing

- `PYPI_API_TOKEN`: PyPI API token for package publishing

### For Documentation Deployment

- `GITHUB_TOKEN`: Automatically provided by GitHub

## Workflow Dependencies

The workflows use the following tools and services:

### Python Environment

- **uv**: Fast Python package manager
- **Python 3.10+**: Primary supported version
- **PyTorch**: CPU and CUDA variants

### Code Quality Tools

- **ruff**: Fast Python linter
- **black**: Code formatter
- **isort**: Import sorter
- **mypy**: Type checker
- **bandit**: Security linter
- **trivy**: Vulnerability scanner

### Testing Framework

- **pytest**: Test runner
- **pytest-cov**: Coverage reporting
- **pytest-xdist**: Parallel testing (optional)

### Documentation

- **sphinx**: Documentation generator
- **sphinx-rtd-theme**: Read the Docs theme
- **myst-parser**: Markdown support

## Usage Guidelines

### For Contributors

1. **Pull Requests**: All PRs trigger CI, code quality, and model validation workflows
2. **Code Style**: Auto-formatting PRs are created for style fixes
3. **Testing**: Ensure tests pass before merging
4. **Documentation**: Update docstrings and README as needed

### For Maintainers

1. **Releases**: Tag releases with `v*` format to trigger release workflow
2. **Dependencies**: Weekly dependency updates are automated
3. **Security**: Regular security scans are performed
4. **Monitoring**: Daily status checks ensure project health

### For Users

1. **Installation**: Use `pip install protpardelle` for stable releases
2. **Development**: Clone repository and follow setup instructions
3. **Issues**: Report issues through GitHub Issues
4. **Documentation**: Check GitHub Pages for latest documentation

## Workflow Customization

### Adding New Tests

1. Add test files to `tests/` directory
2. Use appropriate pytest markers (`@pytest.mark.slow`, `@pytest.mark.gpu`, etc.)
3. Update workflow test commands if needed

### Modifying Code Quality Checks

1. Update tool configurations in workflow files
2. Add new tools to the `code-quality.yml` workflow
3. Configure tool-specific settings in `pyproject.toml`

### Extending Documentation

1. Add documentation files to `docs/` directory
2. Update Sphinx configuration in workflow
3. Modify deployment settings for GitHub Pages

## Troubleshooting

### Common Issues

1. **Test Failures**: Check test markers and dependencies
2. **Linting Errors**: Run `ruff check` and `black` locally
3. **Type Errors**: Use `mypy` for type checking
4. **Security Issues**: Review Trivy and Bandit reports

### Performance Optimization

1. **Cache Dependencies**: Workflows use uv cache for faster builds
2. **Parallel Testing**: Use pytest-xdist for parallel test execution
3. **Selective Testing**: Use pytest markers to run specific test subsets

### Monitoring

1. **Workflow Status**: Check GitHub Actions tab for workflow status
2. **Coverage Reports**: View Codecov reports for test coverage
3. **Security Alerts**: Monitor GitHub Security tab for vulnerabilities
4. **Documentation**: Check GitHub Pages for documentation status

## Contributing to Workflows

When modifying workflows:

1. **Test Locally**: Use `act` to test workflows locally
2. **Follow Best Practices**: Use official actions and maintain security
3. **Document Changes**: Update this README when adding new workflows
4. **Version Actions**: Pin action versions for reproducibility
5. **Security**: Use minimal permissions and avoid hardcoded secrets

## Support

For workflow-related issues:

1. Check workflow logs in GitHub Actions
2. Review this documentation
3. Open an issue with workflow details
4. Contact maintainers for complex issues
