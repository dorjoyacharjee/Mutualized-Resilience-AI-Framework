# Contributing Guidelines

Thank you for your interest in contributing to the Climate-Adaptive Agricultural Insurance AI Framework!

## Code of Conduct

This project adheres to professional research standards. Please be respectful and collaborative.

## How to Contribute

### Reporting Bugs

If you find a bug:
1. Check if it's already reported in Issues
2. Create a new issue with:
   - Clear title
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details (OS, Python version, package versions)
   - Error messages and stack traces

### Suggesting Enhancements

For feature requests:
1. Check existing issues for duplicates
2. Describe the use case clearly
3. Explain why it would benefit the research community
4. Provide example scenarios

### Code Contributions

#### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/yourusername/climate-agricultural-insurance-ai.git
cd climate-agricultural-insurance-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

#### Coding Standards

- **Style:** Follow PEP 8 (use `black` for formatting)
- **Docstrings:** Use Google style docstrings
- **Type hints:** Include type annotations for all functions
- **Comments:** Explain *why*, not *what*
- **Imports:** Organize with `isort`

#### Testing

```bash
# Run tests
pytest tests/

# Check coverage
pytest --cov=src tests/

# Run linter
flake8 src/

# Run type checker
mypy src/
```

#### Pull Request Process

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Make your changes
4. Add/update tests
5. Update documentation
6. Run tests and linters
7. Commit with clear messages
8. Push to your fork
9. Open a Pull Request with:
   - Description of changes
   - Related issue number
   - Test results
   - Screenshots (if UI changes)

#### Commit Messages

Format:
```
type(scope): brief description

Detailed explanation if needed.

Fixes #issue_number
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

Example:
```
feat(mrep): add ensemble LSTM models

Implemented ensemble averaging of 5 LSTM models
to improve loss forecasting accuracy by 12%.

Fixes #42
```

### Documentation

- Update README.md for user-facing changes
- Add docstrings for new functions/classes
- Update DATA_SOURCES.md for new data
- Create tutorials in `notebooks/` for major features

### Research Reproducibility

For changes affecting research outputs:
1. Document all parameters
2. Save random seeds
3. Include example runs
4. Update validation results
5. Cross-reference with paper

## Project Structure

```
climate-agricultural-insurance-ai/
├── src/                 # Source code
│   ├── data_acquisition/
│   ├── preprocessing/
│   ├── models/
│   ├── evaluation/
│   └── visualization/
├── tests/              # Test suite
├── notebooks/          # Jupyter tutorials
├── data/               # Data directories (gitignored)
├── outputs/            # Results (gitignored)
└── docs/               # Documentation
```

## Questions?

- Open a discussion in GitHub Discussions
- Email: [your-email@institution.edu]

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
