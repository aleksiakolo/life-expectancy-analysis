# Architecture Outline

## Purpose

This document defines the **full target architecture** for the `life-expectancy-analysis` project.

The goal is to transform the project into:

- a **reproducible data science system**
- a **proper Python package**
- a **CLI-driven workflow**
- a **clean separation of logic vs exploration**
- a **production-quality, interview-ready codebase**

This includes:
- what exists today
- what we are building now
- what MUST be implemented to reach the final state

---

## 1. Package Structure

### Target package name
`life_expectancy_analysis`

### Target structure (final)

```text
life-expectancy-analysis/
│
├── src/
│   └── life_expectancy_analysis/
│       ├── data/
│       ├── analysis/
│       ├── modeling/
│       ├── utils/
│       ├── config/
│       ├── cli/
│       └── **init**.py
│
├── configs/
│   ├── default.yaml
│   ├── dev.yaml
│   └── experiments/
│
├── notebooks/
├── tests/
├── docs/
├── pyproject.toml
└── README.md
```

### Current vs target

Current:
- `src/` directly contains modules

Target:
- `src/life_expectancy_analysis/` becomes the real package

### Required work
- full package tree migration
- update imports
- ensure installability

---

## 2. Packaging System

### Standard
Use `pyproject.toml`

### Responsibilities
- define metadata
- define dependencies
- register CLI entrypoint
- support editable installs

### Required improvements
- remove oversized / unstructured `requirements.txt`
- introduce dependency locking:
  - `uv`, `poetry`, or `pip-tools`
- ensure reproducible environments

---

## 3. CLI System

### CLI name
`lifeexp`

### Philosophy
CLI = **thin orchestration layer**

- no business logic
- only calls into `src/`

### Target commands

```bash
lifeexp data load
lifeexp data merge
lifeexp preprocess run
lifeexp features build
lifeexp model train
lifeexp model evaluate
lifeexp diagnostics run
lifeexp pipeline run
```

### Responsibilities
- parse arguments
- load config
- call functions from `src/`
- log outputs

### Required work
- CLI skeleton
- modular command structure
- integration with config system

---

## 4. Configuration System

### Design

- YAML-based config files
- loaded via `src/config/`
- passed via CLI

### Example

```bash
lifeexp pipeline run --config configs/default.yaml
````

### Responsibilities

Config should define:

* data paths
* output paths
* preprocessing parameters
* feature settings
* model hyperparameters
* random seeds

### Structure

```text
configs/
├── default.yaml
├── dev.yaml
└── experiments/
```

### Required improvements

* remove hardcoded values from notebooks
* centralize all runtime decisions
* support multiple environments

---

## 5. Data Layer (`src/data/`)

### Responsibilities

* data loading
* schema standardization
* dataset merging
* basic validation

### Principles

* pure functions when possible
* explicit inputs/outputs
* no hidden state

### Required improvements

* ensure all loading logic is reusable
* unify naming conventions
* expose key functions to CLI

---

## 6. Preprocessing & Feature Engineering

### Responsibilities

* cleaning
* missing value handling
* normalization
* transformations
* feature creation

### Principles

* deterministic
* reproducible
* parameterized via config

### Required improvements

* move logic out of notebooks
* ensure pipeline compatibility
* integrate with modeling pipeline

---

## 7. Modeling Layer (`src/modeling/`)

### Responsibilities

* train/test splitting
* pipeline creation
* model training
* evaluation

### Submodules

* pipelines
* baselines
* advanced models
* temporal models
* sequence models
* experiment runner

### Principles

* use sklearn-style pipelines
* modular components
* configurable via config

### Required improvements

* standardize interfaces
* unify training workflow
* integrate with CLI

---

## 8. Analysis & Diagnostics (`src/analysis/`)

### Responsibilities

* error analysis
* diagnostics
* evaluation summaries
* plotting

### Principles

* reusable utilities
* separate computation from plotting

### Required improvements

* ensure no duplication in notebooks
* expose reusable diagnostics functions

---

## 9. Utilities (`src/utils/`)

### Responsibilities

* IO helpers
* plotting utilities
* common helpers

### Required improvements

* centralize file handling
* standardize output formats
* avoid duplication across modules

---

## 10. Notebooks

### Allowed use

* EDA
* visualization
* interpretation
* reporting

### Not allowed

* core pipeline logic
* reusable preprocessing
* production workflows

### Required work

* audit all notebooks
* identify duplicated logic
* migrate reusable code to `src/`
* simplify notebooks to orchestration + visualization

---

## 11. Testing

### Scope

* unit tests for core functions
* integration tests for pipelines

### Structure

```text
tests/
├── data/
├── modeling/
├── analysis/
```

### Required work

* test critical functions first
* ensure deterministic outputs
* validate edge cases

---

## 12. CI/CD

### Goals

* automated testing
* linting
* formatting

### Tools

* GitHub Actions
* pytest
* ruff / black

### Required work

* CI pipeline setup
* enforce code quality

---

## 13. Dependency Management

### Goals

* reproducibility
* minimal dependencies

### Required work

* replace large requirements file
* introduce locking
* ensure environment consistency

---

## 14. Pipeline Orchestration

### Goal

Run full workflow via CLI

```bash
lifeexp pipeline run --config configs/default.yaml
```

### Steps

1. load data
2. preprocess
3. feature engineering
4. train model
5. evaluate
6. generate diagnostics

### Required work

* define pipeline entrypoint
* connect modules
* ensure reproducibility

---

## 15. Logging & Outputs

### Goals

* consistent outputs
* traceability

### Required work

* standard output directories
* structured logging
* save intermediate artifacts

---

## 16. Documentation

### Required files

* README (quickstart)
* architecture.md
* module-level docstrings

### Goals

* easy onboarding
* clear explanation of architecture
