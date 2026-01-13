# GRCWA Documentation

This directory contains comprehensive documentation for GRCWA (autoGradable Rigorous Coupled Wave Analysis) built with MkDocs.

## Documentation Structure

```
docs_mkdocs/
├── index.md                    # Home page
├── introduction.md             # Introduction to RCWA
├── installation.md             # Installation guide
├── quickstart.md               # Quick start tutorial
├── theory/                     # Theory and mathematics
│   ├── principles.md           #   Physical principles
│   ├── mathematics.md          #   Mathematical formulation
│   └── algorithm.md            #   RCWA algorithm
├── guide/                      # User guides
│   ├── concepts.md             #   Basic concepts
│   ├── layers.md               #   Layer definition
│   ├── excitation.md           #   Excitation setup
│   ├── results.md              #   Computing results
│   └── fields.md               #   Field analysis
├── api/                        # API reference
│   ├── core.md                 #   Core classes
│   ├── layers.md               #   Layer methods
│   ├── solver.md               #   Solver methods
│   ├── fields.md               #   Field methods
│   └── utilities.md            #   Utility functions
├── tutorials/                  # Step-by-step tutorials
│   ├── tutorial1.md            #   Simple slab
│   ├── tutorial2.md            #   Patterned layer
│   ├── tutorial3.md            #   Multiple layers
│   ├── tutorial4.md            #   Hexagonal lattice
│   └── tutorial5.md            #   Optimization
├── examples/                   # Example gallery
│   ├── gallery.md              #   Examples overview
│   ├── ex1.md                  #   Square lattice
│   ├── ex2.md                  #   Two layers
│   ├── ex3.md                  #   Optimization
│   └── ex4.md                  #   Hexagonal lattice
├── advanced/                   # Advanced topics
│   ├── autograd.md             #   Automatic differentiation
│   ├── gradients.md            #   Gradient computation
│   └── optimization.md         #   Optimization methods
├── reference/                  # Reference materials
│   ├── variables.md            #   Variables & conventions
│   ├── units.md                #   Physical units
│   ├── troubleshooting.md      #   Troubleshooting guide
│   └── faq.md                  #   FAQ
├── javascripts/                # JavaScript for MathJax
│   └── mathjax.js
└── stylesheets/                # Custom CSS
    └── extra.css
```

## Building the Documentation

### Prerequisites

Install MkDocs and required plugins:

```bash
pip install mkdocs-material
pip install mkdocstrings[python]
pip install pymdown-extensions
```

### Local Development

To view the documentation locally:

```bash
# From the repository root
mkdocs serve
```

Then open your browser to `http://127.0.0.1:8000`

The documentation will auto-reload when you make changes.

### Building Static Site

To build the static HTML files:

```bash
mkdocs build
```

The built site will be in the `site/` directory.

### Deploy to GitHub Pages

The documentation is automatically deployed to GitHub Pages via GitHub Actions when pushing to the main branch.

To manually deploy:

```bash
mkdocs gh-deploy
```

This will build the documentation and push to the `gh-pages` branch.

## Documentation Features

### MkDocs Material Theme

Beautiful, responsive documentation theme with:

- Light/dark mode toggle
- Search functionality
- Navigation tabs and sections
- Code syntax highlighting
- Mobile-friendly design

### Mathematical Equations

Full LaTeX support via MathJax:

```markdown
Inline equation: $E = mc^2$

Display equation:
$$
\nabla \times \mathbf{E} = -\frac{\partial \mathbf{B}}{\partial t}
$$
```

### Code Blocks

Syntax highlighted code with copy button:

````markdown
```python
import grcwa
obj = grcwa.obj(nG=101, L1=[1,0], L2=[0,1], freq=1.0, theta=0, phi=0)
```
````

### Admonitions

Callout boxes for notes, warnings, etc.:

```markdown
!!! note "Important Note"
    This is an important note.

!!! warning "Warning"
    Be careful with this.

!!! tip "Pro Tip"
    Here's a helpful tip.
```

### Tabbed Content

```markdown
=== "Python"
    ```python
    print("Hello")
    ```

=== "Output"
    ```
    Hello
    ```
```

## Contributing to Documentation

### Adding New Pages

1. Create a new `.md` file in the appropriate directory
2. Add the page to `mkdocs.yml` under the `nav` section
3. Use clear headings and structure
4. Include code examples and equations where helpful

### Writing Guidelines

- **Clear and concise**: Write for beginners who don't know RCWA
- **Code examples**: Provide working, complete examples
- **Equations**: Explain the physics and mathematics
- **Cross-references**: Link to related pages
- **Visual aids**: Include diagrams where helpful

### Markdown Best Practices

- Use ATX-style headers (`#`, `##`, `###`)
- Code blocks should specify language (` ```python `)
- Use tables for structured data
- Include alt text for images
- Use inline code for variable names (`` `epsilon` ``)

## Documentation Goals

This documentation aims to:

1. **Introduce RCWA** to researchers unfamiliar with the method
2. **Explain the physics** behind electromagnetic simulations
3. **Document all functions** with clear descriptions
4. **Provide tutorials** for common use cases
5. **Enable quick start** for new users
6. **Support advanced users** with optimization and gradients

## Target Audience

- Graduate students learning computational photonics
- Researchers designing photonic devices
- Engineers optimizing optical components
- Educators teaching electromagnetics
- Anyone interested in RCWA simulations

## Key Documentation Pages

### For Beginners
- [Introduction](docs_mkdocs/introduction.md): What is RCWA?
- [Quick Start](docs_mkdocs/quickstart.md): First simulation in 5 minutes
- [Tutorial 1](docs_mkdocs/tutorials/tutorial1.md): Step-by-step example

### For Understanding Theory
- [Physical Principles](docs_mkdocs/theory/principles.md): Maxwell's equations, Bloch waves
- [Mathematics](docs_mkdocs/theory/mathematics.md): Detailed derivations
- [Algorithm](docs_mkdocs/theory/algorithm.md): Implementation details

### For Practical Use
- [Basic Concepts](docs_mkdocs/guide/concepts.md): Workflow, layers, truncation
- [API Reference](docs_mkdocs/api/core.md): Complete function documentation
- [Examples](docs_mkdocs/examples/gallery.md): Gallery of simulations

### For Troubleshooting
- [Variables](docs_mkdocs/reference/variables.md): All variables explained
- [FAQ](docs_mkdocs/reference/faq.md): Common questions
- [Troubleshooting](docs_mkdocs/reference/troubleshooting.md): Fixing issues

## Viewing Online

Once deployed, the documentation will be available at:

**https://[username].github.io/grcwa/**

(Replace with actual GitHub Pages URL)

## Updating Documentation

To update the documentation:

1. Edit the relevant `.md` files
2. Test locally with `mkdocs serve`
3. Commit and push to the repository
4. GitHub Actions will automatically deploy

## Maintenance

### Regular Updates
- Keep examples up-to-date with latest GRCWA version
- Add new tutorials as use cases emerge
- Update API documentation when functions change
- Fix any broken links or outdated information

### Community Contributions
- Accept pull requests for documentation improvements
- Respond to issues about unclear documentation
- Add FAQ entries for commonly asked questions

## License

The documentation is part of the GRCWA project and follows the same GPL v3 license.

## Contact

For questions about the documentation:

- Open an issue on [GitHub](https://github.com/weiliangjinca/grcwa/issues)
- Email: jwlaaa@gmail.com

---

**Note:** This documentation was created to make RCWA accessible to researchers who may not be familiar with the method, providing both theoretical background and practical guidance for using the GRCWA library effectively.
