"""Write recipe.yaml and render report.md."""

from __future__ import annotations

from pathlib import Path

import yaml
from jinja2 import Environment, PackageLoader, select_autoescape

from infervolt.recipes.schema import Recipe


def yamlish(value: object) -> object:
    """Render booleans the way YAML and engine CLIs spell them; leave everything else alone."""
    if value is True:
        return "true"
    if value is False:
        return "false"
    return value


_env = Environment(
    # Autoescape is intentionally off: these templates render Markdown, not HTML, and
    # HTML-escaping would mangle model ids, CLI flags and quoted knob values.
    loader=PackageLoader("infervolt.recipes", "templates"),
    autoescape=select_autoescape(default=False),
    trim_blocks=True,
    lstrip_blocks=True,
)
_env.filters["yamlish"] = yamlish


def write_recipe(recipe: Recipe, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "recipe.yaml"
    path.write_text(yaml.safe_dump(recipe.model_dump(mode="json"), sort_keys=False, width=100))
    return path


def render_report(recipe: Recipe) -> str:
    return _env.get_template("report.md.j2").render(r=recipe)


def write_report(recipe: Recipe, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "report.md"
    path.write_text(render_report(recipe))
    return path
