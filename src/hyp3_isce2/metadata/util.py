from jinja2 import Environment, PackageLoader, StrictUndefined, select_autoescape


def get_environment() -> Environment:
    env = Environment(
        loader=PackageLoader("hyp3_isce2.metadata", "templates"),
        autoescape=select_autoescape(["html.j2", "xml.j2"]),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )
    return env


def render_template(template: str, payload: dict) -> str:
    env = get_environment()
    template = env.get_template(template)
    rendered = template.render(payload)
    return rendered
