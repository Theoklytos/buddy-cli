"""Configuration management for Buddy CLI chat interface."""

import copy
import os
import shutil
import tempfile
from pathlib import Path

import yaml
from rich.console import Console
from rich.prompt import Prompt


CONFIG_DIR = Path.home() / ".config" / "bud"
BUD_CONFIG_FILE = CONFIG_DIR / "config.yaml"
BUDDY_CONFIG_FILE = CONFIG_DIR / "buddy.yaml"
BUDDY_PROFILES_DIR = CONFIG_DIR / "buddy_profiles"
BUDDY_LOGS_DIR = CONFIG_DIR / "buddy_logs"

DEFAULT_BUDDY_CONFIG = {
    "model": "claude-sonnet-4-20250514",
    "system_prompt": "You are a helpful assistant.",
    "temperature": None,
    "max_tokens": 4096,
    "context_depth": 5,
    "api_key": None,
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge two dicts. Override values win over base values.

    For nested dicts, merging is applied recursively.
    For all other types, the override value replaces the base value.
    """
    result = copy.deepcopy(base)
    for key, override_val in override.items():
        base_val = result.get(key)
        if isinstance(override_val, dict) and isinstance(base_val, dict):
            result[key] = _deep_merge(base_val, override_val)
        else:
            result[key] = override_val
    return result


def load_buddy_config() -> dict:
    """Load buddy config from BUDDY_CONFIG_FILE, deep merged with defaults.

    Returns the default config if the file does not exist or has YAML errors.
    """
    config = copy.deepcopy(DEFAULT_BUDDY_CONFIG)

    if not BUDDY_CONFIG_FILE.exists():
        return config

    try:
        with open(BUDDY_CONFIG_FILE, "r") as f:
            file_config = yaml.safe_load(f) or {}
    except yaml.YAMLError:
        return config

    return _deep_merge(config, file_config)


def save_buddy_config(config: dict) -> Path:
    """Save buddy config to BUDDY_CONFIG_FILE using atomic write.

    Creates CONFIG_DIR if it does not exist. Returns the path written to.
    """
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, dir=CONFIG_DIR, suffix=".yaml"
    ) as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        temp_path = f.name

    shutil.move(temp_path, BUDDY_CONFIG_FILE)
    return BUDDY_CONFIG_FILE


def resolve_api_key(buddy_config: dict) -> str | None:
    """Resolve the Anthropic API key using a 3-layer priority.

    Priority 1: buddy_config["api_key"] if set (not None/empty).
    Priority 2: ANTHROPIC_API_KEY environment variable.
    Priority 3: bud's config.yaml llm.api_key field.

    Returns None if all sources fail.
    """
    # Priority 1: explicit value in buddy config
    api_key = buddy_config.get("api_key")
    if api_key:
        return api_key

    # Priority 2: environment variable
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        return api_key

    # Priority 3: bud's config.yaml
    try:
        if BUD_CONFIG_FILE.exists():
            with open(BUD_CONFIG_FILE, "r") as f:
                bud_config = yaml.safe_load(f) or {}
            api_key = bud_config.get("llm", {}).get("api_key")
            if api_key:
                return api_key
    except (yaml.YAMLError, OSError):
        pass

    return None


def run_config_wizard() -> dict:
    """Interactive first-run setup wizard using Rich Console.

    Prompts the user for configuration values, saves them, and returns the
    resulting config dict.
    """
    console = Console()
    config = copy.deepcopy(DEFAULT_BUDDY_CONFIG)

    console.print("\n[bold cyan]Buddy CLI — first-run setup[/bold cyan]\n")

    # Attempt to import API key from bud's config
    imported_api_key = None
    if BUD_CONFIG_FILE.exists():
        try:
            with open(BUD_CONFIG_FILE, "r") as f:
                bud_cfg = yaml.safe_load(f) or {}
            imported_api_key = bud_cfg.get("llm", {}).get("api_key")
            if imported_api_key:
                console.print(
                    "[green]Found API key in bud's config.yaml — will use it.[/green]"
                )
                config["api_key"] = imported_api_key
        except (yaml.YAMLError, OSError):
            pass

    # Model
    model = Prompt.ask(
        "Model",
        default=DEFAULT_BUDDY_CONFIG["model"],
        console=console,
    )
    config["model"] = model.strip() or DEFAULT_BUDDY_CONFIG["model"]

    # System prompt
    system_prompt = Prompt.ask(
        "System prompt",
        default=DEFAULT_BUDDY_CONFIG["system_prompt"],
        console=console,
    )
    config["system_prompt"] = system_prompt.strip() or DEFAULT_BUDDY_CONFIG["system_prompt"]

    # Temperature
    temp_input = Prompt.ask(
        "Temperature (leave blank for auto)",
        default="",
        console=console,
    )
    if temp_input.strip():
        try:
            config["temperature"] = float(temp_input.strip())
        except ValueError:
            console.print("[yellow]Invalid temperature — using auto (None).[/yellow]")
            config["temperature"] = None
    else:
        config["temperature"] = None

    # Context depth
    depth_input = Prompt.ask(
        "Context depth (number of previous messages to include)",
        default=str(DEFAULT_BUDDY_CONFIG["context_depth"]),
        console=console,
    )
    try:
        config["context_depth"] = int(depth_input.strip())
    except ValueError:
        console.print(
            f"[yellow]Invalid context depth — using default "
            f"({DEFAULT_BUDDY_CONFIG['context_depth']}).[/yellow]"
        )
        config["context_depth"] = DEFAULT_BUDDY_CONFIG["context_depth"]

    save_buddy_config(config)
    console.print("\n[green]Configuration saved.[/green]\n")
    return config


def list_buddy_profiles() -> list[str]:
    """Return a sorted list of profile names (YAML files without extension) in BUDDY_PROFILES_DIR."""
    if not BUDDY_PROFILES_DIR.exists():
        return []
    return sorted(p.stem for p in BUDDY_PROFILES_DIR.glob("*.yaml"))


def load_buddy_profile(name: str) -> dict:
    """Load a named buddy profile from BUDDY_PROFILES_DIR.

    Raises FileNotFoundError if the profile YAML does not exist.
    """
    profile_path = BUDDY_PROFILES_DIR / f"{name}.yaml"
    if not profile_path.exists():
        raise FileNotFoundError(f"Buddy profile not found: {name!r} ({profile_path})")
    with open(profile_path, "r") as f:
        return yaml.safe_load(f) or {}


def save_buddy_profile(name: str, data: dict) -> Path:
    """Save data to BUDDY_PROFILES_DIR/<name>.yaml.

    Creates the profiles directory if it does not exist. Returns the path written to.
    """
    BUDDY_PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    profile_path = BUDDY_PROFILES_DIR / f"{name}.yaml"

    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, dir=BUDDY_PROFILES_DIR, suffix=".yaml"
    ) as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        temp_path = f.name

    shutil.move(temp_path, profile_path)
    return profile_path
