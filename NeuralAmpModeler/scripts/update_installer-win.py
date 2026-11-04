"""Generate Inno metadata in the build directory from config.h."""
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT.parent / "iPlug2/Scripts"))
from parse_config import parse_config


def main():
    mode, = sys.argv[1:]
    mode = {"0": "full", "1": "demo"}.get(mode, mode)
    title = {"full": "DessAmp Installer", "demo": "DessAmp Demo Installer",
             "listening": "DessAmp Private Listening Installer"}[mode]
    readme = "readme-win-demo.rtf" if mode == "demo" else "readme-win.rtf"
    version = parse_config(str(PROJECT))["FULL_VER_STR"]
    settings = (PROJECT / 'installer/DessAmp.iss').read_text(encoding='utf-8').splitlines()
    if any(f'{key}={version}' not in settings for key in ('AppVersion', 'VersionInfoVersion')):
        raise ValueError('Installer template version differs from config.h')
    lines = [f'#define InstallerName "{title}"',
             f'#define ReadmeName "{readme}"']
    if mode == "listening":
        lines.append("#define PrivateListening")
    build = PROJECT / "build-win"
    build.mkdir(exist_ok=True)
    (build / "installer-settings.iss").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
