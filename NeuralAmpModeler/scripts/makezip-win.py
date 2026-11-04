"""Archive Windows products from the current x64 Release build."""
from pathlib import Path
import sys
import zipfile

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT.parent / "iPlug2/Scripts"))
from get_archive_name import get_archive_name


def main():
    mode, zip_arg = sys.argv[1:]
    mode = {"0": "full", "1": "demo"}.get(mode, mode)
    zip_only = int(zip_arg)
    if mode not in ("full", "demo", "listening") or (mode == "listening" and zip_only):
        raise ValueError("Unsupported Windows distribution mode")
    build = PROJECT / "build-win"
    name = get_archive_name(str(PROJECT), "win", "demo" if mode == "demo" else "full")
    if mode == "listening":
        name += "-private-listening"
    out = build / "out"
    out.mkdir(exist_ok=True)
    if zip_only:
        files = [
            build / "DessAmp_x64.exe",
            build / "DessAmp.vst3/Contents/x86_64-win/DessAmp.vst3",
            *sorted((build / "THIRD_PARTY_LICENSES").iterdir()),
        ]
    else:
        files = [
            build / "installer" / {"demo": "DessAmp Demo Installer.exe", "listening": "DessAmp Private Listening Installer.exe", "full": "DessAmp Installer.exe"}[mode],
            PROJECT / "installer/known-issues.txt",
        ]
    pdbs = [build / "pdbs" / f"NeuralAmpModeler-{target}_x64.pdb" for target in ("app", "vst3")]
    for path in files + pdbs:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    for suffix, members in (("", files), ("-pdbs", pdbs)):
        path = out / (name + suffix + ".zip")
        with zipfile.ZipFile(path.with_suffix(".tmp"), "w", zipfile.ZIP_DEFLATED) as archive:
            for member in members:
                arcname = member.relative_to(build) if zip_only and not suffix else member.name
                archive.write(member, arcname)
        path.with_suffix(".tmp").replace(path)
        print(path)


if __name__ == "__main__":
    main()
