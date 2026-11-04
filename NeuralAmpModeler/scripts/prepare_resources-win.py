#!/usr/bin/env python3

import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile

scriptpath = os.path.dirname(os.path.realpath(__file__))
projectpath = os.path.abspath(os.path.join(scriptpath, os.pardir))
repositorypath = os.path.abspath(os.path.join(projectpath, os.pardir))

sys.path.insert(0, os.path.join(repositorypath, "iPlug2", "Scripts"))
sys.path.insert(0, scriptpath)

from parse_config import parse_config
from dessamp_release_licenses import LICENSE_BUNDLE_ENTRIES
from dessamp_factory_defaults import CABINET_HASHES


def write_text_if_changed(path, value):
    try:
        with open(path, "r", encoding="utf-8") as source:
            if source.read() == value:
                return False
    except FileNotFoundError:
        pass
    with open(path, "w", encoding="utf-8", newline="") as destination:
        destination.write(value)
    return True


def update_resource_text(source, config):
    numeric_version = ",".join(
        (config["MAJOR_STR"], config["MINOR_STR"], config["BUGFIX_STR"], "0")
    )
    source = re.sub(r"(?m)^(\s*FILEVERSION\s+).*$", rf"\g<1>{numeric_version}", source)
    source = re.sub(
        r"(?m)^(\s*PRODUCTVERSION\s+).*$", rf"\g<1>{numeric_version}", source
    )
    replacements = {
        "FileVersion": config["FULL_VER_STR"],
        "ProductVersion": config["FULL_VER_STR"],
        "FileDescription": config["PLUG_NAME"],
        "InternalName": config["PLUG_NAME"],
        "ProductName": config["PLUG_NAME"],
        "CompanyName": config["PLUG_MFR"],
        "LegalCopyright": config["PLUG_COPYRIGHT_STR"],
    }
    for key, value in replacements.items():
        source = re.sub(
            rf'(?m)^(\s*VALUE "{re.escape(key)}", ").*("\s*)$',
            rf"\g<1>{value}\g<2>",
            source,
        )
    return source


def prepare_distribution(root, build):
    # Run in a fresh, reviewed export. Verify bytes against its inventory
    # before making them installer inputs. The Apple descriptor-based stager
    # remains separate because Windows lacks POSIX no-follow directory FDs.
    inventory = {}
    for line in (root / "SOURCE-INVENTORY.sha256").read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        inventory[name] = digest
    contents = {}
    entries = []
    for source, name in LICENSE_BUNDLE_ENTRIES:
        path = root / source
        if path.resolve(strict=True) != path:
            raise ValueError(f"Redirected license source: {source}")
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if inventory.get(source) != digest:
            raise ValueError(f"License differs from reviewed inventory: {source}")
        contents[name] = data
        entries.append({"bundle_path": name, "sha256": digest, "source_repository_path": source})
    manifest = {"schema_version": 1, "files": entries}
    contents["LICENSE-MANIFEST.json"] = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    target = build / "THIRD_PARTY_LICENSES"
    target.mkdir(parents=True, exist_ok=True)
    if target.resolve() != target or any(
        p.name not in contents or p.resolve() != p or not p.is_file()
        for p in target.iterdir()
    ):
        raise ValueError("Unexpected entry in generated Windows license bundle")
    for name, data in contents.items():
        (target / name).write_bytes(data)


def write_installer_files(build, prefix, checked, destinations, include):
    stage = Path(tempfile.mkdtemp(prefix=prefix, dir=build))
    rows = []
    for source, folder in destinations.items():
        name = Path(source).name
        (stage / name).write_bytes(checked[source])
        src = f"..\\build-win\\{stage.name}\\{name}"
        for component, resources in (("app", r"{app}\Resources"),
                                     ("vst3_64", r"{commoncf64}\VST3\DessAmp.vst3\Contents\Resources")):
            dest = resources + "\\" + folder.replace("/", "\\")
            rows.append(f'Source: "{src}"; DestDir: "{dest}"; Components: {component}; Flags: ignoreversion')
    include.write_text("\n".join(rows) + "\n", encoding="utf-8")


def prepare_factory_cabinets(root, build):
    include = build / "factory-cabinets.iss"
    include.unlink(missing_ok=True)
    bank = root / "NeuralAmpModeler/resources/FactoryBank"
    if not bank.exists():
        include.write_text("", encoding="utf-8")  # Source-only distribution.
        return
    checked = {}
    for name, expected in CABINET_HASHES.items():
        path = bank / name
        if path.resolve(strict=True) != path:
            raise ValueError(f"Redirected cabinet source: {name}")
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"Version C cabinet hash mismatch: {name}")
        checked[name] = data
    write_installer_files(build, "factory-", checked, {name: "IRs" for name in checked}, include)


def prepare_private_listening(root, ledger, expected_digest, build):
    """Materialize only the fixed factory files named in the reviewed ledger."""
    include = build / "listening-files.iss"
    include.unlink(missing_ok=True)
    raw = ledger.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_digest:
        raise ValueError("Private listening input ledger digest mismatch")
    entries = json.loads(raw)
    destinations = {
        **{f"Amplifiers/amp-{i:02d}.dess": "PrivateValidation/Amplifiers" for i in range(1, 5)},
        **{f"IRs/{name}": "IRs" for name in CABINET_HASHES},
        "Defaults/FactoryDefaults.json": "PrivateValidation",
        **{f"Drives/{name}.nam": "models/DessDrive" for name in ("OD808", "SD1", "TS9", "aesahaettr")},
    }
    if set(entries) != set(destinations):
        raise ValueError("Private listening input ledger does not describe the exact factory bank")
    root = root.resolve(strict=True)
    checked = {}
    for source in destinations:
        path = root / source
        if path.resolve(strict=True) != path or not 0 < path.stat().st_size <= 64 * 1024 * 1024:
            raise ValueError(f"Invalid private listening input: {source}")
        data = path.read_bytes()
        if len(data) != entries[source]["bytes"] or hashlib.sha256(data).hexdigest() != entries[source]["sha256"]:
            raise ValueError(f"Private listening input differs from its ledger: {source}")
        if source.startswith("IRs/") and entries[source]["sha256"] != CABINET_HASHES[path.name]:
            raise ValueError(f"Version C cabinet hash mismatch: {source}")
        checked[source] = data
    if json.loads(checked["Defaults/FactoryDefaults.json"]).get("ir_sha256") != CABINET_HASHES["Desfire.wav"]:
        raise ValueError("Factory profile does not match the Version C bank")
    write_installer_files(build, "listening-", checked, destinations, include)


def main():
    if sys.argv[1:2] == ["--distribution"]:
        build = Path(projectpath).resolve() / "build-win"
        prepare_distribution(Path(repositorypath).resolve(), build)
        prepare_factory_cabinets(Path(repositorypath).resolve(), build)
        if sys.argv[2:] == ["listening"]:
            prepare_private_listening(Path(os.environ["DESSAMP_WINDOWS_BANK"]),
                                      Path(os.environ["DESSAMP_WINDOWS_BANK_LEDGER"]),
                                      os.environ["DESSAMP_WINDOWS_BANK_SHA256"], build)
        elif sys.argv[2:] not in (["full"], ["demo"], []):
            raise ValueError("Unknown Windows distribution type")
    config = parse_config(projectpath)
    resource_path = os.path.join(projectpath, "resources", "main.rc")
    with open(resource_path, "r", encoding="utf-8") as source:
        updated = update_resource_text(source.read(), config)
    write_text_if_changed(resource_path, updated)
    print("Synchronized Windows resources with config.h")


if __name__ == "__main__":
    main()
