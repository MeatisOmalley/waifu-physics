"""Build the extension zip: compile the C step (Windows), fetch it for Linux, validate, package.

    python tools/release.py

Writes dist/waifu_physics-<version>.zip. The C step is compiled with the
Visual C++ build tools into waifu_physics/bin/waifu_physics_step.dll when its source
exists; on other platforms, or without the build tools, the package ships
without it and the numpy step is used. The Linux build (waifu_physics_step.so) comes from
the linux-step GitHub workflow, fetched with the GitHub CLI when its run includes the current
step.c; without it, Linux uses the numpy step. WAIFU_PHYSICS_BLENDER and WAIFU_PHYSICS_VCVARS
override the default Blender and vcvars64.bat locations.
"""
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKAGE = os.path.join(REPO, "waifu_physics")
BLENDER = os.environ.get("WAIFU_PHYSICS_BLENDER",
                         r"C:\Program Files (x86)\Steam\steamapps\common\Blender\blender.exe")
VCVARS = os.environ.get("WAIFU_PHYSICS_VCVARS", r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
                                        r"\VC\Auxiliary\Build\vcvars64.bat")
SOURCE = os.path.join(PACKAGE, "solver", "step.c")
DLL = os.path.join(PACKAGE, "bin", "waifu_physics_step.dll")
NATIVE_STEPS = ((SOURCE, DLL, "waifu_physics_step"),
                (os.path.join(PACKAGE, "cloth", "step.c"),
                 os.path.join(PACKAGE, "bin", "waifu_cloth_step.dll"), "waifu_cloth_step"))


def build_dll():
    if not os.path.exists(SOURCE):
        print("no C step yet (solver/step.c); packaging without a DLL")
        return True
    if sys.platform != "win32" or not os.path.exists(VCVARS):
        print("no Visual C++ build tools here; packaging without a DLL")
        return True
    os.makedirs(os.path.dirname(DLL), exist_ok=True)
    build_dir = os.path.join(REPO, "build")
    os.makedirs(build_dir, exist_ok=True)
    # Separate object paths: both libraries intentionally name their source step.c.
    # Precise floating arithmetic preserves the reference's operation order.
    for source, dll, name in NATIVE_STEPS:
        if not os.path.exists(source):
            continue
        command = (f'"{VCVARS}" >nul && cl /nologo /O2 /fp:precise /LD "{source}" '
                   f'/Fo"{build_dir}\\{name}.obj" /Fe"{dll}" /link /IMPLIB:"{build_dir}\\{name}.lib"')
        proc = subprocess.run(f'cmd /s /c "{command}"', shell=True, capture_output=True, text=True)
        print(proc.stdout.strip())
        if proc.returncode != 0 or not os.path.exists(dll):
            print(proc.stderr.strip())
            print("compiling", name, "failed")
            return False
        print("built", os.path.relpath(dll, REPO))
    return True


def fetch_linux():
    """The Linux C step from the latest successful linux-step workflow run, if that run includes the
    committed step.c (and step.c has no uncommitted changes). Packaging goes on without it otherwise."""
    libraries = [(source, os.path.join(PACKAGE, "bin", name+".so"), name)
                 for source, _dll, name in NATIVE_STEPS]

    def run(*args):
        return subprocess.run(args, cwd=REPO, capture_output=True, text=True)
    for _source, so, _name in libraries:
        if os.path.exists(so):
            os.remove(so)                                # never package a stale one
    found = run("gh", "run", "list", "--workflow", "linux-step.yml", "--status", "success", "--limit", "1",
                "--json", "databaseId,headSha", "-q", r'.[0] | "\(.databaseId) \(.headSha)"')
    if found.returncode != 0 or not found.stdout.strip():
        print("no Linux C step: the GitHub CLI or a successful linux-step run is missing")
        return
    run_id, head = found.stdout.split()
    for source, so, name in libraries:
        if run("git", "diff", "--quiet", "HEAD", "--", source).returncode != 0:
            print(name, "has uncommitted changes: no Linux library until the workflow builds it")
            continue
        changed = run("git", "log", "-1", "--format=%H", "--", source).stdout.strip()
        if not changed or run("git", "merge-base", "--is-ancestor", changed, head).returncode != 0:
            print("no Linux", name, ": the successful workflow predates its source")
            continue
        got = run("gh", "run", "download", run_id, "--name", name+"-linux-x64",
                  "--dir", os.path.join(PACKAGE, "bin"))
        print("fetched Linux "+name if got.returncode == 0 and os.path.exists(so)
              else "fetching Linux "+name+" failed: " + got.stderr.strip())


def blender(*args):
    proc = subprocess.run([BLENDER, "--command", "extension", *args], capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    print(proc.stdout.strip() or proc.stderr.strip())
    return proc.returncode == 0


def main(argv):
    if not build_dll():
        return 1
    if "--dll-only" in argv:
        return 0
    fetch_linux()
    if not blender("validate", PACKAGE):
        print("the manifest or package did not validate")
        return 1
    os.makedirs(os.path.join(REPO, "dist"), exist_ok=True)
    if not blender("build", "--source-dir", PACKAGE, "--output-dir", os.path.join(REPO, "dist")):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
